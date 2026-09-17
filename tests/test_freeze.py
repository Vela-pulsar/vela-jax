"""Ingest: what gets stripped, what gets refused, what gets frozen."""

import numpy as np
import pytest

from vela_jax.errors import FreezeError, UnsupportedModelError
from vela_jax.freeze import is_noise_line, strip_noise_lines

NOISE = [
    "EFAC -f L-wide_ASP 1.0",
    "T2EQUAD -f L-wide 0.1",
    "ECORR -f L-wide 0.5",
    "TNRedAmp -13.5",
    "TNDMGam 2.0",
    "PLREDFREQ 1.0",
    "RNAMP 1e-14",
    "CHI2 1234.5",
]
DELAY = [
    "F0 61.485476554371304592 1 1.7e-11",
    "JUMP -fe Rcvr_800 0.1 1",
    "DMJUMP -fe Rcvr 1e-3",
    "DM 160.0 1",
    "ELAT 5.0 1",
    "FD1 0.001 1",
    "TZRMJD 55000.0",
]


@pytest.mark.unit
@pytest.mark.parametrize("line", NOISE)
def test_noise_lines_are_recognised(line):
    assert is_noise_line(line)


@pytest.mark.unit
@pytest.mark.parametrize("line", DELAY)
def test_delay_lines_survive(line):
    assert not is_noise_line(line)


@pytest.mark.unit
def test_strip_keeps_order_and_ends_with_a_newline():
    text = "F0 1.0\nEFAC -f x 1.0\nDM 2.0\n"
    assert strip_noise_lines(text) == "F0 1.0\nDM 2.0\n"


@pytest.mark.unit
def test_strip_does_not_touch_the_caller_file(tmp_path, examples):
    par = examples / "NGC6440E.par"
    before = par.read_bytes()
    strip_noise_lines(par.read_text())
    assert par.read_bytes() == before


def _delay_only_model(pint, examples, *, planets):
    """A raw PINT pair from the stripped par, bypassing ``load_pint``."""
    import tempfile
    from pathlib import Path

    par = examples / "NGC6440E.par"
    stripped = Path(tempfile.mkdtemp()) / par.name
    stripped.write_text(strip_noise_lines(par.read_text()))
    return pint.models.get_model_and_toas(
        str(stripped),
        str(examples / "NGC6440E.tim"),
        planets=planets,
        add_tzr_to_model=True,
    )


def test_frozen_rows_are_toas_plus_tzr(engine_factory):
    engine = engine_factory("NGC6440E")
    assert engine.frozen.n_rows == engine.toa_count + 1
    assert bool(engine.frozen.is_tzr[-1])
    assert not bool(np.any(np.asarray(engine.frozen.is_tzr)[:-1]))


def test_missing_pulse_numbers_are_refused(examples):
    import pint.models

    from vela_jax import Engine
    from vela_jax.freeze import prepare_model

    model, toas = _delay_only_model(pint, examples, planets=True)
    prepare_model(model, toas)
    with pytest.raises(FreezeError, match="pulse numbers"):
        Engine(model, toas)


def test_planets_are_required(examples):
    import pint.models

    from vela_jax import Engine

    model, toas = _delay_only_model(pint, examples, planets=False)
    with pytest.raises(FreezeError, match="planets"):
        Engine.from_pint(model, toas)


@pytest.mark.parametrize("name", ["sim3", "sim4", "J0613-0200.sim"])
def test_unsupported_models_are_refused_by_name(examples, name):
    from vela_jax import Engine

    par, tim = examples / f"{name}.par", examples / f"{name}.tim"
    if not par.exists():
        pytest.skip(f"fixture {name} not available")
    with pytest.raises(UnsupportedModelError):
        Engine.from_files(par, tim)


def test_noise_lines_do_not_reach_the_residual(examples, tmp_path):
    """T11. Stripping is not cosmetic: the residual must be bit-identical to
    the one from a par that never had the lines, and PINT must not have built
    a noise component that would permute the TOAs."""
    from vela_jax import Engine

    par = examples / "NGC6440E.par"
    original = par.read_text()
    assert any(
        is_noise_line(line) for line in original.splitlines()
    ), "fixture no longer carries noise lines; pick another"

    hand_stripped = tmp_path / par.name
    hand_stripped.write_text(strip_noise_lines(original))
    tim = examples / "NGC6440E.tim"

    with_noise = Engine.from_files(par, tim)
    without = Engine.from_files(hand_stripped, tim)
    assert np.array_equal(with_noise.residuals(), without.residuals())
    assert with_noise.param_names == without.param_names
    assert "ScaleToaError" not in with_noise.pint_model.components


# --- parameter accountability (A1) ----------------------------------------

#: The cheapest ecliptic fixture that sets ``ECL``: 500 TOAs against
#: ``sim_sw``'s 2000, and these tests build a fresh engine per edit.
ECLIPTIC = "J2302+4442.sim"


def _edited(examples, tmp_path, name, *, drop=(), add=()):
    """A fixture par with some lines removed and some appended."""
    dropped = {d.upper() for d in drop}
    keep = [
        line
        for line in strip_noise_lines(
            (examples / f"{name}.par").read_text()
        ).splitlines()
        if not (line.split() and line.split()[0].upper() in dropped)
    ]
    par = tmp_path / f"{name}.par"
    par.write_text("\n".join(keep + list(add)) + "\n")
    return par


@pytest.mark.parametrize(
    "name,parameter,value",
    [
        ("sim_dd", "A0", "1e-5"),  # DD aberration; PINT moves 14 us
        ("sim_dd", "B0", "1e-5"),  # the other aberration term
        ("J2302+4442.sim", "SWM", "1"),  # You+2007; Vela has only the spherical model
    ],
)
def test_a_parameter_no_component_evaluates_is_refused(
    examples, tmp_path, name, parameter, value
):
    """A frozen parameter PINT honours must never be dropped in silence.

    ``validate_model`` allow-lists components and ``Engine`` refuses *free*
    parameters no stage reads; between them sat a gap that swallowed frozen
    ones. Measured against PINT's own residuals on these very fixtures:
    ``A0 = 1e-5`` moves PINT by 14 microseconds and moved this engine by
    nothing at all. ``sim_dd`` sets ``A0``/``B0`` at zero, which the rule
    skips; a non-zero unimplemented term is what parity could never see.
    """
    from vela_jax import Engine

    tim = examples / f"{name}.tim"
    Engine.from_files(_edited(examples, tmp_path, name), tim)  # builds as before
    with pytest.raises(UnsupportedModelError, match=parameter):
        Engine.from_files(
            _edited(
                examples,
                tmp_path,
                name,
                drop=(parameter,),
                add=(f"{parameter} {value}",),
            ),
            tim,
        )


@pytest.mark.unit
def test_dm_derivatives_are_not_unconditionally_inert():
    """A set DM2 that the series does not pack must be refused, not ignored.

    They used to sit on ``INERT_PARAMS`` with a comment "only when the DM
    series consumes them"; the consumed check never ran because inert wins.
    """
    from vela_jax.freeze import INERT_PARAMS

    assert "DM1" not in INERT_PARAMS
    assert "DM2" not in INERT_PARAMS


@pytest.mark.unit
def test_a_false_bool_is_not_a_zero_delay():
    """``float(False) == 0.0`` must not punch a hole in the offender check."""
    from vela_jax.freeze import _is_zero

    class _Param:
        def __init__(self, value):
            self.value = value

    assert _is_zero(_Param(False)) is False
    assert _is_zero(_Param("N")) is False
    assert _is_zero(_Param(0.0)) is True
    assert _is_zero(_Param(None)) is True


@pytest.mark.unit
def test_uses_fbx_refuses_a_binary_par_with_neither_chart():
    """The engine-side backstop. PINT refuses this first, at `validate`.

    A stub, because a real PINT model cannot reach it: `check_required_params`
    demands PB (or the FBX bridge) before this code runs. Everything the real
    ingest path does is covered by the FBX test below -- which is the point:
    this stub used to be the *only* FBX test, and it kept passing throughout
    the outage that test records.
    """
    from vela_jax.binary import uses_fbx

    with pytest.raises(UnsupportedModelError, match="neither PB nor FB0"):
        uses_fbx({})


def _fbx_twin(examples, tmp_path, name):
    """`name`'s par with `PB` replaced by the equivalent `FB0`."""
    text = (examples / f"{name}.par").read_text()
    (pb_line,) = [line for line in text.splitlines() if line.startswith("PB ")]
    fb0 = 1.0 / (float(pb_line.split()[1]) * 86400.0)
    return _edited(examples, tmp_path, name, drop=("PB",), add=(f"FB0 {fb0!r}",))


def test_fbx_is_read_from_fb0_not_from_a_pb_xor(examples, tmp_path):
    """An FBX par builds, and agrees with its PB twin.

    `PulsarBinary._canonicalize_fbx_views` installs `PB` as a `funcParameter`
    view of `FB0` whenever any `FBn` is set, so "exactly one of PB and FB0 is
    set" refused every FBX par on every family. Nothing caught it: no fixture
    here uses the FBX chart, and the only `uses_fbx` test built a dict rather
    than a PINT model.

    Both charts describe the same orbit, so the residuals must agree to far
    better than the engine's budget: measured 3.7e-14 s on a 3 us scale.
    """
    from pint.models.parameter import funcParameter

    from vela_jax import Engine
    from vela_jax.binary import uses_fbx

    pb_engine = Engine.from_files(examples / "sim_dd.par", examples / "sim_dd.tim")
    fbx_engine = Engine.from_files(
        _fbx_twin(examples, tmp_path, "sim_dd"), examples / "sim_dd.tim"
    )

    # PINT's derived view is present and populated: the shape that broke the XOR.
    model = fbx_engine.pint_model
    assert isinstance(model["PB"], funcParameter)
    assert model["PB"].quantity is not None
    assert model["FB0"].quantity is not None

    assert uses_fbx(model) is True
    assert fbx_engine.chain.use_fbx is True
    assert "binary.DD" in fbx_engine.stages

    difference = np.abs(fbx_engine.residuals() - pb_engine.residuals())
    assert difference.max() < 1e-12


def test_the_zero_value_of_an_unconsumed_parameter_is_inert(examples, tmp_path):
    """``A0 0`` is not a lie: every one of these enters its delay additively."""
    from vela_jax import Engine

    engine = Engine.from_files(
        _edited(
            examples, tmp_path, "sim_dd", drop=("A0", "B0"), add=("A0 0.0", "B0 0.0")
        ),
        examples / "sim_dd.tim",
    )
    assert engine.toa_count > 0


def test_wideband_toas_are_refused_rather_than_narrowed(examples):
    """SPEC 1.2 refuses wideband; v1 built the par and returned narrowband
    residuals, which is the same class of silence.

    The wideband *par* also carries ``DispersionJump`` (DMJUMP), which the
    component allow-list refuses first. Pair the supported delay par with
    the wideband tim so this hits the TOA gate, not that earlier one.
    """
    from vela_jax import Engine

    tim = examples / "sim_sw.wb.tim"
    if not tim.exists():
        pytest.skip("no wideband fixture")
    with pytest.raises(UnsupportedModelError, match="wideband TOAs"):
        Engine.from_files(examples / "sim_sw.par", tim)


def test_a_wideband_par_is_refused_as_dispersionjump(examples):
    """``sim_sw.wb.par`` is refused for DMJUMP, not silently narrowed."""
    from vela_jax import Engine

    par = examples / "sim_sw.wb.par"
    if not par.exists():
        pytest.skip("no wideband fixture")
    with pytest.raises(UnsupportedModelError, match="DispersionJump"):
        Engine.from_files(par, examples / "sim_sw.wb.tim")


@pytest.mark.parametrize(
    "ecl,arcsec",
    [("IERS2010", 84381.406), ("IERS2003", 84381.4059), ("IAU1976", 84381.448)],
)
def test_the_ecliptic_obliquity_comes_from_the_par(examples, tmp_path, ecl, arcsec):
    """``ECL`` selects the frame ELONG/ELAT were defined in; read it.

    Vela hard-codes IERS2010 and every Vela fixture that sets ``ECL`` agrees
    with it, which is why this needed a constructed test. On real data it is
    the common case: EPTA and IPTA release pars typically say
    ``ECL IERS2003``, and that is ~100 ns RMS of Roemer delay away.
    """
    import numpy as np
    from pint.models import get_model

    from vela_jax.constants import obliquity_radians

    # ECL is a par keyword. Engine.from_files would ingest the TIM and download
    # BIPM2023; pytest-xdist races that fetch on a cold cache.
    par = _edited(examples, tmp_path, ECLIPTIC, drop=("ECL",), add=(f"ECL {ecl}",))
    model = get_model(str(par))
    assert obliquity_radians(model["ECL"].value) == pytest.approx(
        np.deg2rad(arcsec / 3600.0), rel=0, abs=1e-15
    )


def test_an_unknown_obliquity_is_refused(examples, tmp_path):
    """PINT gets there first, and that is the right answer: it refuses the
    name while building the model, so this engine never sees a par whose
    ``ECL`` it could not resolve. The test pins that it *is* refused, by
    whichever of the two notices -- not silently defaulted."""
    from vela_jax import Engine

    with pytest.raises(Exception, match="IERS1066"):
        Engine.from_files(
            _edited(examples, tmp_path, ECLIPTIC, drop=("ECL",), add=("ECL IERS1066",)),
            examples / f"{ECLIPTIC}.tim",
        )


def test_changing_the_obliquity_moves_the_residual_like_pint(examples, tmp_path):
    """The fix, measured the way the bug was: against PINT on the same pair."""
    import numpy as np
    from pint.residuals import Residuals

    from vela_jax import Engine

    tim = examples / f"{ECLIPTIC}.tim"

    def pair(ecl):
        engine = Engine.from_files(
            _edited(examples, tmp_path, ECLIPTIC, drop=("ECL",), add=(f"ECL {ecl}",)),
            tim,
        )
        theirs = Residuals(
            engine.pint_toas, engine.pint_model, subtract_mean=False
        ).time_resids.to_value("s")
        ours = engine.residuals()
        return ours - ours.mean(), theirs - theirs.mean()

    ours_a, theirs_a = pair("IERS2010")
    ours_b, theirs_b = pair("IERS1992")
    moved_pint = np.max(np.abs(theirs_b - theirs_a))
    moved_ours = np.max(np.abs(ours_b - ours_a))
    assert moved_pint > 1e-6, "the fixture must actually move under PINT"
    assert abs(moved_ours - moved_pint) / moved_pint < 1e-3


def _dmx_index_oracle(model, toas, prefix="DMX_"):
    """The per-TOA ``find_prefix_bytime`` loop the vectorised finder replaced.

    Kept here rather than in the package: it is the reference this must agree
    with, and PINT is the one asked, so a disagreement is a real change of
    answer rather than a change of implementation.
    """
    from pint.utils import find_prefix_bytime

    index = np.zeros(len(toas), dtype=np.int32)
    for i in range(len(toas)):
        found = find_prefix_bytime(model, prefix, toas.table["mjd"][i])
        assert np.isscalar(found), f"TOA {i} is in {found} ranges"
        index[i] = int(found)
    return index


def test_the_dmx_window_finder_matches_pints_own(engine_factory):
    """Bit-identical to the oracle, on a real DMX par."""
    from vela_jax.freeze import dmx_index

    engine = engine_factory("sim_dmx")
    model, toas = engine.pint_model, engine.pint_toas
    mine = dmx_index(model, toas)
    assert np.array_equal(mine, _dmx_index_oracle(model, toas))
    assert mine.dtype == np.int32


def test_the_site_arrival_times_keep_their_longdouble(engine_factory):
    """The freeze's SAT is wider than what ``get_mjds()`` would have given.

    float64 days quantise an MJD near 55000 to ~0.6 us, which is three orders
    of magnitude above the engine's budget and is what every consumer reads
    back as ``stoas``.
    """
    from vela_jax.freeze import sat_mjd_longdouble

    engine = engine_factory("sim_dmx")
    exact = sat_mjd_longdouble(engine.pint_toas)
    assert exact.dtype == np.longdouble

    rounded = np.asarray(engine.pint_toas.get_mjds().value, dtype=np.longdouble)
    lost = np.max(np.abs(exact - rounded)) * 86400.0
    assert lost > 1e-9, "the float64 path was not actually lossy on this fixture"

    frozen = np.asarray(engine.toa_columns.sat_seconds, dtype=np.longdouble)
    assert np.array_equal(frozen, exact * np.longdouble(86400.0))


def _pint_time_resids(engine):
    from pint.residuals import Residuals

    return Residuals(
        engine.pint_toas, engine.pint_model, subtract_mean=False
    ).time_resids.to_value("s")


@pytest.mark.unit
def test_bare_dmx_is_inert_not_consumed():
    """The delay reads ``DMX_NNNN``; the bare name is bookkeeping.

    Putting it on ``Chain.consumed`` would also let a *free* ``DMX`` into
    the layout as a zero column. ``INERT_PARAMS`` accepts the frozen
    info line and still refuses it as a free parameter.
    """
    from vela_jax.freeze import INERT_PARAMS, PINNED_PARAMS

    assert "DMX" in INERT_PARAMS
    assert "DMX" not in PINNED_PARAMS
    assert "T2CMETHOD" in INERT_PARAMS
    assert "TIMEEPH" in INERT_PARAMS
    assert "T2CMETHOD" not in PINNED_PARAMS
    assert "TIMEEPH" not in PINNED_PARAMS


@pytest.mark.parametrize("dmx_value", ["0.5", "14.0"])
def test_bare_dmx_does_not_move_the_residual(examples, tmp_path, dmx_value):
    """Non-zero bare ``DMX`` is inert in PINT and in this engine.

    ``sim_dmx`` sets ``DMX 0.0``, which the zero-skip accepts, so the
    fixture suite could never see a NANOGrav-style value. NG9 writes
    ``DMX 14``; some later releases write ``0.5``. Both are durations
    on an info line PINT stores as ``pc / cm3`` and never differentiates.
    """
    from vela_jax import Engine
    from vela_jax.freeze import INERT_PARAMS

    tim = examples / "sim_dmx.tim"
    par = _edited(
        examples, tmp_path, "sim_dmx", drop=("DMX",), add=(f"DMX {dmx_value}",)
    )
    engine = Engine.from_files(par, tim)
    baseline = Engine.from_files(examples / "sim_dmx.par", tim)
    assert "DMX" not in engine.chain.consumed
    assert "DMX" in INERT_PARAMS
    assert np.array_equal(engine.residuals(), baseline.residuals())
    assert np.array_equal(_pint_time_resids(engine), _pint_time_resids(baseline))


def test_a_free_dmx_is_refused_as_unconsumed(examples, tmp_path):
    """A fitted bare ``DMX`` is not a delay axis; do not sample a zero column."""
    from vela_jax import Engine

    with pytest.raises(UnsupportedModelError, match="DMX"):
        Engine.from_files(
            _edited(examples, tmp_path, "sim_dmx", drop=("DMX",), add=("DMX 0.5 1",)),
            examples / "sim_dmx.tim",
        )


def test_dmdata_is_bookkeeping_not_a_delay(examples, tmp_path):
    """``DMDATA`` records whether a previous fit used per-TOA DMs.

    Wideband is a property of the TOAs and is refused separately. The flag
    itself does not enter the delay, including at Y.
    """
    from vela_jax import Engine
    from vela_jax.freeze import INERT_PARAMS

    assert "DMDATA" in INERT_PARAMS
    engine = Engine.from_files(
        _edited(examples, tmp_path, "sim_dmx", drop=("DMDATA",), add=("DMDATA Y",)),
        examples / "sim_dmx.tim",
    )
    assert engine.toa_count > 0


# --- DDR: build-time refusals and parameter accountability -----------------
#
# Mode flags, galaxy constants and the frozen ``TGEO`` epoch are all resolved
# once at build. Everything that can go out of domain while *sampling* is a
# traced mask instead and lives in ``tests/test_ddr.py``.

DDR_MODEL_PAR = """\
PSR              SIMDDRCHK
EPHEM            DE440
CLOCK            TT(BIPM2021)
UNITS            TDB
RAJ              18:00:00.00000000         0
DECJ             -20:00:00.0000000         0
PMRA             3.0                       0
PMDEC            -5.0                      0
PX               1.2                       0
F0               250.0                     1  1.0e-12
F1               -1.0e-15                  1  1.0e-22
PEPOCH           55000.0
POSEPOCH         55000.0
PLANET_SHAPIRO   N
BINARY           DDR
PB               1.0                       0
A1               5.0                       1  1.0e-7
TASC             55000.0                   1  1.0e-9
EPS1             0.02                      1  1.0e-7
EPS2             -0.03                     1  1.0e-7
M2               0.8                       1  1.0e-4
COSI             0.5                       1  1.0e-4
KOM              30.0                      1  1.0e-2
DDRPK            Y
DDRPBDOT         kinematic
DDRGEO           Y
DDRKINE          Y
TZRMJD           55000.0
TZRFRQ           1400.0
TZRSITE          gbt
"""


def _ddr_model(**mutations):
    """A valid geometry-on DDR model, mutated *after* PINT's own validation.

    PINT refuses most invalid flag combinations while parsing, so a text par
    cannot reach the engine's resolver for them. Mutating the parsed model is
    how the engine-side refusal gets exercised at all -- and it has to be
    exercised: PINT refusing first today is not a guarantee it will tomorrow.
    """
    import io

    from pint.models import get_model

    model = get_model(io.StringIO(DDR_MODEL_PAR))
    for name, value in mutations.items():
        if value is None:
            _unset(model, name)
        else:
            model[name].value = value
    return model


def _unset(model, name) -> None:
    """Clear a parameter the way a par that omitted it would have.

    Both PINT setters refuse to discard an existing quantity, and several of
    these fields cannot be left out of a *parseable* DDR par at all -- PINT's
    own validation demands them first. SPEC §11 still wants the engine-side
    refusal exercised, on the grounds that PINT refusing first today is not a
    guarantee about tomorrow, so the backing slot is cleared directly.
    """
    model[name]._quantity = None


def _resolve(model, *, use_fbx=False, ecliptic=False):
    from vela_jax.binary import resolve_ddr_config
    from vela_jax.constants import OBL

    return resolve_ddr_config(model, use_fbx=use_fbx, ecliptic=ecliptic, obliquity=OBL)


@pytest.mark.unit
def test_a_valid_ddr_model_resolves_its_six_static_flags():
    config = _resolve(_ddr_model())
    assert (config.use_fbx, config.use_pk) == (False, True)
    assert (config.pbdot_kinematic, config.use_geo, config.use_kine) == (
        True,
        True,
        True,
    )
    assert config.ecliptic_coordinates is False


@pytest.mark.unit
def test_an_unknown_ddrpbdot_mode_is_refused():
    with pytest.raises(UnsupportedModelError, match="DDRPBDOT"):
        _resolve(_ddr_model(DDRPBDOT="quadrupole"))


@pytest.mark.unit
@pytest.mark.parametrize("mutations", [{"DDRKINE": False}, {"DDRPBDOT": "kinematic"}])
def test_fbx_refuses_a_kinematic_or_shklovskii_pbdot(mutations):
    """The FBX chart encodes the whole phase in ``FBn``; a second source is a
    double count."""
    base = {"DDRPBDOT": "absorb_gw", "DDRKINE": False, "DDRGEO": False}
    base.update(mutations)
    if base["DDRPBDOT"] == "absorb_gw" and not base["DDRKINE"]:
        pytest.skip("this combination is the legal one")
    with pytest.raises(UnsupportedModelError, match="FBX"):
        _resolve(_ddr_model(**base), use_fbx=True)


@pytest.mark.unit
def test_fbx_with_a_kinematic_pbdot_is_refused():
    with pytest.raises(UnsupportedModelError, match="FBX"):
        _resolve(_ddr_model(DDRPBDOT="kinematic", DDRKINE=False), use_fbx=True)


@pytest.mark.unit
def test_fbx_with_kinematics_on_is_refused():
    with pytest.raises(UnsupportedModelError, match="FBX"):
        _resolve(_ddr_model(DDRPBDOT="absorb_gw", DDRKINE=True), use_fbx=True)


@pytest.mark.unit
def test_geometry_without_kinematics_is_refused_in_the_pb_chart():
    with pytest.raises(UnsupportedModelError, match="DDRGEO"):
        _resolve(_ddr_model(DDRGEO=True, DDRKINE=False))


@pytest.mark.unit
@pytest.mark.parametrize("px", [None, 0.0, -1.0])
def test_geometry_or_kinematics_without_a_positive_parallax_is_refused(px):
    model = _ddr_model()
    if px is None:
        _unset(model, "PX")
    else:
        model["PX"].value = px
    with pytest.raises(UnsupportedModelError, match="PX"):
        _resolve(model)


@pytest.mark.unit
def test_geometry_without_kom_is_refused():
    model = _ddr_model()
    _unset(model, "KOM")
    with pytest.raises(UnsupportedModelError, match="KOM"):
        _resolve(model)


@pytest.mark.unit
def test_a_missing_required_parameter_is_named():
    model = _ddr_model()
    _unset(model, "COSI")
    with pytest.raises(UnsupportedModelError, match="COSI"):
        _resolve(model)


@pytest.mark.unit
def test_ggamma_is_required_only_when_the_gr_maps_are_off():
    """``DDRPK Y`` derives ``g_gamma``; ``DDRPK N`` reads it."""
    model = _ddr_model(DDRPK=True)
    assert model["GGAMMA"].quantity is None
    _resolve(model)  # builds: the GR map supplies g_gamma

    model = _ddr_model(DDRPK=False)
    _unset(model, "GGAMMA")
    with pytest.raises(UnsupportedModelError, match="GGAMMA"):
        _resolve(model)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,default",
    [
        ("DDRR0", 8.178),
        ("DDRTHETA0", 220.0),
        ("DDRRHO0", 0.10),
        ("DDRZ0", 180.0),
        ("DDRZSUN", 20.0),
    ],
)
def test_the_galaxy_constants_are_pinned_at_their_defaults(name, default):
    """The physics uses Vela's already-converted literals, so a par that moves
    one of these would be silently ignored -- which is what ``PINNED_PARAMS``
    exists to refuse."""
    from vela_jax.freeze import PINNED_PARAMS, account_for_parameters

    assert PINNED_PARAMS[name] == default

    model = _ddr_model()
    consumed = set(model.params) - {name}
    account_for_parameters(model, None, consumed)  # at the default: accepted

    model[name].value = default * 1.5
    with pytest.raises(UnsupportedModelError, match=name):
        account_for_parameters(model, None, consumed)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["EDOT", "EPS1DOT", "EPS2DOT", "DR", "DTH"])
def test_the_unsupported_placeholders_are_inert_at_zero_and_refused_otherwise(name):
    from vela_jax.freeze import account_for_parameters

    model = _ddr_model()
    consumed = set(model.params) - {name}

    model[name].value = 0.0
    account_for_parameters(model, None, consumed)

    model[name].value = 1e-12
    with pytest.raises(UnsupportedModelError, match=name):
        account_for_parameters(model, None, consumed)


@pytest.mark.unit
def test_a_free_tgeo_is_refused():
    """``TGEO`` is the frozen origin of the projector; the freeze owns it."""
    from vela_jax.freeze import validate_model

    model = _ddr_model()
    validate_model(model)
    model["TGEO"].frozen = False
    with pytest.raises(UnsupportedModelError, match="TGEO"):
        validate_model(model)


@pytest.mark.unit
def test_a_materialised_tgeo_passes_accountability_with_geometry_off():
    """PINT sets ``TGEO = TASC`` whether or not geometry reads it.

    ``ddr_consumed`` correctly leaves ``TGEO`` out with geometry off, so
    without the ``INERT_PARAMS`` classification a perfectly ordinary
    geometry-off DDR par would fail accountability on a nonzero epoch it never
    uses.
    """
    from vela_jax.binary import ddr_consumed
    from vela_jax.freeze import INERT_PARAMS, account_for_parameters

    assert "TGEO" in INERT_PARAMS

    model = _ddr_model(DDRGEO=False, DDRKINE=False, DDRPBDOT="absorb_gw")
    _unset(model, "KOM")
    config = _resolve(model)
    assert "TGEO" not in ddr_consumed(config)
    assert model["TGEO"].quantity is not None
    assert model["TGEO"].value != 0.0
    account_for_parameters(
        model, None, ddr_consumed(config) | set(model.params) - {"TGEO"}
    )


@pytest.mark.unit
def test_ddr_consumes_only_what_its_selected_modes_read():
    """R5.3b: a static union over every mode would claim an inactive field
    reaches the trace."""
    from vela_jax.binary import ddr_consumed
    from vela_jax.binary.ddr import DDRConfig

    def config(**kwargs):
        flags = dict(
            use_fbx=False,
            ecliptic_coordinates=False,
            use_pk=True,
            pbdot_kinematic=False,
            use_geo=False,
            use_kine=False,
            obliquity=0.4,
        )
        flags.update(kwargs)
        return DDRConfig(**flags)

    plain = ddr_consumed(config())
    assert {"A1", "A1DOT", "TASC", "EPS1", "EPS2", "M2", "COSI"} <= plain
    assert "COSI" in plain  # always, in every mode
    assert "PB" in plain and "PBDOT" in plain
    assert not {"XPBDOT", "GGAMMA", "OMDOT", "PX", "TGEO", "KOM"} & plain

    assert "XPBDOT" in ddr_consumed(config(pbdot_kinematic=True))
    assert "PBDOT" not in ddr_consumed(config(pbdot_kinematic=True))
    assert {"GGAMMA", "OMDOT"} <= ddr_consumed(config(use_pk=False))

    kine = ddr_consumed(config(use_kine=True))
    assert {"PX", "TGEO", "POSEPOCH", "RAJ", "DECJ", "PMRA", "PMDEC"} <= kine
    assert "KOM" not in kine
    ecliptic = ddr_consumed(config(use_kine=True, ecliptic_coordinates=True))
    assert {"ELONG", "ELAT", "PMELONG", "PMELAT"} <= ecliptic
    assert not {"RAJ", "DECJ"} & ecliptic
    assert "KOM" in ddr_consumed(config(use_geo=True, use_kine=True))

    fbx = ddr_consumed(config(use_fbx=True))
    assert "FB0" in fbx and "PB" not in fbx and "PBDOT" not in fbx


@pytest.mark.unit
def test_the_ddr_mode_flags_are_inert_parameters():
    from vela_jax.freeze import INERT_PARAMS

    assert {"DDRPK", "DDRPBDOT", "DDRGEO", "DDRKINE"} <= INERT_PARAMS
