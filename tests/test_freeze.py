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
def test_uses_fbx_refuses_both_and_neither():
    """PINT also refuses both at parse; this is the engine-side backup."""
    from vela_jax.binary import uses_fbx

    class _Param:
        def __init__(self, quantity):
            self.quantity = quantity

    both = {"PB": _Param(1.0), "FB0": _Param(1.0)}
    with pytest.raises(UnsupportedModelError, match="exactly one of PB and FB0"):
        uses_fbx(both)
    with pytest.raises(UnsupportedModelError, match="exactly one of PB and FB0"):
        uses_fbx({})


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
