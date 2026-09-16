"""tempo2 as the timing package that reads the files (H)."""

from pathlib import Path

import numpy as np
import pytest

from vela_jax import Engine
from vela_jax.constants import DAY_S
from vela_jax.read_tempo2 import ROEMER_TOLERANCE_S

pytestmark = pytest.mark.tempo2

#: Fixtures tempo2 can open, spanning both frames and the binary families.
#: Every one of these forks a sandboxed tempo2, so the breadth half runs at
#: checkpoints and the behaviour half -- one equatorial, one ecliptic, one
#: binary -- runs by default. ``sim_jump`` is omitted: its TOAs precede the
#: GBT clock file, so PINT extrapolates and tempo2 applies no correction.
TEMPO2_FIXTURES = [
    pytest.param("NGC6440E"),  # isolated, equatorial
    pytest.param("sim_dd"),  # DD
    pytest.param("J2302+4442.sim"),  # ecliptic DDS: the frame round trip
    pytest.param("sim_ddk", marks=pytest.mark.slow),  # DDK, parallax in the Roemer
    pytest.param("sim_sw", marks=pytest.mark.slow),  # ecliptic + solar wind
    pytest.param("sim_dmx", marks=pytest.mark.slow),
    pytest.param("J1802-2124.sim", marks=pytest.mark.slow),  # ELL1 + proper motion
]


@pytest.mark.parametrize("name", TEMPO2_FIXTURES)
def test_tdbld_is_the_observatory_tdb_not_bbat(tempo2_engine_factory, name):
    """H1. ``bbat`` is barycentric; using it would double-count the Roemer.

    The engine's time argument must be ``SAT + clock + (TT->TB)``, which
    differs from ``bbat`` by exactly the barycentring tempo2 already did --
    hundreds of seconds, not a rounding difference.
    """
    engine = tempo2_engine_factory(name)
    columns = engine.tempo2_columns
    pepoch = engine.pint_model["PEPOCH"].value
    tau = np.asarray((columns.tdbld - pepoch) * DAY_S, dtype=float)
    assert np.allclose(tau, np.asarray(engine.frozen.tau)[:-1], atol=1e-6)
    # The Roemer delay tempo2 removed is 10^2 s; the engine must not have it.
    assert np.max(np.abs(columns.roemer_residual)) < ROEMER_TOLERANCE_S
    assert np.max(np.abs(np.asarray(engine.frozen.tau))) > 1.0


@pytest.mark.parametrize("name", TEMPO2_FIXTURES)
def test_the_geometry_columns_reproduce_tempo2s_own_roemer(tempo2_engine_factory, name):
    """H4. The load-bearing column test (see ``read_columns``)."""
    residual = tempo2_engine_factory(name).tempo2_columns.roemer_residual
    assert np.sqrt(np.mean(residual**2)) < 1e-12, "RMS above a picosecond"


def test_science_toas_are_never_reclocked(examples, monkeypatch):
    """H2. PINT must not recompute clocks, posvels or pulse numbers.

    Only the TZR pseudo-TOA, a separate one-row ``TOAs`` object, may go
    through PINT's own geometry; the science table is tempo2's alone.
    """
    from pint.toa import TOAs

    touched = []

    def guard(method):
        original = getattr(TOAs, method)

        def wrapper(self, *args, **kwargs):
            touched.append((method, getattr(self, "vela_jax_timing_package", None)))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(TOAs, method, wrapper)

    for method in ("compute_TDBs", "compute_posvels", "compute_pulse_numbers"):
        guard(method)

    Engine.from_tempo2(examples / "NGC6440E.par", examples / "NGC6440E.tim")
    assert [call for call in touched if call[1] == "tempo2"] == []


def test_from_pint_on_the_injected_table_is_identical(tempo2_engine_factory):
    """H3. The load-bearing test: the timing package only supplies arrays.

    Handing the injected ``(model, toas)`` back through the ordinary PINT
    constructor must reproduce the engine bit for bit -- if it does not, the
    difference is in the injection, not in the physics.
    """
    tempo2_engine = tempo2_engine_factory("sim_dd")
    rebuilt = Engine.from_pint(tempo2_engine.pint_model, tempo2_engine.pint_toas)
    assert np.array_equal(rebuilt.residuals(), tempo2_engine.residuals())
    assert rebuilt.param_names == tempo2_engine.param_names
    assert np.array_equal(rebuilt.design_matrix(), tempo2_engine.design_matrix())


def test_the_site_velocity_comes_from_tempo2s_own_field(tempo2_engine_factory):
    """``observatory_earth[3:6]`` is not the site velocity.

    tempo2 stores the site position there and leaves the velocity half at
    zero (``tempo2.h``); ``dm_delays.C`` builds the observatory velocity as
    ``earth_ssb[3:6] + siteVel``. The Earth's rotation is 1.3% of that, and
    dropping it is silent in every position-based check.
    """
    engine = tempo2_engine_factory("sim_dd")
    velocity = np.stack(
        [np.asarray(component) for component in engine.frozen.ssb_obs_vel], axis=1
    )[:-1]
    earth = np.asarray(engine.tempo2_columns.ssb_obs_vel)
    assert np.allclose(velocity, earth, atol=0)
    speed = np.linalg.norm(velocity, axis=1)
    # Earth orbital ~1e-4 c; the site term is a per-cent-level modulation, so
    # a velocity that has lost it has a visibly smaller spread.
    assert np.ptp(speed) / speed.mean() > 1e-3


def test_zero_delta_and_a_finite_jacobian(tempo2_engine_factory):
    engine = tempo2_engine_factory("sim_ddk")
    delta = np.zeros(len(engine.param_names))
    assert np.array_equal(
        np.asarray(engine.residual_delta_jax(delta)), np.zeros(engine.toa_count)
    )
    assert np.all(np.isfinite(engine.residual_jacobian()))


def test_the_design_oracle_is_pints_on_the_injected_table(tempo2_engine_factory):
    """The R9.4 oracle is PINT's matrix on tempo2's columns, never
    libstempo's tangent, and on the same rows -- nothing reorders TOAs."""
    engine = tempo2_engine_factory("sim_dd")
    matrix, names, _ = engine.pint_model.designmatrix(engine.pint_toas, incoffset=False)
    index = {name: i for i, name in enumerate(names)}
    expected = matrix[:, [index[n] for n in engine.param_names]]
    assert np.array_equal(engine.design_matrix(source="pint"), expected)
    # The product is the Jacobian, not the oracle (R9.3/R9.6).
    assert np.array_equal(engine.design_matrix(), -engine.residual_jacobian())


def test_pulse_numbers_come_from_tempo2_when_the_tim_carries_them(
    tempo2_engine_factory,
):
    """H6. ``pulse_number``, never ``nphase`` -- and only when tempo2 has it."""
    engine = tempo2_engine_factory("NGC6440E")
    assert engine.pulse_number_source in ("tempo2", "model")
    pulse_numbers = np.asarray(engine.pint_toas.table["pulse_number"])
    assert np.all(pulse_numbers == np.round(pulse_numbers))
    # phi_ref stays small: that is what says the fiducial was resolved.
    assert np.max(np.abs(np.asarray(engine.frozen.phi_ref))) < 1e7


def test_timing_package_metadata(tempo2_engine_factory, engine_factory):
    tempo2_engine = tempo2_engine_factory("NGC6440E")
    assert tempo2_engine.timing_package == "tempo2"
    assert tempo2_engine.source_units in ("TDB", "TCB")
    assert "UNITS" in (tempo2_engine.par_text or "")
    assert engine_factory("NGC6440E").timing_package == "pint"


def test_a_tcb_par_is_converted_before_pint_sees_it(examples, tmp_path):
    """H5. ``UNITS SI`` on a tempo2 file is TCB, and PINT must never see it."""
    source = (examples / "NGC6440E.par").read_text().replace("UNITS", "C UNITS")
    par = tmp_path / "tcb.par"
    par.write_text(source + "UNITS SI\n")

    engine = Engine.from_tempo2(par, examples / "NGC6440E.tim")
    assert engine.source_units == "TCB"
    assert "UNITS TDB" in " ".join(engine.par_text.split())

    # PINT parsed converted numbers, not the ones in the file: the spin
    # frequency has moved by the IFTE rate (~1.55e-8 fractional).
    original = next(line for line in source.splitlines() if line.split()[:1] == ["F0"])
    ratio = float(engine.reference_theta_exact()["F0"]) / float(original.split()[1])
    assert 1e-9 < abs(ratio - 1.0) < 1e-7


def test_a_troposphere_par_builds_with_a_warning(examples):
    """H8. tempo2 applied a delay we do not model; that is an offset, not a
    reason to refuse the file."""
    par = examples / "J0613-0200.InPTA.NB.par"
    if not par.exists():
        pytest.skip("no troposphere fixture")
    with pytest.warns(UserWarning, match="TROPOSPHERE"):
        engine = Engine.from_tempo2(par, examples / "J0613-0200.InPTA.NB.tim")
    assert engine.toa_count > 0


# --- H7: both timing packages, ECL pinned to tempo2's obliquity -------------

#: tempo2 has no ``ECL`` keyword (``readParfile.C``); it always rotates with
#: ``ECLIPTIC_OBLIQUITY_VAL`` = 84381.4059", which is PINT's ``IERS2003``.
#: PINT honours ``ECL``. Setting ``ECL IERS2003`` on the PINT copy makes
#: both packages use tempo2's default on the same ELONG/ELAT numbers.
#: That is a freeze pin, not a sky-coordinate transform.
TEMPO2_ECL = "IERS2003"

#: Interpolation / clock-file ceiling. Measured floor is ~1.5 ns on
#: observatory fixtures whose clocks both packages apply.
TWO_PACKAGE_RMS_S = 100e-9

#: Vela ``sim_jump`` (GBT clock coverage) and ``sim_sw`` (mixed-engine
#: solar-wind / planet-Shapiro profile) are not this list. Replacements
#: live under ``tests/data/``.
_H7_LOCAL = {
    "sim_jump_clk": Path(__file__).resolve().parent / "data" / "sim_jump_clk",
    "sim_sw_aligned": Path(__file__).resolve().parent / "data" / "sim_sw_aligned",
}

#: Vela.jl examples plus the two local replacements. Non-core names are
#: ``slow``.
_H7_NAMES = [
    "NGC6440E",
    "sim1",
    "sim2",
    "pure_rotator",
    "J1856-3754.sim",
    "sim_dmx",
    "sim_fd",
    "sim_jump_clk",
    "sim_jump_ex",
    "sim_sw_aligned",
    "sim_dd",
    "J0955-6150.sim",
    "sim_ddk",
    "J0453+1559.sim",
    "J1208-5936.sim",
    "J2302+4442.sim",
    "J1802-2124.sim",
    "J1227-6208.sim",
    "sim_ell1k",
]
_H7_CORE = {
    "NGC6440E",
    "sim_dd",
    "J1208-5936.sim",
    "J2302+4442.sim",
    "sim_ddk",
    "J1802-2124.sim",
    "J1227-6208.sim",
    "sim_ell1k",
    "sim_dmx",
}


def _h7_params():
    return [
        pytest.param(name, marks=() if name in _H7_CORE else pytest.mark.slow)
        for name in _H7_NAMES
    ]


def frame_pinned(examples, name, directory):
    """Par text with ``ECL`` set to tempo2's obliquity so PINT matches it."""
    source = (examples / f"{name}.par").read_text()
    keep = [line for line in source.splitlines() if line.split()[:1] != ["ECL"]]
    if "ECL" in source:
        keep.append(f"ECL {TEMPO2_ECL}")
    par = Path(directory) / f"{name}.par"
    par.write_text("\n".join(keep) + "\n")
    return par


def h7_files(examples, name, directory):
    """Par/tim for an H7 row: local replacements, or Vela with ECL pinned."""
    local = _H7_LOCAL.get(name)
    if local is not None:
        return local / f"{name}.par", local / f"{name}.tim"
    return frame_pinned(examples, name, directory), examples / f"{name}.tim"


@pytest.fixture(scope="module")
def h7_directory(tmp_path_factory):
    return tmp_path_factory.mktemp("h7_ecl")


@pytest.mark.parametrize("name", _h7_params())
def test_the_two_timing_packages_agree_to_the_clock_floor(examples, name, h7_directory):
    """H7 (A.6.1). Both vela-jax timing packages, same files, same frame.

    ``Engine.residuals()`` is the JAX chain on each freeze, not
    ``PINT.Residuals`` vs libstempo. ``ECL IERS2003`` on Vela ecliptic
    pars tells PINT to use tempo2's default obliquity. Local replacements
    are clock-covered (``sim_jump_clk``) or MetaPulsar mixed-engine aligned
    (``sim_sw_aligned``). Conventions stay ``"pint"`` on both so the
    comparison is the freeze, not the ELL1 truncation.
    """
    par, tim = h7_files(examples, name, h7_directory)
    if not (par.is_file() and tim.is_file()):
        pytest.skip(f"fixture {name} not available")
    pint_engine = Engine.from_files(par, tim, timing_package="pint")
    tempo2_engine = Engine.from_files(par, tim, timing_package="tempo2")
    assert pint_engine.toa_count == tempo2_engine.toa_count
    difference = tempo2_engine.residuals() - pint_engine.residuals()
    difference = difference - difference.mean()
    rms = float(np.sqrt(np.mean(difference**2)))
    assert rms < TWO_PACKAGE_RMS_S, (
        f"{name}: RMS {rms * 1e9:.2f} ns between timing packages "
        f"(gate {TWO_PACKAGE_RMS_S * 1e9:.0f} ns)"
    )


def test_an_unpinned_ecliptic_frame(examples):
    """Leaving ``ECL IERS2010`` is PINT's frame, not tempo2's.

    The pin in H7 exists because this difference is ~100 ns of Roemer, real,
    and nothing to do with clocks.
    """
    name = "J2302+4442.sim"
    par, tim = examples / f"{name}.par", examples / f"{name}.tim"
    if not (par.is_file() and tim.is_file()):
        pytest.skip(f"fixture {name} not available")
    pint_engine = Engine.from_files(par, tim, timing_package="pint")
    tempo2_engine = Engine.from_files(par, tim, timing_package="tempo2")
    difference = tempo2_engine.residuals() - pint_engine.residuals()
    difference = difference - difference.mean()
    rms = float(np.sqrt(np.mean(difference**2)))
    assert rms > 50e-9, (
        f"unpinned {name} RMS {rms * 1e9:.1f} ns; expected the IERS2010 vs "
        "IERS2003 frame, not a clock-scale difference"
    )


# --- Barycentric ELL1: the packages, not freeze identity --------------------

#: ``DM 0``, ``ECL IERS2003``, ``UNITS TDB``, ``TZRSITE @``, TOAs at ``bat``.
#: PINT and libstempo agree to a fraction of a nanosecond on this pair.
#: Both vela-jax reads inject ``ssb_obs_pos = 0``, so their residuals can
#: be bit-identical without that being a PINT–tempo2 measurement.
_MATCHED = Path(__file__).resolve().parent / "data" / "J1909-3744-sim"
_MATCHED_PAR = _MATCHED / "J1909-3744.par"
_MATCHED_TIM = _MATCHED / "J1909-3744.tim"


def test_j1909_packages_agree_under_1ns():
    """PINT vs libstempo on the barycentric ELL1 pair (not vela-jax identity)."""
    pytest.importorskip("libstempo")
    if not (_MATCHED_PAR.is_file() and _MATCHED_TIM.is_file()):
        pytest.skip("matched J1909-3744-sim files are missing")
    import astropy.units as u
    from libstempo.sandbox import tempopulsar
    from pint.models import get_model
    from pint.residuals import Residuals
    from pint.toa import get_TOAs

    model = get_model(str(_MATCHED_PAR))
    toas = get_TOAs(str(_MATCHED_TIM), model=model, include_pn=True)
    r_pint = np.asarray(Residuals(toas, model).time_resids.to_value(u.s))
    psr = tempopulsar(parfile=str(_MATCHED_PAR), timfile=str(_MATCHED_TIM), dofit=False)
    r_t2 = np.asarray(
        psr.residuals(updatebats=True, formresiduals=True, removemean=True),
        dtype=float,
    )
    r_pint = r_pint - r_pint.mean()
    r_t2 = r_t2 - r_t2.mean()
    rms_ns = float(np.sqrt(np.mean((r_pint - r_t2) ** 2))) * 1e9
    assert rms_ns < 1.0, f"PINT vs libstempo RMS {rms_ns:.4f} ns"


@pytest.mark.no_tempo2
@pytest.mark.unit
def test_native_libstempo_is_not_a_path():
    from vela_jax.read_tempo2 import load

    with pytest.raises(TypeError, match="libstempo.sandbox"):
        load("unused.par", "unused.tim", sandbox=False)
