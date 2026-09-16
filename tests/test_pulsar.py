"""``TimingPulsar``: composition, dispatch, and the kwargs contract (SPEC B.6).

The *record*'s own gates -- the one-source identities, the sort, ``dmx``,
``planetssb`` -- live in ``test_pulsar_data.py``. This file is about the object
that holds it: what it forwards, what it will answer for, and what it refuses.
"""

import numpy as np
import pytest

from vela_jax import TimingPulsar
from vela_jax.backend import VelaJaxTimingEngine

CASES = ["NGC6440E", "sim_dd", "sim_sw", "J1802-2124.sim"]


@pytest.fixture(scope="module")
def pulsars(engine_factory):
    cache = {}

    def build(name):
        if name not in cache:
            cache[name] = TimingPulsar(engine_factory(name))
        return cache[name]

    return build


def test_the_pulsar_and_its_engine_are_the_same_object_underneath(pulsars):
    """R-B6.1. What ``validate_engine_against_pulsar`` asks for, by
    construction rather than by luck: one freeze, one matrix, one fitpar list."""
    psr = pulsars("sim_dd")
    engine = psr.timing_engine("vela_jax")
    assert tuple(psr.fitpars) == tuple(engine.fitpars)
    assert np.array_equal(psr.Mmat, engine.design_matrix())
    assert np.array_equal(psr.residuals, psr.engine.residuals())


def test_gauge_column_is_present(pulsars):
    """nltiming requires an ``Offset``/``PHOFF`` column; pyvela's ``fix_params``
    forces PHOFF free, so the engine always carries one."""
    for name in CASES:
        assert "PHOFF" in pulsars(name).fitpars


def test_the_gauge_direction_is_the_phoff_column(pulsars):
    """R10.5. The gauge freedom is a phase offset, so it moves residual ``i``
    by ``1/F_i`` -- not by a constant, which is what a residual that divides
    by a constant ``F0`` would give. That direction *is* the ``PHOFF``
    column of the design matrix, to machine precision."""
    psr = pulsars("sim_dd")
    backend = psr.timing_engine("vela_jax")
    direction = backend.gauge_direction()
    column = psr.Mmat[:, list(psr.fitpars).index("PHOFF")]
    assert np.allclose(
        column / np.linalg.norm(column),
        direction / np.linalg.norm(direction),
        atol=1e-12,
    )
    # And it is genuinely not the constant vector on a real pulsar.
    assert np.ptp(direction) / direction.mean() > 1e-5


def test_the_record_names_its_producer_and_partim(pulsars):
    from psrdata import SINGLE_KEY

    psr = pulsars("sim_dd")
    assert psr.producer == "vela_jax"
    assert dict(psr.timing_package) == {SINGLE_KEY: "vela_jax"}
    assert dict(psr.partim_compatibility) == {SINGLE_KEY: psr.engine.timing_package}
    assert psr.residual_centering[SINGLE_KEY].stored_residuals == "none"
    assert set(psr.fitpars) <= set(psr.setpars)
    assert set(psr.parameters) == set(psr.setpars)


def test_timing_engine_dispatch(pulsars):
    """R-B6.3. One leg, one timing package, vela-jax."""
    psr = pulsars("sim_dd")
    assert psr.can_use_engines("vela_jax")
    assert psr.can_use_engines({"pint": "vela_jax", "tempo2": "vela_jax"})
    assert not psr.can_use_engines("jug")
    assert not psr.can_use_engines({"pint": "vela_jax", "tempo2": "libstempo"})
    with pytest.raises(ValueError, match="cannot be honoured"):
        psr.timing_engine("jug")


def test_engine_kwargs_are_checked_not_swallowed(pulsars):
    """R-B6.4. The six nltiming passes are accepted; a typo raises."""
    psr = pulsars("sim_dd")
    engine = psr.timing_engine(
        "vela_jax",
        tempo2_native="fixed_state_stripped",
        tempo2_jug_options=None,
        prime_sessions=True,
        verify_wiring=False,
        subtract_tzr=False,
    )
    assert isinstance(engine, VelaJaxTimingEngine)
    with pytest.raises(TypeError, match="unexpected keyword"):
        psr.timing_engine("vela_jax", subtrat_tzr=False)
    with pytest.raises(ValueError, match="subtract_tzr"):
        psr.timing_engine("vela_jax", subtract_tzr=True)


def test_derivative_method_is_recorded_not_ignored(pulsars):
    """R10.2. Under R9.3 the two routes are the same matrix, so both values are
    accepted and recorded for the run manifest; anything else raises."""
    psr = pulsars("sim_dd")
    for method in ("analytic", "autodiff"):
        engine = psr.timing_engine("vela_jax", derivative_method=method)
        assert engine.derivative_method == method
        assert np.array_equal(engine.design_matrix(), -psr.engine.residual_jacobian())
    with pytest.raises(ValueError, match="derivative_method"):
        psr.timing_engine("vela_jax", derivative_method="finite-difference")


def test_the_hybrid_mode_is_the_perturbative_engine(pulsars):
    """R10.3. ``nonlinear_params`` is not a stub here: vela-jax's perturbative
    engine *is* that residual formula, exact to ~1e-12 s in float64."""
    psr = pulsars("sim_dd")
    hybrid = psr.timing_engine("vela_jax", nonlinear_params="binary")
    assert hybrid.nonlinear_params == "binary"
    assert isinstance(hybrid, VelaJaxTimingEngine)

    full = psr.timing_engine("vela_jax")
    assert full.nonlinear_params is None
    index = full.fitpars.index("A1")
    delta = np.zeros(len(full.fitpars))
    delta[index] = 1e-7
    assert (
        np.max(np.abs(hybrid.residual_delta(delta) - full.residual_delta(delta)))
        < 1e-12
    )


def test_pulse_number_source_is_recorded(pulsars):
    assert pulsars("sim_dd").engine.pulse_number_source == "model"


def test_a_residuals_only_look_does_not_need_the_pulsar(examples):
    """R9.6. Building the record computes the Jacobian; ``Engine`` does not."""
    from vela_jax import Engine

    engine = Engine.from_files(examples / "NGC6440E.par", examples / "NGC6440E.tim")
    assert engine._jacobian is None
    assert engine.residuals().shape == (engine.toa_count,)
    assert engine._jacobian is None
    TimingPulsar(engine)
    assert engine._jacobian is not None


def test_from_pint_takes_a_built_pair(engine_factory):
    """Same positional contract as ``Engine.from_pint(model, toas)``."""
    engine = engine_factory("NGC6440E")
    psr = TimingPulsar.from_pint(engine.pint_model, engine.pint_toas)
    assert len(psr.toas) == engine.toa_count
    assert psr.engine.timing_package == "pint"


@pytest.mark.tempo2
def test_a_tempo2_pulsar_is_the_same_surface(examples):
    psr = TimingPulsar.from_files(
        examples / "sim_dd.par", examples / "sim_dd.tim", timing_package="tempo2"
    )
    assert psr.engine.timing_package == "tempo2"
    assert len(psr.toas) == len(psr.residuals) == psr.Mmat.shape[0]
    engine = psr.timing_engine({"tempo2": "vela_jax"})
    assert tuple(engine.fitpars) == tuple(psr.fitpars)
