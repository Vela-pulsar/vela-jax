"""The consumer contract, checked against nltiming itself when it is installed.

vela-jax does not depend on nltiming -- the protocols are structural. That is
exactly why this test exists: nothing else would notice if the shape drifted.
"""

import numpy as np
import pytest

from vela_jax import TimingPulsar


@pytest.fixture(scope="module")
def nltiming():
    """Imported on first use, so collection does not pay for it."""
    return pytest.importorskip("nltiming")


@pytest.fixture(scope="module")
def pulsar(engine_factory):
    return TimingPulsar(engine_factory("sim_dd"))


def test_the_backend_satisfies_the_runtime_checkable_protocols(pulsar, nltiming):
    from nltiming.protocols import (
        JacobianTimingEngine,
        JaxTimingEngine,
        TimingEngine,
    )

    engine = pulsar.timing_engine("vela_jax")
    assert isinstance(engine, TimingEngine)
    assert isinstance(engine, JacobianTimingEngine)
    assert isinstance(engine, JaxTimingEngine)


def test_the_pulsar_satisfies_the_pulsar_protocols(pulsar, nltiming):
    from nltiming.protocols import PulsarData
    from nltiming.protocols import TimingPulsar as TimingPulsarProtocol

    assert isinstance(pulsar, PulsarData)
    assert isinstance(pulsar, TimingPulsarProtocol)


def test_engine_selection_resolves(pulsar, nltiming):
    from nltiming.engine_config import normalize_engines

    assert normalize_engines("vela_jax") == {
        "pint": "vela_jax",
        "tempo2": "vela_jax",
    }


def test_nltimings_own_validators_pass(pulsar, nltiming):
    from nltiming.engine_support import validate_engine_against_pulsar

    validate_engine_against_pulsar(pulsar.timing_engine("vela_jax"), pulsar)


def test_a_timing_context_builds_and_partitions_every_fitpar(pulsar, nltiming):
    """The full nltiming path: linearity resolution, chart candidacy, the plan."""
    spec = nltiming.TimingSpec(engines="vela_jax", name="timing")
    context = spec.for_pulsar(pulsar)
    covered = set(context.sampled) | set(context.marginalized)
    assert covered >= set(pulsar.fitpars) - {"ECC", "OM", "T0"}
    # The gauge column is marginalized, the binary axes are sampled.
    assert "PHOFF" in context.marginalized
    assert {"A1", "PB"} <= set(context.sampled)


def test_the_engine_declares_its_binary_chart_facts(engine_factory, nltiming):
    """Chart candidacy is decided from the engine's own binary stage, not a
    name search over the par."""
    from vela_jax import TimingPulsar

    engine = TimingPulsar(engine_factory("sim_dd")).timing_engine("vela_jax")
    capability = engine.binary_chart_capability("kepler_laplace", "")
    assert capability.kepler_convention == "dd"
    assert capability.epoch_shift_exact
    assert not capability.origin_certified


def test_the_hybrid_mode_reaches_nltimings_manifest_check(pulsar, nltiming):
    from nltiming.nonlinear_timing_model import _check_engine_nonlinear_params

    for mode in (None, "binary", "binary+"):
        engine = pulsar.timing_engine("vela_jax", nonlinear_params=mode)
        _check_engine_nonlinear_params(engine, mode)
    with pytest.raises(ValueError, match="nonlinear_params"):
        _check_engine_nonlinear_params(pulsar.timing_engine("vela_jax"), "binary")


def test_residual_delta_is_finite_and_zero_at_the_reference(pulsar, nltiming):
    engine = pulsar.timing_engine("vela_jax")
    zero = np.zeros(len(engine.fitpars))
    assert np.array_equal(engine.residual_delta(zero), np.zeros(len(pulsar.toas)))


def test_the_binary_registry_covers_the_engines_live_set(engine_factory, nltiming):
    """nltiming decides what a binary axis is; this engine must agree with it.

    The registry is the union of JUG's and this engine's, so it may name axes
    this engine cannot evaluate -- but for a par whose axes it *does* evaluate,
    the two must partition identically, or a hybrid mode would put an axis on
    the wrong path without saying so.
    """
    from nltiming.hybrid import is_binary_axis

    engine = engine_factory("sim_dd")
    live = set(engine.perturbative("binary").live_nonlinear)
    assert {name for name in engine.param_names if is_binary_axis(name)} == live
