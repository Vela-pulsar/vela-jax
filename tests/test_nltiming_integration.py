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


@pytest.fixture(scope="module")
def ddr_pulsar(ddr_fixture):
    from vela_jax import Engine

    return TimingPulsar(Engine.from_files(*ddr_fixture))


def test_ddr_does_not_claim_the_dd_chart(ddr_pulsar, nltiming):
    """DDR's native coordinates are its own, and nltiming has to be told.

    ``supports_domain=False`` is the second half: a valid box prior on DDR's
    independent inputs does not guarantee a physical state, so a consumer that
    assumed a total domain would treat the engine's NaN as a bug.
    """
    engine = ddr_pulsar.timing_engine("vela_jax")
    capability = engine.binary_chart_capability("kepler_laplace", "")
    assert capability.kepler_convention == "ddr"
    assert not capability.epoch_shift_exact
    assert not capability.supports_domain


def test_the_registry_agrees_with_ddrs_live_set(ddr_pulsar, nltiming):
    """``COSI`` is the axis this would have caught: without the companion
    registry entry, the hybrid split hands it to the linear path, where no
    engine evaluates it."""
    from nltiming.hybrid import is_binary_axis

    engine = ddr_pulsar.engine
    live = set(engine.perturbative("binary").live_nonlinear)
    assert "COSI" in live
    assert {name for name in engine.param_names if is_binary_axis(name)} == live


def test_an_invalid_ddr_point_reaches_the_consumer_as_a_non_finite_residual(
    ddr_pulsar, nltiming
):
    """D13: this package returns NaN and stops there.

    Turning a non-finite residual into a ``-inf`` log density is the sampler's
    job -- Discovery's or nltiming's -- and deliberately not a likelihood layer
    here. What vela-jax owes the consumer is that the signal arrives at all:
    NaN, from an ordinary ``residual_delta`` call, without an exception.
    """
    engine = ddr_pulsar.timing_engine("vela_jax")
    fitpars = list(engine.fitpars)
    delta = np.zeros(len(fitpars))
    delta[fitpars.index("M2")] = -0.6  # inferred pulsar mass goes negative

    residuals = np.asarray(engine.residual_delta(delta), dtype=float)
    assert np.all(np.isnan(residuals))
    # And the reference point is unaffected: the NaN is the sampled point's.
    assert np.all(np.isfinite(engine.residual_delta(np.zeros(len(fitpars)))))
