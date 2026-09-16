"""The nltiming ``TimingEngine`` surface, checked without importing nltiming."""

import numpy as np
import pytest

from vela_jax.backend import VelaJaxTimingEngine

#: ``nltiming.protocols``: the members each protocol declares (SPEC §10).
TIMING_ENGINE = (
    "fitpars",
    "native_units",
    "reference_theta",
    "reference_theta_exact",
    "residual_delta",
    "design_matrix",
    "residual_centering",
)
JACOBIAN_ENGINE = ("residual_jacobian",)
JAX_ENGINE = ("residual_delta_jax", "precision_critical_fitpars")


@pytest.fixture(scope="module")
def backend(engine_factory):
    return VelaJaxTimingEngine(engine_factory("sim_dd"))


@pytest.mark.parametrize("member", TIMING_ENGINE + JACOBIAN_ENGINE + JAX_ENGINE)
def test_the_protocol_members_are_all_present(backend, member):
    assert hasattr(backend, member), member


def test_it_renames_rather_than_recomputes(backend, engine_factory):
    engine = engine_factory("sim_dd")
    assert backend.fitpars == engine.param_names
    assert backend.native_units == dict(engine.param_units)
    assert np.array_equal(backend.design_matrix(), engine.design_matrix())
    assert np.array_equal(
        backend.design_matrix(params={"x": 1.0}), engine.design_matrix()
    )
    public = backend.reference_theta_exact()
    internal = engine.reference_theta_exact()
    assert set(public) == set(internal)
    for name, value in public.items():
        if name == "PHOFF":
            continue
        assert value == internal[name]


def test_the_backend_exposes_the_record_phoff_origin(engine_factory):
    """R-3.2.2: the public PHOFF coordinate is a delta around stored residuals.

    The JAX chain keeps PINT's nonzero PHOFF internally; residual_delta is
    unchanged. The nltiming adapter and the Feather linear engine must agree
    with the record, not with that private origin.
    """
    from decimal import Decimal

    from vela_jax import TimingPulsar

    engine = engine_factory("sim_dd")
    psr = TimingPulsar(engine)
    backend = psr.timing_engine("vela_jax")
    linear = psr.data.linear_engine()

    internal = Decimal(engine.reference_theta_exact()["PHOFF"])
    assert internal != 0
    for exact in (
        psr.parameters["PHOFF"].value,
        backend.reference_theta_exact()["PHOFF"],
        linear.reference_theta_exact()["PHOFF"],
    ):
        assert Decimal(exact) == 0
    assert backend.reference_theta()[backend.fitpars.index("PHOFF")] == 0.0
    zero = np.zeros(len(backend.fitpars))
    assert np.array_equal(backend.residual_delta(zero), engine.residual_delta(zero))


def test_residuals_are_stored_without_centering(backend):
    """Vela removes no mean."""
    from psrdata import SINGLE_KEY

    centering = backend.residual_centering
    assert tuple(centering) == (SINGLE_KEY,)
    assert centering[SINGLE_KEY].stored_residuals == "none"
    assert centering[SINGLE_KEY].standard_output == "mean_removed"
    assert centering[SINGLE_KEY].standard_weighted is True


def test_chart_capability_is_never_self_certified(backend):
    capability = backend.binary_chart_capability("kepler_laplace", "")
    assert capability.kepler_convention == "dd"
    assert capability.supports_domain
    assert capability.origin_certified is False
    assert backend.binary_chart_capability("something_else", "") is None


def test_chart_capability_reaches_through_a_perturbative_engine(engine_factory):
    perturbative = engine_factory("sim_dd").perturbative("binary")
    backend = VelaJaxTimingEngine(perturbative, name="sim_dd")
    assert (
        backend.binary_chart_capability("kepler_laplace", "").kepler_convention == "dd"
    )


def test_an_isolated_pulsar_has_no_chart(engine_factory):
    assert (
        VelaJaxTimingEngine(engine_factory("NGC6440E")).binary_chart_capability(
            "kepler_laplace", ""
        )
        is None
    )
