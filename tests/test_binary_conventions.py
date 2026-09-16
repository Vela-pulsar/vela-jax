"""The ``binary_conventions`` flag (S): tempo2's ELL1 Roemer truncation."""

import jax.numpy as jnp
import numpy as np
import pytest

from vela_jax import Engine
from vela_jax.binary.ell1 import (
    ELL1State,
    d2_romer_d_phi2_t2,
    d_romer_d_phi_t2,
    romer_delay_t2,
)


def test_unknown_conventions_are_refused(engine_factory):
    engine = engine_factory("sim_dd")
    with pytest.raises(ValueError, match="binary_conventions"):
        Engine.from_pint(
            engine.pint_model, engine.pint_toas, binary_conventions="tempo1"
        )


def test_ell1_truncation_matches_tempo2s_polynomials():
    """S2. tempo2 ``ELL1model.C``: Roemer to O(e), no harmonics in either
    derivative."""
    phi = jnp.linspace(-3.0, 3.0, 41)
    a1, eps1, eps2 = 1.9, 3.1e-5, -7.4e-5
    trigs = tuple((jnp.sin(k * phi), jnp.cos(k * phi)) for k in (1, 2, 3, 4))
    state = ELL1State(trigs, 1e-5, a1, eps1, eps2, 0.2, 0.99)

    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    expected = a1 * (sin_phi + 0.5 * (eps2 * np.sin(2 * phi) - eps1 * np.cos(2 * phi)))
    assert np.allclose(np.asarray(romer_delay_t2(state)), expected, atol=1e-15)
    assert np.allclose(np.asarray(d_romer_d_phi_t2(state)), a1 * cos_phi, atol=1e-15)
    assert np.allclose(np.asarray(d2_romer_d_phi2_t2(state)), -a1 * sin_phi, atol=1e-15)


def test_the_truncation_is_the_first_order_part_of_velas_polynomial():
    """The two agree as the eccentricity goes to zero, and differ at O(e^2)."""
    from vela_jax.binary.ell1 import romer_delay

    phi = jnp.linspace(-3.0, 3.0, 41)
    trigs = tuple((jnp.sin(k * phi), jnp.cos(k * phi)) for k in (1, 2, 3, 4))
    for eps, tolerance in [(0.0, 1e-15), (1e-6, 1e-11), (1e-2, 1e-3)]:
        state = ELL1State(trigs, 1e-5, 1.9, eps, eps, 0.2, 0.99)
        difference = np.max(
            np.abs(np.asarray(romer_delay(state, "ELL1") - romer_delay_t2(state)))
        )
        assert difference < tolerance


@pytest.mark.parametrize("name", ["sim_dd", "sim_ddk", "sim_dmx"])
def test_conventions_do_nothing_without_ell1(engine_factory, name):
    """S3, negative half: no ELL1 means the flag cannot bite."""
    reference = engine_factory(name)
    hybrid = Engine.from_pint(
        reference.pint_model, reference.pint_toas, binary_conventions="tempo2"
    )
    assert np.array_equal(hybrid.residuals(), reference.residuals())


def test_binary_facts_report_the_conventions(engine_factory):
    reference = engine_factory("J1802-2124.sim")
    hybrid = Engine.from_pint(
        reference.pint_model, reference.pint_toas, binary_conventions="tempo2"
    )
    assert not reference.binary_chart_facts().ell1_t2
    assert hybrid.binary_chart_facts().ell1_t2


@pytest.mark.tempo2
def test_the_flag_composes_with_the_tempo2_timing_package(tempo2_engine_factory):
    """S5's mirror: timing package and conventions are orthogonal choices."""
    pint_conventions = tempo2_engine_factory("J1802-2124.sim")
    both = tempo2_engine_factory("J1802-2124.sim", "tempo2")
    assert both.timing_package == "tempo2" and both.binary_conventions == "tempo2"
    assert pint_conventions.binary_conventions == "pint"
    difference = both.residuals() - pint_conventions.residuals()
    assert np.max(np.abs(difference - difference.mean())) > 1e-10
