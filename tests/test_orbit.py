"""Mikkola's Kepler solver, and the build-time orbit-count reduction."""

import jax.numpy as jnp
import numpy as np
import pytest

from vela_jax.binary.orbit import mikkola
from vela_jax.precision import reduce_orbits

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("e", [0.0, 1e-8, 0.05, 0.3, 0.7, 0.9])
def test_mikkola_solves_kepler(e):
    l = np.linspace(-20 * np.pi, 20 * np.pi, 401)
    u = np.asarray(mikkola(jnp.asarray(l), e))
    assert np.max(np.abs(u - e * np.sin(u) - l)) < 1e-13


def test_mikkola_is_2pi_equivariant():
    """The orbit-count reduction removes whole orbits; the solver must not care."""
    l = np.linspace(-np.pi, np.pi, 101)
    for n in (0, 1, 137):
        shifted = np.asarray(mikkola(jnp.asarray(l + 2 * np.pi * n), 0.4))
        assert np.allclose(
            shifted - 2 * np.pi * n, np.asarray(mikkola(jnp.asarray(l), 0.4))
        )


def test_mikkola_gradient_is_finite_at_the_singular_inputs():
    """A NaN in the unselected branch of a `where` would poison the gradient."""
    import jax

    grad = jax.jacfwd(lambda x: mikkola(x, 0.3))(jnp.array([0.0, 1e-30, 1.0]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_orbit_reduction_stays_within_half_a_period():
    period = 86400.0 * 1.5334494515
    tau = np.linspace(-3.2e8, 3.2e8, 1000).astype(np.longdouble)
    reduction = reduce_orbits(tau, 0.0, period)
    assert np.all(np.abs(reduction.dt_red) <= period / 2 + 1e-6)
    assert np.all(reduction.n_orb == np.round(reduction.n_orb))
