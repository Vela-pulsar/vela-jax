"""Orbital phase, mean motion, and the Kepler solver.

Vela: ``src/model/binary/orbit.jl``. The mean anomaly differs from Vela's in
one respect only: the whole reference orbits have been removed at build time
(SPEC §4.5), which is invisible to every trigonometric function downstream and
removes a ~0.4 ps float64 sawtooth at 1e4 orbits.
"""

from __future__ import annotations

import jax.numpy as jnp

from .. import numerics as vm
from ..taylor import factorial_series, taylor_horner

TWO_PI = 2.0 * jnp.pi


def mean_anomaly(fb, dt_full, dt_red, p, use_fbx: bool):
    """Orbital phase, with the frozen integer orbit count divided out.

    PB mode: ``(t-T0)/PB = n + dt_red/PB - n*dPB/PB``; the integer ``n`` is
    dropped and the ``n*dPB`` term kept, because that term is how a live PB
    moves late-time orbital phase.

    FBX mode: ``FB0*(n P* + dt_red) = n + n*(FB0 P* - 1) + FB0*dt_red``, plus
    the FB1.. integral, which is small enough to take the full ``dt``.
    """
    if use_fbx:
        fb0 = p.FB[0]
        linear = fb0 * dt_red + fb.n_orb * (fb0 * fb.period_ref_s - 1.0)
        return TWO_PI * (linear + factorial_series(dt_full, p.FB[1:], 2))
    return TWO_PI * (
        dt_red / p.PB - fb.n_orb * p.dPB / p.PB - 0.5 * p.PBDOT * (dt_full / p.PB) ** 2
    )


def mean_motion(dt_full, p, use_fbx: bool):
    if use_fbx:
        return TWO_PI * taylor_horner(dt_full, p.FB)
    return TWO_PI / (p.PB + p.PBDOT * dt_full)


def mikkola(l, e):
    """Kepler's equation by Mikkola's method (Mikkola 1987).

    Vela short-circuits ``iszero(e) || iszero(l) -> l``. Under tracing both
    branches are evaluated, and a NaN in the unselected branch of a
    ``where`` poisons the gradient, so the singular inputs are replaced by
    safe substitutes first and selected out afterwards. That "substitute,
    then select" pattern is the general rule for every Vela ``if`` guarding a
    singular expression.
    """
    trivial = (e == 0.0) | (l == 0.0)
    e = jnp.where(trivial, 0.5, e)
    l0 = jnp.where(trivial, 1.0, l)

    sgn = jnp.sign(l0)
    la = jnp.abs(l0)
    ncycles = jnp.floor(la / TWO_PI)
    la = la - TWO_PI * ncycles
    flag = la > jnp.pi
    la = jnp.where(flag, TWO_PI - la, la)

    alpha = (1.0 - e) / (4.0 * e + 0.5)
    beta = (la / 2.0) / (4.0 * e + 0.5)
    root = jnp.sqrt(alpha**3 + beta * beta)
    z = jnp.cbrt(jnp.where(beta > 0, beta + root, beta - root))

    s = z - alpha / z
    w = s - 0.078 * s**5 / (1.0 + e)
    e0 = la + e * (3.0 * w - 4.0 * w**3)

    su, cu = jnp.sin(e0), jnp.cos(e0)
    esu, ecu = e * su, e * cu
    fu = e0 - esu - la
    f1, f2, f3, f4 = 1.0 - ecu, esu, ecu, -esu
    u1 = -fu / f1
    u2 = -fu / (f1 + f2 * u1 / 2)
    u3 = -fu / (f1 + f2 * u2 / 2 + f3 * u2 * u2 / 6.0)
    u4 = -fu / (f1 + f2 * u3 / 2 + f3 * u3 * u3 / 6.0 + f4 * u3**3 / 24.0)
    xi = e0 + u4

    sol = jnp.where(flag, TWO_PI - xi, xi)
    u = sgn * (sol + ncycles * TWO_PI)
    return jnp.where(trivial, l, u)


def eccentric_anomaly(l, e):
    """Kepler solve, dispatched through :mod:`vela_jax.numerics`."""
    return vm.kepler(l, e, mikkola)
