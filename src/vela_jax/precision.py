"""Build-time reductions that keep the float64 residual honest (SPEC §4).

Only two quantities in pulsar timing are too large for float64: the spin
phase ``F0*(t-PEPOCH)`` (2.5e12 turns and ~700 ns of ulp at F0~800 Hz over
100 yr; 3e10 turns and 20 ns on a typical 3 yr span) and the unreduced
orbital phase ``2*pi*(t-T0)/PB`` (a ~0.4 ps sawtooth at 1e4 orbits). Both
are reduced here, once, in numpy longdouble, so that everything the JAX
graph ever sees is small.

The two algorithms are re-implementations of the ones documented in JUG
(``jug/delays/barycentric_jax.py``, ``jug/utils/orbit_reduction.py``); the code
is local and shares nothing with that package.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .constants import DAY_S
from .taylor import taylor_horner, taylor_horner_integral


def seconds_since_epoch(mjd_ld: np.ndarray, epoch_mjd) -> np.ndarray:
    """``tau`` in longdouble seconds, from PINT's longdouble ``tdbld`` column."""
    return (np.asarray(mjd_ld, dtype=np.longdouble) - np.longdouble(epoch_mjd)) * (
        np.longdouble(DAY_S)
    )


def reference_phase(
    tau_ld: np.ndarray, pulse_number: np.ndarray, tau_tzr_ld, f_ld
) -> np.ndarray:
    """The frozen part of the spin phase (SPEC §4.3).

    ``phi_ref = phi(tau) - N - phi(tau_tzr)`` in longdouble, cast to float64,
    where ``phi`` is the reference spin Taylor series *on the undelayed* TDB
    time. It contains no delay physics whatsoever — only the reference spin
    coefficients, the raw TDB arrival times and the pulse numbers — so both
    ``r(theta*)`` and ``r(theta*+delta)`` see identical physics in the trace.

    Its magnitude is that of the phase the delays are responsible for,
    ``~F0*D <= 8e5`` turns at F0~800 Hz and D~1000 s, which float64 carries
    to 0.22 ps.
    """
    n_ld = np.asarray(pulse_number, dtype=np.longdouble)
    tau = np.asarray(tau_ld, dtype=np.longdouble)
    phi = taylor_horner_integral(tau, f_ld) - n_ld
    return (phi - taylor_horner_integral(np.longdouble(tau_tzr_ld), f_ld)).astype(
        np.float64
    )


def spin_coefficients(tau_ld: np.ndarray, f_ld) -> tuple[np.ndarray, ...]:
    """Reference spin phase re-expanded about each TOA's *undelayed* time.

    Vela keeps the whole ``F0 dt`` product in Double64. JAX has no such type,
    so instead of carrying the large absolute time into the trace we hand the
    trace the derivatives of the reference phase at ``tau``:

        phi(tau + xi) - phi(tau) = c1 xi + c2 xi^2/2! + ...,  c_m = phi^(m)(tau)

    with ``xi = -delay``, at most a few thousand seconds. Every ``c_m`` is
    computed in longdouble and then fits float64 comfortably (``c1`` is just
    the spin frequency), so the spin phase is exact to well under a
    picosecond no matter how large ``F1`` is. The corresponding
    ``phi(tau) - N`` constant is :func:`reference_phase`.
    """
    tau = np.asarray(tau_ld, dtype=np.longdouble)
    return tuple(
        np.asarray(taylor_horner(tau, f_ld[m:]), dtype=np.float64)
        for m in range(len(f_ld))
    )


class OrbitReduction(NamedTuple):
    """Frozen integer-orbit reduction of the binary time argument (SPEC §4.5)."""

    dt_red: np.ndarray  # float64 seconds, |dt_red| <= P*/2
    n_orb: np.ndarray  # exact float64 integers
    period_ref_s: float  # P* used for the reduction


def reduce_orbits(tau_ld: np.ndarray, epoch_rel_s, period_s) -> OrbitReduction:
    """Split ``t - T0`` into whole reference orbits plus a small remainder.

    The integer orbit count drops out of every trigonometric function, and it
    is retained so that a *live* ``PB`` stays exact: the late-time orbital
    phase moves by ``-n * dPB / PB``, which is physics, not bookkeeping.
    """
    period_ld = np.longdouble(period_s)
    dt_ld = np.asarray(tau_ld, dtype=np.longdouble) - np.longdouble(epoch_rel_s)
    n_orb = np.round(dt_ld / period_ld)
    dt_red = (dt_ld - n_orb * period_ld).astype(np.float64)
    return OrbitReduction(dt_red, n_orb.astype(np.float64), float(period_s))
