"""Rotational phase and instantaneous spin frequency.

Vela: ``src/model/spindown.jl``. Vela splits ``F0`` into a frozen ``Double64``
high part ``F_`` and a free low part so that the big ``F_ * dt`` product is
formed in double-double. JAX has no such type, so the split moves to build time
and goes one step further (SPEC §4.3-4.4): the *entire* reference spin series
is evaluated at the undelayed TDB time in longdouble, and the trace evaluates
only its Taylor expansion in the delay,

    phase = phi_ref + c1*xi + c2*xi^2/2 + ...  +  sum_k dF_k t^(k+1)/(k+1)!

with ``xi = -delay`` (a few thousand seconds at most) and ``dF`` the *free*
spin deltas. Both formulations are algebraically identical; this one has no
large float64 product anywhere, so it stays exact for a fast-spinning-down
pulsar as well as for an MSP.
"""

from __future__ import annotations

from .correction import Correction, corrected_time
from .taylor import taylor_horner, taylor_horner_integral


def spindown(frozen, corr: Correction, p):
    xi = -corr.delay
    t = corrected_time(frozen, corr)
    phase = (
        frozen.phi_ref
        + taylor_horner_integral(xi, frozen.spin_coeffs)
        + taylor_horner_integral(t, p.dF)
    )
    spin_frequency = taylor_horner(xi, frozen.spin_coeffs) + taylor_horner(t, p.dF)
    return corr.add_phase(phase, spin_frequency)
