"""Frequency-dependent profile-evolution delays.

Vela: ``src/model/frequency_dependent.jl`` (`FrequencyDependent`,
`FrequencyDependentJump`). Both use the *barycentric* observing frequency.
FD sits after the binary in the chain: it is a residual/phase effect, not a
shift of the time the binary sees.
"""

from __future__ import annotations

from . import numerics as vm
from .constants import NU_REF
from .correction import Correction, log_freq_ratio


def frequency_dependent(frozen, corr: Correction, p):
    lam = log_freq_ratio(frozen, corr, NU_REF)
    delay = 0.0
    for power, fd in enumerate(p.FD, start=1):
        delay = delay + fd * lam**power
    return corr.add_delay(delay)


def frequency_dependent_jump(frozen, corr: Correction, p):
    lam = log_freq_ratio(frozen, corr, NU_REF)
    delay = 0.0
    for fdj, power, mask in zip(p.FDJUMP, frozen.fdjump_exp, frozen.fdjump_masks):
        delay = delay + vm.where(mask, fdj * lam**power, 0.0)
    return corr.add_delay(vm.where(frozen.is_tzr, 0.0, delay))
