"""Spherically symmetric solar-wind dispersion.

Vela: ``src/model/solarwind.jl`` (`SolarWindDispersion`). Needs
``ssb_psr_pos``, so the chain always places it after ``solar_system``.
"""

from __future__ import annotations

from . import numerics as vm
from .constants import AU_LS
from .correction import Correction, corrected_time, inverse_freq_sqr
from .taylor import taylor_horner


def sun_angle_and_distance(frozen, corr: Correction):
    """Pulsar-Sun-observatory angle rho, and the observatory-Sun distance.

    Vela's angle, not PINT's: theirs is ``pi - rho``.
    """
    rvec = frozen.obs_sun_pos
    r = vm.norm3(rvec)
    cos_rho = -vm.dot3(corr.ssb_psr_pos, rvec) / r
    return vm.arccos(vm.clip(cos_rho, -1.0, 1.0)), r


def solar_wind(frozen, corr: Correction, p):
    rho, r = sun_angle_and_distance(frozen, corr)
    t = corrected_time(frozen, corr) - p.SWEPOCH
    ne_sw = taylor_horner(t, p.NE_SW)
    slope = ne_sw * AU_LS * AU_LS * rho / (r * vm.sin(rho))
    delay = slope * inverse_freq_sqr(frozen, corr)
    return corr.add_delay(vm.where(frozen.is_bary, 0.0, delay))
