"""Dispersion delays.

Vela: ``src/model/dispersion.jl`` (`DispersionTaylor`, `DispersionPiecewise`)
plus the shared ``dm / nu^2`` in ``src/model/component.jl``. The barycentric
observing frequency is the one Vela uses.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import numerics as vm
from .correction import Correction, corrected_time, inverse_freq_sqr
from .taylor import taylor_horner


def dispersion_taylor(frozen, corr: Correction, p):
    t = corrected_time(frozen, corr) - p.DMEPOCH
    dm = taylor_horner(t, p.DM)
    return corr.add_delay(dm * inverse_freq_sqr(frozen, corr))


def dispersion_piecewise(frozen, corr: Correction, p):
    # Index 0 means "no DMX window"; Vela's mask uses the same convention.
    table = jnp.concatenate([jnp.zeros(1), jnp.stack(list(p.DMX_))])
    dmx = vm.where(frozen.is_tzr, 0.0, table[frozen.dmx_index])
    return corr.add_delay(dmx * inverse_freq_sqr(frozen, corr))
