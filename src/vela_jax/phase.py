"""Phase offset and system-dependent phase jumps.

Vela: ``src/model/phase_offset.jl``, ``src/model/jump.jl``. Note that Vela
multiplies a JUMP by the *constant* ``F_ + F[1]`` (i.e. F0), not by the
instantaneous spin frequency; we do the same.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import numerics as vm
from .correction import Correction


def phase_offset(frozen, corr: Correction, p):
    return corr.add_phase(vm.where(frozen.is_tzr, 0.0, -p.PHOFF))


def phase_jump_exclusive(frozen, corr: Correction, p):
    f0 = p.F0_ref + p.dF0
    table = jnp.concatenate([jnp.zeros(1), jnp.stack(list(p.JUMP))])
    jump = table[frozen.jump_index]
    return corr.add_phase(vm.where(frozen.is_tzr, 0.0, jump * f0))


def phase_jump(frozen, corr: Correction, p):
    f0 = p.F0_ref + p.dF0
    jump = 0.0
    for value, mask in zip(p.JUMP, frozen.jump_masks):
        jump = jump + vm.where(mask, value, 0.0)
    return corr.add_phase(vm.where(frozen.is_tzr, 0.0, jump * f0))
