"""The accumulated per-TOA correction threaded through the component chain.

Vela: ``src/toa/toa.jl``, ``TOACorrection``. The ``efac``/``equad2`` fields are
absent by design — noise belongs to Discovery/Enterprise, never here.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from . import numerics as vm


class Correction(NamedTuple):
    """Vela ``TOACorrection``, one entry per row (TOAs plus the TZR row)."""

    delay: Any  # seconds
    phase: Any  # turns, small by construction (SPEC §4.3)
    spin_frequency: Any  # Hz in the pulsar frame; 0 until Spindown
    doppler: Any  # dimensionless
    ssb_psr_pos: vm.Vec3  # unit vector; zeros until SolarSystem

    @staticmethod
    def initial(n_rows: int, dtype=None) -> "Correction":
        zeros = jnp.zeros(n_rows, dtype=dtype)
        return Correction(zeros, zeros, zeros, zeros, (zeros, zeros, zeros))

    def add_delay(self, delay, doppler=0.0, ssb_psr_pos=None) -> "Correction":
        return self._replace(
            delay=self.delay + delay,
            doppler=self.doppler + doppler,
            ssb_psr_pos=self.ssb_psr_pos if ssb_psr_pos is None else ssb_psr_pos,
        )

    def add_phase(self, phase, delta_spin_frequency=0.0) -> "Correction":
        return self._replace(
            phase=self.phase + phase,
            spin_frequency=self.spin_frequency + delta_spin_frequency,
        )


def corrected_time(frozen, corr: Correction):
    """Vela ``corrected_toa_value``: TDB seconds since PEPOCH, delay-corrected."""
    return frozen.tau - corr.delay


def bary_freq(frozen, corr: Correction):
    """Vela ``doppler_corrected_observing_frequency``."""
    return frozen.freq_hz * (1.0 - corr.doppler)


def inverse_freq_sqr(frozen, corr: Correction):
    """``1 / nu^2``, exactly zero for an infinite-frequency row.

    A TOA -- or, more often, a TZR pseudo-TOA with no ``TZRFRQ`` -- may have
    PINT's infinite observing frequency, meaning it carries no dispersive
    delay at all. ``dm / inf**2`` gives the right *value* but an ``inf * 0``
    derivative, which is NaN and poisons every Jacobian column. The frozen
    frequency is therefore finite everywhere and the dispersive terms are
    selected off instead.
    """
    nu = bary_freq(frozen, corr)
    return vm.where(frozen.finite_freq, 1.0 / (nu * nu), 0.0)


def log_freq_ratio(frozen, corr: Correction, reference_hz):
    """``log(nu / nu_ref)``, zero for an infinite-frequency row."""
    nu = bary_freq(frozen, corr)
    return vm.where(frozen.finite_freq, vm.log(nu / reference_hz), 0.0)


def topo_spin_frequency(corr: Correction):
    """Vela ``doppler_shifted_spin_frequency``.

    Kept as Vela's named quantity and as the relation between the two residual
    conventions -- it is **not** the divisor
    :func:`~vela_jax.pipeline.form_residuals` uses (§8, G2).
    """
    return corr.spin_frequency * (1.0 + corr.doppler)
