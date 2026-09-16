"""Delta-formulated engine over a restricted live set (SPEC §11).

The full residual cannot run in float32: the Roemer delay is 500 s and float32
resolves it to 3e-5 s, so ``r(theta*+delta) - r(theta*)`` computed as two
absolute evaluations would be pure cancellation noise. Every *delta*, however,
can — provided it is never formed as a difference of absolutes.

This engine gets that for free. It runs the same component chain as
:class:`~vela_jax.engine.Engine` over
:class:`~vela_jax.perturbative.dual.Pert` values, which carry
``(reference, perturbation)`` and implement each operation as a
cancellation-free difference identity. The result is

    Delta r = -M @ delta_lin  -  (Delta D - Delta D_tzr)

with the identically-linear (and merely linear) axes served by the fp64-baked
design matrix and the live nonlinear axes by the delta chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..correction import Correction
from ..errors import UnsupportedModelError
from .dual import Pert

#: Live axes the delta kernels cover (SPEC §11.2). Spin, DM, FD, JUMP, PHOFF
#: and DMX are never perturbative-live: they are identically linear, or linear
#: far below float32 resolution for a PTA MSP.
# fmt: off
ASTROMETRY_AXES = (
    "RAJ", "DECJ", "ELONG", "ELAT", "PMRA", "PMDEC", "PMELONG", "PMELAT", "PX",
)
BINARY_AXES = (
    "A1", "PB", "FB0", "ECC", "OM", "T0", "EPS1", "EPS2", "TASC",
    "SINI", "M2", "H3", "STIGMA", "SHAPMAX", "KIN", "KOM", "GAMMA",
    "OMDOT", "PBDOT", "EDOT", "A1DOT", "EPS1DOT", "EPS2DOT", "LNEDOT",
)
# fmt: on

#: Named modes, matching JUG's and nltiming's ``nonlinear_params`` vocabulary.
MODES = {
    "binary": BINARY_AXES,
    "binary+": BINARY_AXES + ("PX",),
    "binary+astrometry": BINARY_AXES + ASTROMETRY_AXES,
    "astrometry": ASTROMETRY_AXES,
}

#: Stages that contribute phase rather than delay; the perturbative residual
#: gets those from the design matrix, so the delta chain stops before them.
_PHASE_STAGES = ("spindown", "phase_offset", "phase_jump")


@dataclass(frozen=True)
class CertifyReport:
    """Outcome of comparing a perturbative engine against its fp64 parent."""

    n_cases: int
    max_abs: float  # worst |Delta r_pert - Delta r_exact|, seconds
    max_rel: float  # the same, relative to the size of Delta r
    max_scale: float  # the largest |Delta r| in the suite
    rtol: float
    atol: float
    jacobian_rel: float | None = None
    jacobian_rtol: float = 1e-4

    @property
    def quadratic_coefficient(self) -> float:
        """``max_abs / max_scale^2``, in inverse seconds.

        The assembly (:meth:`PerturbativeEngine._residual_delta`) is first
        order in ``Delta D``; what it drops is second order in it -- the
        spin-frequency drift across the delay change. (While the residual
        divided by the doppler-shifted frequency it also dropped
        ``Delta doppler * Delta D``; the §8 divisor change removed that term.)
        If this number
        is stable while ``max_abs`` grows, the residual is simply too large
        for a first-order assembly, and the delta formulation is not at
        fault -- see the validity note in ``docs/PARITY.md``.
        """
        return self.max_abs / self.max_scale**2 if self.max_scale else 0.0

    @property
    def passed(self) -> bool:
        if self.max_abs > self.rtol * self.max_scale + self.atol:
            return False
        return self.jacobian_rel is None or self.jacobian_rel <= self.jacobian_rtol

    def __str__(self) -> str:  # pragma: no cover - reporting
        verdict = "pass" if self.passed else "FAIL"
        return (
            f"CertifyReport[{verdict}] n={self.n_cases} "
            f"max_abs={self.max_abs:.3e}s max_rel={self.max_rel:.3e} "
            f"quadratic={self.quadratic_coefficient:.3e}/s "
            f"jac_rel={self.jacobian_rel}"
        )


def resolve_live(param_names, live_nonlinear) -> tuple[str, ...]:
    """Turn a mode name or an explicit list into the live axis tuple."""
    if isinstance(live_nonlinear, str):
        if live_nonlinear not in MODES:
            raise ValueError(
                f"unknown perturbative mode {live_nonlinear!r}; "
                f"known: {sorted(MODES)}"
            )
        allowed = MODES[live_nonlinear]
        return tuple(n for n in param_names if n in allowed)
    live = tuple(live_nonlinear)
    unknown = [n for n in live if n not in param_names]
    if unknown:
        raise ValueError(f"not free parameters of this engine: {unknown}")
    refused = [n for n in live if n not in ASTROMETRY_AXES + BINARY_AXES]
    if refused:
        raise UnsupportedModelError(
            f"perturbative live axes {refused}",
            ASTROMETRY_AXES + BINARY_AXES,
        )
    return live


class PerturbativeEngine:
    """float32-capable hybrid residual delta over a restricted live set."""

    def __init__(self, engine, live_nonlinear, *, dtype=None, jit: bool = True):
        self.parent = engine
        self.dtype = jnp.dtype(dtype) if dtype is not None else jnp.dtype(jnp.float64)
        self.param_names = engine.param_names
        self.param_units = engine.param_units
        self.toa_count = engine.toa_count

        self.live_nonlinear = resolve_live(engine.param_names, live_nonlinear)
        self._live = frozenset(self.live_nonlinear)
        index = {name: i for i, name in enumerate(self.param_names)}
        self._nl_index = np.array([index[n] for n in self.live_nonlinear], dtype=int)
        self._lin_index = np.array(
            [i for n, i in index.items() if n not in self._live], dtype=int
        )

        # The linear block is the canonical -M of R9.3 -- the parent's own
        # Jacobian -- so the two halves of the assembly are the same object's
        # derivatives, not two different linearisations.
        self._design = jnp.asarray(
            engine.design_matrix()[:, self._lin_index], dtype=self.dtype
        )
        # The frozen state stays float64: it feeds the *reference* channel,
        # which R11.3 rule 1 keeps in float64 whatever the working dtype is.
        # Only quantities entering the perturbation channel are narrowed, and
        # each of those casts is marked at its site (``Pert._cast``).
        self._frozen = engine.frozen
        self._stages = tuple(
            (name, stage)
            for name, stage in engine.chain.stages
            if name not in _PHASE_STAGES
        )
        reference = engine._reference_correction()
        spin = np.asarray(reference.spin_frequency)
        # The TZR row's phase change is -F_spin(t_tzr) dD_tzr, but the residual
        # divides by F_spin(t_i). The ratio is 1 to a part in 1e9 for an MSP and
        # several percent for a fast-spinning-down pulsar, so it is carried.
        self._tzr_spin_ratio = jnp.asarray(spin[-1] / spin[:-1], dtype=self.dtype)
        self.residual_delta_jax = (
            jax.jit(self._residual_delta) if jit else self._residual_delta
        )

    # --- the delta chain ---------------------------------------------------

    def _wrap(self, name, ref, step):
        """Inject a dual number on the live axes; leave the rest at reference.

        The reference value stays float64 (R11.3 rule 1); only the step --
        which *is* the perturbation -- is narrowed to the working dtype.
        """
        if name not in self._live:
            return ref, 0.0
        step = jnp.asarray(step, dtype=self.dtype)
        ref = jnp.asarray(ref, dtype=jnp.float64)
        zero = jnp.zeros((), dtype=jnp.float64)
        return Pert(ref, step, self.dtype), Pert(zero, step, self.dtype)

    def delay_delta(self, delta_theta):
        """``Delta D`` per row, from the live nonlinear axes alone."""
        delta = jnp.asarray(delta_theta)
        nonlinear = jnp.zeros_like(delta).at[self._nl_index].set(delta[self._nl_index])
        params = self.parent.layout.build(nonlinear, wrap=self._wrap)
        corr = Correction.initial(self._frozen.n_rows)
        for _, stage in self._stages:
            corr = stage(self._frozen, corr, params)
        delay = corr.delay
        if isinstance(delay, Pert):
            return delay.delta
        return jnp.zeros(self._frozen.n_rows, dtype=self.dtype)

    def _residual_delta(self, delta_theta):
        delta = jnp.asarray(delta_theta, dtype=self.dtype)
        linear = -self._design @ delta[self._lin_index]
        if len(self._nl_index) == 0:
            return linear
        dd = self.delay_delta(delta)
        # d psi = -F_spin dD to first order and r = psi / F_spin (§8), so the
        # spin frequency cancels outright for the TOA's own term -- no doppler
        # factor survives. (It did while the residual divided by the
        # doppler-shifted frequency; that factor is gone with the divisor.)
        # The TZR row moves too, and carries its own spin frequency into the
        # ratio.
        tzr = self._tzr_spin_ratio * dd[-1]
        return linear - (dd[:-1] - tzr)

    def residual_delta(self, delta_theta) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.param_names),):
            raise ValueError(
                f"delta_theta has shape {delta.shape}, "
                f"expected ({len(self.param_names)},)"
            )
        return np.asarray(self.residual_delta_jax(delta), dtype=float)

    def residual_jacobian(self) -> np.ndarray:
        zero = jnp.zeros(len(self.param_names), dtype=self.dtype)
        return np.asarray(jax.jacfwd(self._residual_delta)(zero), dtype=float)

    # --- facts (delegated: the reference model is the parent's) ------------

    def reference_theta_exact(self) -> Mapping[str, str]:
        return self.parent.reference_theta_exact()

    def reference_theta(self) -> np.ndarray:
        return self.parent.reference_theta()

    def design_matrix(self, *, source: str = "jacobian") -> np.ndarray:
        return self.parent.design_matrix(source=source)

    def precision_critical_params(self) -> frozenset[str]:
        return self.parent.precision_critical_params()

    def identically_linear_params(self) -> frozenset[str]:
        return self.parent.identically_linear_params()

    @property
    def residual_centering(self):
        return self.parent.residual_centering

    def certify(
        self,
        deltas=None,
        *,
        rtol=1e-5,
        atol=1e-12,
        n_random=32,
        seed=0,
        residual_cap=None,
        scales=(-3.0, -1.0, 1.0, 3.0),
    ):
        """Compare against the fp64 parent on posterior-scale deltas.

        The default cases are +-1 sigma and +-3 sigma on each live axis plus
        random joint draws, all restricted to the live nonlinear axes: the
        linear block is the parent's own design matrix, so including it would
        measure linearisation error rather than the delta formulation.
        """
        from .certify import certify

        return certify(
            self,
            deltas=deltas,
            rtol=rtol,
            atol=atol,
            n_random=n_random,
            seed=seed,
            residual_cap=residual_cap,
            scales=scales,
        )

    def __repr__(self) -> str:
        return (
            f"<vela_jax.PerturbativeEngine {np.dtype(self.dtype).name}: "
            f"live={list(self.live_nonlinear)}>"
        )


__all__ = ["PerturbativeEngine", "CertifyReport", "MODES", "Pert"]
