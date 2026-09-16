"""The nltiming ``TimingEngine`` surface, spelled in nltiming's vocabulary.

:class:`~vela_jax.engine.Engine` already *is* the backend nltiming's protocols
describe (SPEC §10); it just uses this package's own names for things --
``param_names`` where nltiming says ``fitpars``, ``param_units`` where it says
``native_units``. This module is the twenty lines that rename them, plus the
one signature difference (``design_matrix(params=None)``).

It deliberately does not import nltiming. The protocols are structural and
``runtime_checkable``, so an object with the right attributes *is* a
``JaxTimingEngine``; mirroring the shape here keeps vela-jax standalone.
Residual centering is serialized in the record, so its type is psrdata's,
which both this package and nltiming import. The protocol text this is
written against is quoted in SPEC §10.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Any, Mapping

import numpy as np
from psrdata.record import PHASE_OFFSET_RE


class VelaJaxTimingEngine:
    """``nltiming.protocols.JaxTimingEngine`` over a :class:`Engine`.

    Also satisfies ``JacobianTimingEngine`` (``residual_jacobian``) and carries
    the two optional capability hooks nltiming looks for by name.
    """

    engine_name = "vela_jax"

    #: ``derivative_method`` values this engine accepts. Under R9.3 the two
    #: routes are the *same* matrix -- the design matrix is ``-J`` -- so the
    #: knob stops selecting between two truths and becomes manifest metadata.
    #: It is still recorded, because the manifest must describe what ran.
    DERIVATIVE_METHODS = ("analytic", "autodiff")

    def __init__(
        self,
        engine,
        *,
        name: str | None = None,
        derivative_method: str = "analytic",
        nonlinear_params: str | None = None,
    ):
        if derivative_method not in self.DERIVATIVE_METHODS:
            raise ValueError(
                f"derivative_method must be one of {list(self.DERIVATIVE_METHODS)}; "
                f"got {derivative_method!r}"
            )
        self.engine = engine
        model = getattr(engine, "pint_model", None) or getattr(
            getattr(engine, "parent", None), "pint_model", None
        )
        self.name = name or (str(model["PSR"].value or "") if model else "")
        self.fitpars = tuple(engine.param_names)
        self.native_units = dict(engine.param_units)
        #: The route recorded in nltiming's run manifest (R10.2).
        self.derivative_method = derivative_method
        #: The hybrid residual mode this engine actually executes (R10.3).
        self.nonlinear_params = nonlinear_params

    # --- TimingEngine ------------------------------------------------------

    def reference_theta(self) -> np.ndarray:
        """Public fit coordinates, matching the record (R-3.2.2, R-3.3.5).

        Phase-offset columns are a delta around the stored residuals, so
        ``PHOFF`` is decimal zero here. The JAX chain still holds the PINT
        value internally; :meth:`residual_delta` is unchanged.
        """
        exact = self.reference_theta_exact()
        with localcontext() as ctx:
            ctx.prec = 60
            return np.array(
                [float(Decimal(exact[name])) for name in self.fitpars],
                dtype=float,
            )

    def reference_theta_exact(self) -> Mapping[str, str]:
        exact = dict(self.engine.reference_theta_exact())
        for name in exact:
            if PHASE_OFFSET_RE.match(name):
                exact[name] = "0"
        return exact

    def residual_delta(self, delta_theta) -> np.ndarray:
        return self.engine.residual_delta(delta_theta)

    def design_matrix(self, params: Any | None = None) -> np.ndarray:
        """The canonical ``-J`` matrix (R9.3); frozen at ``theta*``."""
        _ = params
        return self.engine.design_matrix()

    @property
    def residual_centering(self):
        """Vela removes no mean; one data-set mapping for a standalone pulsar."""
        from psrdata import SINGLE_KEY, ResidualCentering

        centering = self.engine.residual_centering
        if isinstance(centering, ResidualCentering):
            return {SINGLE_KEY: centering}
        return dict(centering)

    # --- JacobianTimingEngine / JaxTimingEngine ---------------------------

    def residual_jacobian(self) -> np.ndarray:
        return self.engine.residual_jacobian()

    def residual_delta_jax(self, delta_theta):
        return self.engine.residual_delta_jax(delta_theta)

    def precision_critical_fitpars(self) -> frozenset[str]:
        return self.engine.precision_critical_params()

    # --- optional capability hooks ----------------------------------------

    def binary_chart_capability(self, chart_family: str, suffix: str):
        """Authoritative chart facts, from the engine's own binary stage.

        ``origin_certified`` is False: no backend may claim the empirical
        origin certification without a recorded certification run.
        """
        if chart_family != "kepler_laplace":
            return None
        facts = _binary_facts(self.engine)
        if facts is None:
            return None
        return _ChartCapability(
            kepler_convention=facts.kepler_convention,
            epoch_shift_exact=facts.epoch_shift_exact,
            secular_terms=facts.secular_terms,
            origin_certified=False,
            supports_domain=True,
        )

    def identically_linear_fitpars(self) -> frozenset[str]:
        return self.engine.identically_linear_params()

    def gauge_direction(self) -> np.ndarray:
        """Forwarded from the engine, which is where the number lives.

        Hoisted onto :class:`~vela_jax.engine.Engine` so a consumer -- a
        MetaPulsar leg, say -- can ask for the gauge direction without
        constructing an nltiming adapter to get at it.
        """
        engine = self.engine
        if not hasattr(engine, "gauge_direction"):
            engine = engine.parent
        return engine.gauge_direction()

    def __repr__(self) -> str:
        return (
            f"<vela_jax.VelaJaxTimingEngine {self.name}: {len(self.fitpars)} fitpars>"
        )


def _binary_facts(engine):
    """Binary facts, reaching through a perturbative engine to its parent."""
    facts = getattr(engine, "binary_chart_facts", None)
    if facts is not None:
        return facts()
    parent = getattr(engine, "parent", None)
    return parent.binary_chart_facts() if parent is not None else None


class _ChartCapability:
    """Mirror of ``nltiming.protocols.BinaryChartCapability`` (SPEC §10).

    The mirror enforces the same invariant the real dataclass does: a
    certification claim without auditable provenance is invalid, so
    ``origin_certified=True`` without a ``certification_ref`` raises. A mirror
    that is *weaker* than the contract it mirrors is worse than no mirror
    (R-B7.2).
    """

    __slots__ = (
        "kepler_convention",
        "epoch_shift_exact",
        "secular_terms",
        "origin_certified",
        "supports_domain",
        "certification_ref",
    )

    def __init__(
        self,
        *,
        kepler_convention,
        epoch_shift_exact,
        secular_terms,
        origin_certified,
        supports_domain,
        certification_ref=None,
    ):
        self.kepler_convention = kepler_convention
        self.epoch_shift_exact = epoch_shift_exact
        self.secular_terms = secular_terms
        if origin_certified and not certification_ref:
            raise ValueError(
                "BinaryChartCapability: origin_certified=True requires a "
                "certification_ref (the recorded certification run/PR)"
            )
        self.origin_certified = origin_certified
        self.supports_domain = supports_domain
        self.certification_ref = certification_ref

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BinaryChartCapability({self.kepler_convention}, "
            f"epoch_shift_exact={self.epoch_shift_exact})"
        )
