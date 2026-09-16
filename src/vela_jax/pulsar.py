"""One pulsar: a frozen record and the engine that produced it.

:class:`TimingPulsar` *holds* a :class:`psrdata.PulsarData` and
the :class:`~vela_jax.engine.Engine` it came from, and forwards the pulsar
attribute names to the record. It does not inherit from the frozen dataclass
(a frozen dataclass plus an engine plus methods fights the type: ``__init__``
order, accidental copies), and it does not recompute or copy anything -- every
forwarded array ``is`` the record's array.

The protocols are structural, so forwarding satisfies nltiming's ``isinstance``
checks. Nothing here imports nltiming.

This is the single-pulsar path, and it says so: a two-PTA combination is
MetaPulsar's job, and a residuals-only look wants
:class:`~vela_jax.engine.Engine` rather than this class -- building a
``TimingPulsar`` computes the Jacobian (R9.6).
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from psrdata import PulsarData

from .backend import VelaJaxTimingEngine
from .engine import Engine

#: The keyword arguments nltiming's ``TimingSpec`` passes, plus the two named
#: ones. Values that belong to another timing engine are ignored *by name*:
#: an unknown keyword raises, so a typo in a contract knob cannot silently
#: no-op (R-B6.4).
ACCEPTED_KWARGS = frozenset(
    {
        "tempo2_native",
        "tempo2_jug_options",
        "prime_sessions",
        "verify_wiring",
        "subtract_tzr",
    }
)


class TimingPulsar:
    """One pulsar's frozen record, with its vela-jax timing engine attached."""

    #: The record fields forwarded onto the pulsar itself.
    FORWARDED = tuple(field.name for field in dataclass_fields(PulsarData))

    def __init__(self, engine: Engine):
        self.engine = engine
        self.data: PulsarData = engine.pulsar_data()
        # Bound as real instance attributes rather than through __getattr__:
        # Python 3.12's runtime-checkable protocol isinstance() reads members
        # with inspect.getattr_static, which never fires __getattr__, so a
        # forwarding hook would silently fail every structural check. Each name
        # is bound to the record's own array -- ``is``-identical, not a copy.
        for name in self.FORWARDED:
            object.__setattr__(self, name, getattr(self.data, name))

    # --- construction ------------------------------------------------------

    @classmethod
    def from_files(cls, par, tim, *, timing_package: str = "pint", **engine_kwargs):
        """Build from par/tim. ``timing_package`` chooses PINT or tempo2."""
        return cls(
            Engine.from_files(par, tim, timing_package=timing_package, **engine_kwargs)
        )

    @classmethod
    def from_pint(cls, model, toas, **engine_kwargs):
        """Freeze a caller-built PINT pair.

        Same contract as :meth:`Engine.from_pint`.
        """
        return cls(Engine.from_pint(model, toas, **engine_kwargs))

    @classmethod
    def from_tempo2(cls, par, tim, **engine_kwargs):
        """Alias for :meth:`from_files` with ``timing_package="tempo2"``."""
        return cls.from_files(par, tim, timing_package="tempo2", **engine_kwargs)

    # --- nltiming's TimingPulsar surface ----------------------------------

    def pint_model(self):
        return self.engine.pint_model

    def can_use_engines(self, engines="vela_jax", **_ignored) -> bool:
        """Answer for exactly what this object is (R-B6.3).

        A single-leg vela-jax pulsar, one timing package. A
        ``{"pint": ..., "tempo2": ...}`` mapping is honoured iff every
        value is ``"vela_jax"``; mixed-engine and multi-PTA requests
        belong to MetaPulsar.
        """
        if isinstance(engines, str):
            return engines == "vela_jax"
        return set(dict(engines).values()) <= {"vela_jax"}

    def timing_engine(
        self,
        engines="vela_jax",
        *,
        derivative_method: str = "analytic",
        nonlinear_params: str | None = None,
        **engine_kwargs,
    ) -> VelaJaxTimingEngine:
        """Return the nltiming-shaped timing engine for this pulsar (B.7).

        ``nonlinear_params`` is the hybrid residual mode, and vela-jax
        *executes* it: the perturbative engine is that formula
        (``dr = -M d_lin - dD_nonlinear``), delta-exact rather than
        hand-linearised. ``None`` keeps every axis on the full nonlinear path.
        The engine reports the mode actually executed, which is what
        nltiming's manifest check reads.

        ``derivative_method`` is honoured rather than ignored: under R9.3 the
        ``"analytic"`` and ``"autodiff"`` routes are the same matrix, so both
        are accepted and recorded; anything else raises.
        """
        unknown = sorted(set(engine_kwargs) - ACCEPTED_KWARGS)
        if unknown:
            raise TypeError(
                f"timing_engine got unexpected keyword arguments {unknown}; "
                f"known keywords are {sorted(ACCEPTED_KWARGS)}"
            )
        if engine_kwargs.get("subtract_tzr"):
            raise ValueError(
                "subtract_tzr=True is incompatible with this engine: it never "
                "mean-subtracts (stored_residuals='none')"
            )
        if not self.can_use_engines(engines):
            raise ValueError(
                f"engines {engines!r} cannot be honoured by a vela-jax pulsar"
            )
        engine: Any = self.engine
        if nonlinear_params is not None:
            engine = engine.perturbative(nonlinear_params)
        return VelaJaxTimingEngine(
            engine,
            name=self.name,
            derivative_method=derivative_method,
            nonlinear_params=nonlinear_params,
        )

    def to_feather(self, path):
        self.data.to_feather(path)
        return path

    def __len__(self) -> int:
        return len(self.data.toas)

    def __repr__(self) -> str:
        return (
            f"<vela_jax.TimingPulsar {self.name}: {len(self)} TOAs, "
            f"{len(self.fitpars)} fitpars, "
            f"timing_package={self.engine.timing_package}>"
        )


__all__ = ["TimingPulsar", "PulsarData"]
