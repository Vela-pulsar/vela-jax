"""Exception types. Every refusal is explicit; nothing is silently skipped."""

from __future__ import annotations


class VelaJaxError(Exception):
    """Base class for every error this package raises."""


class FreezeError(VelaJaxError):
    """The PINT model/TOAs cannot be frozen into engine arrays."""


class UnsupportedModelError(VelaJaxError):
    """The timing model contains a component or parameter v1 refuses.

    Carries the supported set so the message is actionable.
    """

    def __init__(self, offender: str, supported=()):
        self.offender = offender
        self.supported = tuple(supported)
        message = f"vela-jax does not support {offender}"
        if self.supported:
            message += f"; supported: {', '.join(self.supported)}"
        super().__init__(message)


class Tempo2Error(VelaJaxError):
    """tempo2 could not read the par/tim or produce a usable frozen state."""


class PrecisionError(VelaJaxError):
    """Build-time longdouble arithmetic cannot reach the promised precision."""
