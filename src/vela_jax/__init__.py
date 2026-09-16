"""vela-jax: Vela.jl's deterministic timing-delay engine, in JAX.

    import jax; jax.config.update("jax_enable_x64", True)
    from vela_jax import Engine

    engine = Engine.from_files("psr.par", "psr.tim")
    r  = engine.residual_delta(delta)          # seconds, param_names order
    J  = engine.residual_jacobian()            # jacfwd of the same function

float64 is required (SPEC §4.8): the residual's error budget is written for it
and enabling x64 late is a footgun, so the engine refuses rather than guesses.

PINT is the default timing package (par/tim, clocks, TDB, ephemerides, pulse
numbers); ``Engine.from_tempo2`` (or ``from_files(..., timing_package="tempo2")``)
swaps in tempo2 for that job without changing any physics. Vela.jl is the
authority for every component; Discovery/Enterprise own all noise.
"""

from __future__ import annotations

from .backend import VelaJaxTimingEngine
from .binary.facts import BinaryFacts
from .engine import Engine
from .errors import (
    FreezeError,
    PrecisionError,
    Tempo2Error,
    UnsupportedModelError,
    VelaJaxError,
)
from .pulsar import TimingPulsar

try:  # pragma: no cover - packaging detail
    from importlib.metadata import version

    __version__ = version("vela-jax")
except Exception:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    "Engine",
    "TimingPulsar",
    "VelaJaxTimingEngine",
    "BinaryFacts",
    "VelaJaxError",
    "FreezeError",
    "Tempo2Error",
    "UnsupportedModelError",
    "PrecisionError",
    "__version__",
]
