"""Runtime configuration guards.

The engine's error budget (SPEC §4) is written for float64 JAX. We refuse to
build rather than silently produce rounding noise of ~700 ns at F0~800 Hz
over 100 yr (~20 ns on a typical 3 yr MSP span), and we never enable x64
ourselves: flipping ``jax_enable_x64`` after other arrays exist is a
well-known footgun that leaves a process with two incompatible defaults.
"""

from __future__ import annotations

import numpy as np

from .errors import PrecisionError


def require_x64() -> None:
    """Raise unless JAX is in float64 mode."""
    import jax

    if not jax.config.jax_enable_x64:
        raise PrecisionError(
            "vela-jax requires JAX float64. Put\n"
            "    import jax; jax.config.update('jax_enable_x64', True)\n"
            "before creating any JAX array (or set JAX_ENABLE_X64=1)."
        )


def require_longdouble() -> None:
    """Raise unless numpy's longdouble is wider than float64.

    The reference spin phase (SPEC §4.3) is the one build-time computation that
    genuinely needs it: on a platform where ``longdouble is float64`` the
    ``F0*tau - N`` cancellation carries ~700 ns of rounding noise into every
    residual at F0~800 Hz over 100 yr (~20 ns on a 3 yr span). That is silent
    and fatal, so it is a build-time refusal.
    """
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise PrecisionError(
            "numpy.longdouble is not wider than float64 on this platform, so "
            "the reference-phase reduction (SPEC 4.3) cannot be computed to "
            "the engine's advertised precision."
        )
