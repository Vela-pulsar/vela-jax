"""The numeric interface the component chain is written against.

Every physics module in this package computes with ``+ - * /`` and the
functions below, never with ``jnp.sin`` directly. For the ordinary float64
engine the functions *are* ``jnp``; the indirection costs nothing.

It buys one thing, and it is the reason the single-precision engine (SPEC §11)
is not a second copy of the physics: a value type may implement
:class:`Numeric` and carry ``(reference, perturbation)`` instead of a number,
with each rule below written as a cancellation-free difference identity
(SPEC §11.4). Running the *same* component chain over that type yields
``ΔD(δ)`` directly, with no absolute quantity ever formed.

Three-vectors are 3-tuples of ``(R,)`` arrays, mirroring Vela's
``NTuple{3,GQ}``, so the same code works for both value types.
"""

from __future__ import annotations

import abc
from typing import Any, Tuple

import jax.numpy as jnp

#: A three-vector, as Vela's `NTuple{3,GQ}`: three (R,)-shaped components,
#: so the same code works for arrays and for any :class:`Numeric`.
Vec3 = Tuple[Any, Any, Any]


class Numeric(abc.ABC):
    """A value that overrides the elementwise algebra of this package.

    Implementations are responsible for staying accurate under their own
    representation; the component chain makes no assumptions beyond these
    methods and the Python arithmetic operators.
    """

    @abc.abstractmethod
    def sin(self): ...

    @abc.abstractmethod
    def cos(self): ...

    @abc.abstractmethod
    def log(self): ...

    @abc.abstractmethod
    def exp(self): ...

    @abc.abstractmethod
    def sqrt(self): ...

    @abc.abstractmethod
    def arccos(self): ...

    @abc.abstractmethod
    def arctan2(self, x): ...

    @abc.abstractmethod
    def cbrt(self): ...

    @abc.abstractmethod
    def kepler(self, e, solver): ...

    @abc.abstractmethod
    def regular_kepler(self, h, k, solver): ...

    @abc.abstractmethod
    def select(self, cond, other): ...

    @abc.abstractmethod
    def nan_where(self, valid):
        """NaN wherever ``valid`` is false, in *every* channel.

        ``select(valid, self, nan)`` is not this: lifting a scalar NaN gives
        it a zero perturbation, so an invalid point would look like no change.
        """

    @property
    @abc.abstractmethod
    def reference(self):
        """The float64 reference channel. Convergence tests only, never a delta."""

    @property
    @abc.abstractmethod
    def value(self):
        """The perturbed value. Predicates and reporting only, never a delta."""

    @abc.abstractmethod
    def clip(self, lo, hi): ...

    @abc.abstractmethod
    def lift_like(self, x):
        """``x`` as this value type, matching this instance's representation.

        The dispatch helpers below use it when a plain array meets a
        :class:`Numeric` on the other side of a binary operation, so that the
        implementation -- not this module -- decides what a constant means
        (for the perturbative dual: a zero perturbation, in *its* dtype).
        """


def _is(x) -> bool:
    return isinstance(x, Numeric)


def sin(x):
    return x.sin() if _is(x) else jnp.sin(x)


def cos(x):
    return x.cos() if _is(x) else jnp.cos(x)


def sincos(x):
    """Vela writes ``sincos(x)``; the pair is used together everywhere."""
    return sin(x), cos(x)


def log(x):
    return x.log() if _is(x) else jnp.log(x)


def exp(x):
    return x.exp() if _is(x) else jnp.exp(x)


def sqrt(x):
    return x.sqrt() if _is(x) else jnp.sqrt(x)


def cbrt(x):
    """``x ** (1/3)``. Vela's GR maps are the only one-third powers here.

    ``Pert.__pow__`` deliberately takes non-negative integers only; this is
    the one fractional power the chain needs, not a general ``__pow__``.
    """
    return x.cbrt() if _is(x) else jnp.cbrt(x)


def arccos(x):
    return x.arccos() if _is(x) else jnp.arccos(x)


def arctan2(y, x):
    if _is(y):
        return y.arctan2(x)
    if _is(x):
        return x.lift_like(y).arctan2(x)
    return jnp.arctan2(y, x)


def clip(x, lo, hi):
    return x.clip(lo, hi) if _is(x) else jnp.clip(x, lo, hi)


def where(cond, a, b):
    """Select ``a`` where ``cond``, else ``b``.

    ``cond`` may be traced. Apply the branch to every channel, and substitute
    a finite value *before* a singular expression — do not select a NaN away
    afterwards.
    """
    if _is(a):
        return a.select(cond, b)
    if _is(b):
        return b.lift_like(a).select(cond, b)
    return jnp.where(cond, a, b)


def kepler(l, e, solver):
    """Eccentric anomaly ``u`` with ``u - e sin u = l``.

    ``solver`` is the plain-array Kepler solver (Mikkola); a :class:`Numeric`
    may use it for its reference channel and solve for the perturbation in the
    difference variable instead.
    """
    if _is(l) or _is(e):
        source = l if _is(l) else e
        return source.lift_like(l).kepler(e, solver)
    return solver(l, e)


def regular_kepler(lam, h, k, solver):
    """``F`` with ``F - k sin F + h cos F = lam`` (Laplace-Lagrange).

    Not the polar equation :func:`kepler` takes. A :class:`Numeric` may solve
    the perturbation in the difference variable.
    """
    if _is(lam) or _is(h) or _is(k):
        source = lam if _is(lam) else (h if _is(h) else k)
        return source.lift_like(lam).regular_kepler(h, k, solver)
    return solver(lam, h, k)


# --- channel access and traced validity ------------------------------------


def value(x):
    """The perturbed value: a traced domain predicate or a report, never a delta."""
    return x.value if _is(x) else x


def reference(x):
    """The float64 reference channel; the solver convergence test uses it."""
    return x.reference if _is(x) else x


def isfinite(x):
    """Elementwise finiteness of the *sampled* value, as a traced predicate."""
    return jnp.isfinite(value(x))


def nan_where(valid, x):
    """NaN wherever ``valid`` is false, in every channel ``x`` carries."""
    return x.nan_where(valid) if _is(x) else jnp.where(valid, x, jnp.nan)


# --- three-vector helpers (Vela's NTuple{3} arithmetic) ---------------------


def dot3(a: Vec3, b: Vec3):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm3(a: Vec3):
    return sqrt(dot3(a, a))


def add3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale3(s, a: Vec3) -> Vec3:
    return (s * a[0], s * a[1], s * a[2])


def where3(cond, a: Vec3, b: Vec3) -> Vec3:
    return tuple(where(cond, ai, bi) for ai, bi in zip(a, b))
