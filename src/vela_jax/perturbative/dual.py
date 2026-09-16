"""A value type that carries ``(reference, perturbation)`` instead of a number.

This is the whole single-precision story (SPEC §11) in one class. Every rule
below is the cancellation-free difference identity for one operation, so an
expression evaluated over :class:`Pert` never forms an absolute quantity in the
perturbation channel: ``sin`` uses the half-angle form, ``log`` uses
``log1p``, ``sqrt`` uses the ``1/(1+sqrt(1+eps))`` form, ``atan2`` uses the
angle-difference identity, and Kepler's equation is solved *in the difference
variable*.

The consequence is that ``ΔD`` has the relative accuracy of ``ΔD`` itself, not
of ``D``: a 1 microsecond change in a 500 second Roemer delay is still good to
~1e-13 s in float32, where evaluating the delay twice and subtracting would
leave 3e-5 s of cancellation noise.

**Two dtypes, one physics (R11.3 rule 1).** The reference channel ``ref`` is
float64 *always*, whatever the working dtype is; only the perturbation channel
``delta`` is narrowed. Reference factors are cast to the working dtype at the
point where they enter the perturbation channel, never before -- which is what
:meth:`Pert._cast` marks at every site. The reference depends on nothing
traced, so under ``jit`` XLA constant-folds it and the extra precision is free
at run time. v1 violated this: it recomputed the reference in the working
dtype, and a strongly cancelling reference (DDS ``SHAPMAX ~ 9``,
``sin i = 0.99989``, whose Shapiro log argument sits at 1e-4) poisoned the
float32 delta on ``J2302+4442``.

Because the component chain is written against :mod:`vela_jax.numerics`, this
type reuses that chain verbatim. There is no second copy of the physics.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..numerics import Numeric


class Pert(Numeric):
    """``x = ref + delta``: ``ref`` in float64, ``delta`` in the working dtype."""

    __slots__ = ("ref", "delta", "dtype")

    def __init__(self, ref, delta, dtype=None):
        self.ref = ref
        self.delta = delta
        self.dtype = jnp.result_type(delta) if dtype is None else jnp.dtype(dtype)

    def lift_like(self, x) -> "Pert":
        """``x`` as a :class:`Pert` with a zero perturbation of this dtype."""
        if isinstance(x, Pert):
            return x
        return Pert(x, jnp.zeros((), dtype=self.dtype), self.dtype)

    def _cast(self, x):
        """A reference quantity entering the perturbation channel (R11.3-1)."""
        return jnp.asarray(x).astype(self.dtype)

    def _new(self, ref, delta) -> "Pert":
        return Pert(ref, delta, self.dtype)

    @property
    def value(self):
        """The perturbed value ``ref + delta``.

        For a traced domain predicate or a report — never arithmetic in the
        perturbation channel.
        """
        return self.ref + self.delta

    @property
    def reference(self):
        """The float64 reference channel. Kepler convergence reads this."""
        return self.ref

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Pert(ref={self.ref!r}, delta={self.delta!r})"

    # --- linear structure --------------------------------------------------

    def __add__(self, other):
        o = self.lift_like(other)
        # The reference sum happens in float64, which is where a cancelling
        # reference (the DDS Shapiro log argument, say) has to happen.
        return self._new(self.ref + o.ref, self.delta + o.delta)

    __radd__ = __add__

    def __neg__(self):
        return self._new(-self.ref, -self.delta)

    def __sub__(self, other):
        return self + (-self.lift_like(other))

    def __rsub__(self, other):
        return self.lift_like(other) + (-self)

    # --- products ----------------------------------------------------------

    def __mul__(self, other):
        o = self.lift_like(other)
        # d(ab) = a' db + b* da -- one perturbed factor, one reference factor.
        a_perturbed = self._cast(self.ref) + self.delta
        return self._new(
            self.ref * o.ref,
            a_perturbed * o.delta + self._cast(o.ref) * self.delta,
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "Pert":
        # d(1/b) = -db / (b* b')
        ref = self._cast(self.ref)
        return self._new(1.0 / self.ref, -self.delta / (ref * (ref + self.delta)))

    def __truediv__(self, other):
        return self * self.lift_like(other).reciprocal()

    def __rtruediv__(self, other):
        return self.lift_like(other) * self.reciprocal()

    def __pow__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise TypeError("Pert supports non-negative integer powers only")
        result = self._new(
            jnp.ones_like(jnp.asarray(self.ref)), jnp.zeros((), dtype=self.dtype)
        )
        for _ in range(n):
            result = result * self
        return result

    # --- transcendentals ---------------------------------------------------

    def sin(self):
        s, c = self._cast(jnp.sin(self.ref)), self._cast(jnp.cos(self.ref))
        cm1, sx = _cos_minus_one(self.delta), jnp.sin(self.delta)
        return self._new(jnp.sin(self.ref), s * cm1 + c * sx)

    def cos(self):
        s, c = self._cast(jnp.sin(self.ref)), self._cast(jnp.cos(self.ref))
        cm1, sx = _cos_minus_one(self.delta), jnp.sin(self.delta)
        return self._new(jnp.cos(self.ref), c * cm1 - s * sx)

    def log(self):
        return self._new(
            jnp.log(self.ref), jnp.log1p(self.delta / self._cast(self.ref))
        )

    def exp(self):
        ref = jnp.exp(self.ref)
        return self._new(ref, self._cast(ref) * jnp.expm1(self.delta))

    def sqrt(self):
        root = jnp.sqrt(self.ref)
        eps = self.delta / self._cast(self.ref)
        # sqrt(1+eps) - 1, written so that no cancellation occurs for small eps.
        rel = eps / (1.0 + jnp.sqrt(1.0 + eps))
        return self._new(root, self._cast(root) * rel)

    def cbrt(self):
        """``cbrt(a+da) - cbrt(a)`` via ``b - a = (b^3 - a^3)/(b^2 + ab + a^2)``.

        Subtracting two cube roots cancels the whole root. The caller
        substitutes a positive value before the cube root.
        """
        a = jnp.cbrt(self.ref)
        b = jnp.cbrt(self._cast(self.ref) + self.delta)
        den = b * b + b * self._cast(a) + self._cast(a * a)
        return self._new(a, self.delta / den)

    def arccos(self):
        """Third-order expansion of ``arccos(c*+dc) - arccos(c*)``.

        There is no clean cancellation-free closed form here, and none is
        needed: ``arccos`` appears only in the solar-wind sun angle, where
        ``dc`` is the perturbation of a unit vector's projection (~1e-9 for any
        posterior-scale astrometric delta), so the O(dc^4) truncation is below
        1e-36 rad.

        At ``|c*|=1`` the expansion is singular (``1/sqrt(1-c^2)``). The only
        way the chain lands there is the unit-vector clip of ``cos rho``; the
        clip identity then zeros the perturbation, and both branches of the
        ``where`` must stay finite or ``jacfwd`` is NaN.
        """
        c = self._cast(self.ref)
        d = self.delta
        q = 1.0 - c * c
        interior = q > 0.0
        q_safe = jnp.where(interior, q, 1.0)
        root = jnp.sqrt(q_safe)
        first = -d / root
        second = -0.5 * c * d * d / q_safe**1.5
        third = -(1.0 / 6.0) * (1.0 + 2.0 * c * c) * d**3 / q_safe**2.5
        return self._new(
            jnp.arccos(jnp.clip(self.ref, -1.0, 1.0)),
            jnp.where(interior, first + second + third, 0.0 * d),
        )

    def arctan2(self, x):
        """``atan2(y', x') - atan2(y*, x*)`` as one angle difference.

        Written on the *normalised* numerator and denominator: the raw form
        ``y'x* - x'y*`` cancels the whole ``y*x*`` product, so it is expanded
        by hand into ``dy x* - dx y*`` and both halves are divided by
        ``x*^2 + y*^2``. Nothing in the perturbation channel is then larger
        than one.
        """
        xp = self.lift_like(x)
        y_ref, x_ref = self._cast(self.ref), self._cast(xp.ref)
        norm = self._cast(self.ref * self.ref + xp.ref * xp.ref)
        numerator = (self.delta * x_ref - xp.delta * y_ref) / norm
        denominator = 1.0 + (xp.delta * x_ref + self.delta * y_ref) / norm
        return self._new(
            jnp.arctan2(self.ref, xp.ref), jnp.arctan2(numerator, denominator)
        )

    def kepler(self, e, solver):
        """Eccentric-anomaly perturbation, solved in the difference variable.

        ``u' - e' sin u' = l'`` with ``u' = u* + x`` reduces to
        ``x - e'[sin u*(cos x - 1) + cos u* sin x] = dl + de sin u*``,
        whose right-hand side is small by construction. Four Newton steps from
        the linear guess converge to machine precision for any posterior-scale
        perturbation. The reference solve is float64; only the difference
        iteration runs in the working dtype.
        """
        ep = self.lift_like(e)
        u_ref = solver(self.ref, ep.ref)
        su, cu = self._cast(jnp.sin(u_ref)), self._cast(jnp.cos(u_ref))
        e_new = self._cast(ep.ref) + ep.delta
        rhs = self.delta + ep.delta * su

        x = rhs / (1.0 - e_new * cu)
        for _ in range(4):
            sx, cm1 = jnp.sin(x), _cos_minus_one(x)
            g = x - e_new * (su * cm1 + cu * sx) - rhs
            # d/dx of the left-hand side: 1 - e' cos(u* + x), expanded so no
            # absolute angle is re-formed in the working dtype.
            slope = 1.0 - e_new * (cu * (1.0 + cm1) - su * sx)
            x = x - g / slope
        return self._new(u_ref, x)

    def regular_kepler(self, h, k, solver):
        """Laplace-Lagrange perturbation, solved in the difference variable.

        ``F' = F* + x``; ``sin F*`` and ``cos F*`` enter as reference factors.
        A perturbation may cross the principal-2π cut of the reference solve;
        this follows the nearby continuous root (downstream uses of ``F`` are
        2π-periodic; precession reads ``lam_secular`` instead).
        """
        hp = self.lift_like(h)
        kp = self.lift_like(k)
        F_ref = solver(self.ref, hp.ref, kp.ref)
        sin_F = self._cast(jnp.sin(F_ref))
        cos_F = self._cast(jnp.cos(F_ref))
        h_new = self._cast(hp.ref) + hp.delta
        k_new = self._cast(kp.ref) + kp.delta

        rhs = self.delta + kp.delta * sin_F - hp.delta * cos_F
        x = rhs / (1.0 - k_new * cos_F - h_new * sin_F)
        for _ in range(4):
            sin_x = jnp.sin(x)
            cos_x_minus_one = _cos_minus_one(x)
            delta_sin = sin_F * cos_x_minus_one + cos_F * sin_x
            delta_cos = cos_F * cos_x_minus_one - sin_F * sin_x
            g = (
                x
                - k_new * delta_sin
                - kp.delta * sin_F
                + h_new * delta_cos
                + hp.delta * cos_F
                - self.delta
            )
            sin_new = sin_F * (1.0 + cos_x_minus_one) + cos_F * sin_x
            cos_new = cos_F * (1.0 + cos_x_minus_one) - sin_F * sin_x
            slope = 1.0 - k_new * cos_new - h_new * sin_new
            x = x - g / slope
        return self._new(F_ref, x)

    # --- structure ---------------------------------------------------------

    def select(self, cond, other):
        o = self.lift_like(other)
        return self._new(
            jnp.where(cond, self.ref, o.ref),
            jnp.where(cond, self.delta, o.delta),
        )

    def nan_where(self, valid):
        """NaN in *both* channels where ``valid`` is false.

        ``select(valid, self, nan)`` gives the scalar NaN a zero perturbation.
        """
        return self._new(
            jnp.where(valid, self.ref, jnp.nan),
            jnp.where(valid, self.delta, jnp.nan),
        )

    def clip(self, lo, hi):
        """``clip(x*+dx) - clip(x*)``.

        Forming ``x*+dx`` is safe: the only call site is a unit-vector cosine,
        O(1). When the clip is inactive the identity is ``delta`` itself --
        subtracting two clipped values would reintroduce the cancellation
        these identities exist to avoid. Keeping the unclipped perturbation
        after clipping the reference (the previous body) left ``arccos`` with
        ``c*=±1`` and a nonzero ``dc``, which is Inf/NaN.
        """
        clipped_ref = jnp.clip(self.ref, lo, hi)
        perturbed = self._cast(self.ref) + self.delta
        clipped_perturbed = jnp.clip(perturbed, lo, hi)
        inactive = (clipped_ref == self.ref) & (clipped_perturbed == perturbed)
        bound_delta = clipped_perturbed - self._cast(clipped_ref)
        return self._new(clipped_ref, jnp.where(inactive, self.delta, bound_delta))


def _cos_minus_one(x):
    """``cos(x) - 1`` without cancellation: ``-2 sin^2(x/2)``."""
    half = jnp.sin(0.5 * x)
    return -2.0 * half * half
