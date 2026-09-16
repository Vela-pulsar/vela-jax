"""The Damour-Deruelle-Regular family: DDR.

Vela: ``src/model/binary/binary_ddr.jl``. A third family beside DD and ELL1:
regular Laplace-Lagrange ``F - k sin F + h cos F = lam`` in ``(EPS1, EPS2,
TASC)``, a Cartesian projector frozen at ``TGEO``, and periapsis advance
through ``q = nu - M``.

Deviations from Vela, marked at the site: R4.5 / R4.5b orbit reduction;
traced validity (mask, substitute, ``nan_where``); a 16-pass Kepler with an
implicit-function JVP; ``Pert.regular_kepler`` / ``Pert.cbrt``; geometry uses
the par's resolved ``ECL``. Model formulas go through
:mod:`vela_jax.numerics`; direct ``jnp`` is only in the solver kernel.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .. import numerics as vm
from ..astrometry import ecliptic_to_equatorial, equatorial_to_ecliptic
from ..constants import M_SUN
from ..correction import Correction, corrected_time
from ..taylor import factorial_series, taylor_horner

PI = jnp.pi
TWO_PI = 2.0 * PI

#: Vela ``binary_ddr.jl`` galaxy constants, internal units. Local to this
#: model; ``freeze.PINNED_PARAMS`` refuses a par that moves them.
DDR_R0 = 8.417380286943844e11
DDR_THETA0_OVER_C = 0.0007338410094359345
DDR_RHO0 = 4.517103049894966e-31
DDR_Z0 = 1.852688250978102e10
DDR_ZSUN = 2.05854250108678e9

#: Vela ``ICRS_TO_GAL``: rows of the ICRS-to-Galactic rotation.
ICRS_TO_GAL = (
    (-0.054875657712591633, -0.87343705195561583, -0.48383507361671546),
    (0.49410943719272682, -0.44482972122329512, 0.7469821839866676),
    (-0.8676661375596576, -0.19807633727300059, 0.45598381368730162),
)


class DDRConfig(NamedTuple):
    """Vela's ``BinaryDDR`` struct: six static mode flags, plus the obliquity.

    Bound at build time, never traced. ``obliquity`` is the same value
    ``solar_system`` uses.
    """

    use_fbx: bool
    ecliptic_coordinates: bool
    use_pk: bool
    pbdot_kinematic: bool
    use_geo: bool
    use_kine: bool
    obliquity: float


class DDRState(NamedTuple):
    """Vela ``DDRState``, in Vela's field order.

    Each field is an ``(R,)`` array over the TOAs plus the TZR row. ``valid``
    is that same ``R``-row mask: residuals are
    ``(psi[:-1] - psi[-1]) / F_spin[:-1]``, so an invalid TZR row NaNs every
    residual through the phase offset even when every science TOA is inside
    the domain.
    """

    x: Any
    n: Any
    c: Any
    s: Any
    c_e: Any
    s_e: Any
    g_gamma: Any
    X: Any
    Y: Any
    dX: Any
    dY: Any
    d2X: Any
    d2Y: Any
    I: Any
    J: Any
    B_S: Any
    m2: Any
    valid: Any


# --- traced-validity helpers ------------------------------------------------


def _valid_number(x):
    """Vela ``_ddr_isfinite``, on the *sampled* value."""
    return vm.isfinite(x)


def _all_valid(*xs):
    """Vela ``_ddr_allfinite``."""
    ok = _valid_number(xs[0])
    for x in xs[1:]:
        ok = ok & _valid_number(x)
    return ok


def _safe(ok, value, fallback):
    """``value`` where ``ok``, else a finite substitute.

    ``ok`` must already have been folded into the state's ``valid`` mask: a
    predicate derived from a value that was *itself* substituted is not a
    predicate about the model. The fallback is a constant, never
    ``0.0 * value + fallback`` -- exactly where ``value`` is NaN, ``0.0 * NaN``
    is still NaN and the supposedly safe branch is poisoned.
    """
    return vm.where(ok, value, fallback)


# --- the regular Kepler equation -------------------------------------------


def reduce_longitude(lam):
    """Vela ``reduce_longitude``: nearest-orbit rounding to a principal branch."""
    n_orb = jnp.round(lam / TWO_PI)
    return lam - n_orb * TWO_PI, n_orb


@jax.custom_jvp
def _solve_F_value(lam, h, k):
    """Vela ``solve_F``'s Newton/bracket update, 16 array passes.

    A traced array cannot break, so converged rows freeze via ``done``.
    Sixteen is measured (PARITY.md); the bracket moves only on bisection.
    """
    lam_red, _ = reduce_longitude(lam)
    lo = lam_red - 1.0
    hi = lam_red + 1.0
    F = jnp.clip(lam_red + k * jnp.sin(lam_red) - h * jnp.cos(lam_red), lo, hi)
    atol = 4.0 * jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, jnp.abs(lam_red))
    done = jnp.zeros_like(F, dtype=bool)

    for _ in range(16):
        sin_F, cos_F = jnp.sin(F), jnp.cos(F)
        residual = F - k * sin_F + h * cos_F - lam_red
        converged = jnp.abs(residual) <= atol
        D = 1.0 - k * cos_F - h * sin_F
        trial = F - residual / D
        good = jnp.isfinite(trial) & (lo <= trial) & (trial <= hi)

        sin_lo, cos_lo = jnp.sin(lo), jnp.cos(lo)
        residual_lo = lo - k * sin_lo + h * cos_lo - lam_red
        go_hi = residual_lo * residual > 0.0
        lo_new = jnp.where(go_hi, F, lo)
        hi_new = jnp.where(go_hi, hi, F)
        next_F = jnp.where(good, trial, 0.5 * (lo_new + hi_new))
        next_lo = jnp.where(good, lo, lo_new)
        next_hi = jnp.where(good, hi, hi_new)
        done = done | converged
        F = jnp.where(done, F, next_F)
        lo = jnp.where(done, lo, next_lo)
        hi = jnp.where(done, hi, next_hi)
    return F


@_solve_F_value.defjvp
def _solve_F_value_jvp(primals, tangents):
    """Implicit differentiation of ``F - k sin F + h cos F = lam``.

    Differentiating the loop itself would differentiate branch predicates and
    a data-dependent convergence history; the implicit derivative is exact for
    the root the primal actually returned.
    """
    lam, h, k = primals
    dlam, dh, dk = tangents
    F = _solve_F_value(lam, h, k)
    sin_F, cos_F = jnp.sin(F), jnp.cos(F)
    D = 1.0 - k * cos_F - h * sin_F
    dF = (dlam + sin_F * dk - cos_F * dh) / D
    return F, jnp.broadcast_to(dF, jnp.shape(F))


def solve_F(lam, h, k):
    """Vela ``solve_F``: ``(F, D, c_e, s_e, converged)``.

    Convergence is read off the float64 *reference* channel. The perturbative
    difference solve is float32-capable and cannot meet a float64-epsilon
    residual bound.
    """
    F = vm.regular_kepler(lam, h, k, _solve_F_value)
    sin_F, cos_F = vm.sincos(F)
    D = 1.0 - k * cos_F - h * sin_F
    c_e = k * cos_F + h * sin_F
    s_e = k * sin_F - h * cos_F

    F_ref = vm.reference(F)
    h_ref = vm.reference(h)
    k_ref = vm.reference(k)
    lam_red, _ = reduce_longitude(vm.reference(lam))
    residual = F_ref - k_ref * jnp.sin(F_ref)
    residual = residual + h_ref * jnp.cos(F_ref) - lam_red
    atol = 4.0 * jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, jnp.abs(lam_red))
    return F, D, c_e, s_e, jnp.abs(residual) <= atol


# --- the regular orbit ------------------------------------------------------


def static_XY(F, h, k):
    """Vela ``static_XY``: the frozen-element projections and two derivatives."""
    E2 = h * h + k * k
    eta = vm.sqrt(1.0 - E2)
    b = 1.0 / (1.0 + eta)
    sin_F, cos_F = vm.sincos(F)
    Y0 = (1.0 - b * k * k) * sin_F + b * h * k * cos_F - h
    X0 = (1.0 - b * h * h) * cos_F + b * h * k * sin_F - k
    dY0 = (1.0 - b * k * k) * cos_F - b * h * k * sin_F
    d2Y0 = -(1.0 - b * k * k) * sin_F - b * h * k * cos_F
    dX0 = -(1.0 - b * h * h) * sin_F + b * h * k * cos_F
    d2X0 = -(1.0 - b * h * h) * cos_F - b * h * k * sin_F
    return X0, Y0, dX0, dY0, d2X0, d2Y0


def q_nu_minus_M(c_e, s_e, E2):
    """Vela ``q_nu_minus_M``: the regular ``nu - M``."""
    return s_e + 2.0 * vm.arctan2(s_e, 1.0 + vm.sqrt(1.0 - E2) - c_e)


def q_at_tasc(h, k):
    """Vela ``q_at_tasc``, with the solver's convergence flag carried out."""
    _, _, c_e, s_e, converged = solve_F(0.0 * h, h, k)
    return q_nu_minus_M(c_e, s_e, h * h + k * k), converged


def precession_delta(lam, q, q_star, kappa):
    """Vela ``precession_delta``. ``lam`` here is unwrapped (R4.5b)."""
    return kappa * (lam + q - q_star)


def rotate_XY(X0, Y0, delta):
    """Vela ``rotate_XY``."""
    sin_delta, cos_delta = vm.sincos(delta)
    return (
        X0 * cos_delta - Y0 * sin_delta,
        Y0 * cos_delta + X0 * sin_delta,
    )


def sini_from_cosi(c):
    """Vela ``sini_from_cosi``, written so neither factor cancels."""
    return vm.sqrt((1.0 - c) * (1.0 + c))


def mean_longitude(fb, dt_full, dt_red, PB, dPB, pbdot):
    """Vela ``mean_longitude``, with the frozen integer orbit count divided out.

    ``dt/PB = n_orb + dt_red/PB - n_orb*dPB/PB`` (mod 1). The integer drops
    out of every downstream trig function; ``n_orb*dPB`` stays because that
    is how a live ``PB`` moves late-time phase. ``PBDOT`` is small enough
    to take the full ``dt``.
    """
    phase = dt_red / PB - fb.n_orb * dPB / PB
    phase = phase - 0.5 * pbdot * (dt_full / PB) ** 2
    lamdot = TWO_PI * (1.0 / PB - pbdot * dt_full / (PB * PB))
    return TWO_PI * phase, lamdot


def fbx_mean_longitude(fb, dt_full, dt_red, FB):
    """Vela's FBX phase branch, reduced the same way.

    ``FB0*(n P* + dt_red) = n + n*(FB0 P* - 1) + FB0*dt_red``. The product
    ``FB0*P*`` rounds to exactly 1.0; ``n_orb`` multiplies that residual.
    """
    phase = (
        FB[0] * dt_red
        + fb.n_orb * (FB[0] * fb.period_ref_s - 1.0)
        + factorial_series(dt_full, FB[1:], 2)
    )
    return TWO_PI * phase, TWO_PI * taylor_horner(dt_full, FB)


# --- mass and period-derivative maps ---------------------------------------


def pulsar_mass(n, x_star, mc, c):
    """Vela ``pulsar_mass``: ``(mp, s)`` from the mass function.

    ``mc`` is dimensionless solar masses (``m2 / M_SUN``); handing this the raw
    internal-time ``M2`` would double-count ``Tsun``.
    """
    mass_function = n * n * x_star**3 / M_SUN
    s = sini_from_cosi(c)
    mc_s = mc * s
    mp = mc_s * vm.sqrt(mc_s) / vm.sqrt(mass_function) - mc
    return mp, s


def g_gamma_gr(n, mp, mc):
    """Vela ``g_gamma_gr``: the GR Einstein coefficient, seconds."""
    n_tsun_13 = vm.cbrt(n * M_SUN)
    total_13 = vm.cbrt(mp + mc)
    return M_SUN / n_tsun_13 * mc * (mp + 2.0 * mc) / total_13**4


def kappa_gr(x_star, mc, s, E2):
    """Vela ``kappa_gr``: GR periastron advance per unit mean longitude."""
    return 3.0 * (M_SUN / x_star) * mc * s / (1.0 - E2)


def pbdot_gw(n, mp, mc, E2):
    """Vela ``pbdot_gw``: the quadrupole orbital decay."""
    E4 = E2 * E2
    one_minus = 1.0 - E2
    denominator = one_minus**3 * vm.sqrt(one_minus)
    fe = (1.0 + (73.0 / 24.0) * E2 + (37.0 / 96.0) * E4) / denominator
    n_tsun_13 = vm.cbrt(n * M_SUN)
    total_13 = vm.cbrt(mp + mc)
    return -192.0 * PI / 5.0 * n_tsun_13**5 * fe * mp * mc / total_13


# --- sky geometry -----------------------------------------------------------


def sky_motion(alpha, delta, mu_alpha, mu_delta, dt):
    """Vela ``sky_motion``: the unit direction and its *tangential* rate.

    ``ndot`` is the proper-motion vector with its radial part projected out
    and divided by the norm -- not the pre-normalisation ``ndot_p``, which is
    what the Shklovskii term and the ``(east, north)`` triad would otherwise
    silently get wrong.
    """
    sin_a, cos_a = vm.sincos(alpha)
    sin_d, cos_d = vm.sincos(delta)
    n_p = (cos_a * cos_d, sin_a * cos_d, sin_d)
    ndot_p = (
        -sin_a * mu_alpha - cos_a * sin_d * mu_delta,
        cos_a * mu_alpha - sin_a * sin_d * mu_delta,
        cos_d * mu_delta,
    )
    r = tuple(n_i + ndot_i * dt for n_i, ndot_i in zip(n_p, ndot_p))
    rnorm = vm.norm3(r)
    n = vm.scale3(1.0 / rnorm, r)
    radial = vm.dot3(ndot_p, n)
    ndot = tuple((v - radial * n_i) / rnorm for v, n_i in zip(ndot_p, n))
    return n, ndot


def _ddr_sky_basis(alpha, delta, mu_alpha, mu_delta, tgeo, posepoch):
    """Vela ``_ddr_sky_basis``: the frozen ``TGEO`` triad."""
    n0, ndot = sky_motion(alpha, delta, mu_alpha, mu_delta, tgeo - posepoch)
    rho = vm.sqrt(n0[0] * n0[0] + n0[1] * n0[1])
    east = (-n0[1] / rho, n0[0] / rho, 0.0 * n0[0])
    north = (
        n0[1] * east[2] - n0[2] * east[1],
        n0[2] * east[0] - n0[0] * east[2],
        n0[0] * east[1] - n0[1] * east[0],
    )
    return n0, ndot, east, north


def IJ_from_v(v_I, v_J, omega):
    """Vela ``IJ_from_v``."""
    sin_omega, cos_omega = vm.sincos(omega)
    return (-v_I * sin_omega + v_J * cos_omega, v_I * cos_omega + v_J * sin_omega)


def v_from_mu_parallax(mu_I, mu_J, px, d_I, d_J, dt_K):
    """Vela ``v_from_mu_parallax``."""
    return (mu_I * dt_K - px * d_I, mu_J * dt_K - px * d_J)


def _ddr_geometry(
    config,
    ssb_obs_pos,
    tcorr,
    alpha,
    delta,
    mu_alpha,
    mu_delta,
    px,
    tgeo,
    posepoch,
    kom,
):
    """Vela ``_ddr_geometry``: the Cartesian viewing projector ``(I, J)``.

    ``ssb_obs_pos`` is ICRS. An ecliptic model rotates it into the sky frame
    of ``KOM`` and the ``(east, north)`` triad first.
    """
    _, ndot, east, north = _ddr_sky_basis(
        alpha, delta, mu_alpha, mu_delta, tgeo, posepoch
    )
    observer = (
        equatorial_to_ecliptic(ssb_obs_pos, config.obliquity)
        if config.ecliptic_coordinates
        else ssb_obs_pos
    )
    mu_I = vm.dot3(east, ndot)
    mu_J = vm.dot3(north, ndot)
    d_I = vm.dot3(observer, east)
    d_J = vm.dot3(observer, north)
    v_I, v_J = v_from_mu_parallax(mu_I, mu_J, px, d_I, d_J, tcorr - tgeo)
    return IJ_from_v(v_I, v_J, kom)


def _galactic_direction(config, alpha, delta, mu_alpha, mu_delta, tgeo, posepoch):
    """Vela ``_galactic_direction``: Galactic ``(l, b)`` of the frozen direction."""
    n0, _, _, _ = _ddr_sky_basis(alpha, delta, mu_alpha, mu_delta, tgeo, posepoch)
    n_icrs = (
        ecliptic_to_equatorial(n0, config.obliquity)
        if config.ecliptic_coordinates
        else n0
    )
    n_gal = (
        vm.dot3(ICRS_TO_GAL[0], n_icrs),
        vm.dot3(ICRS_TO_GAL[1], n_icrs),
        vm.dot3(ICRS_TO_GAL[2], n_icrs),
    )
    l = vm.arctan2(n_gal[1], n_gal[0])
    b = vm.arctan2(n_gal[2], vm.sqrt(n_gal[0] * n_gal[0] + n_gal[1] * n_gal[1]))
    return l, b


def _a_z(z):
    """The vertical Galactic acceleration of Vela's ``a_z`` closure."""
    return -4.0 * PI * DDR_RHO0 * DDR_Z0 * z / vm.sqrt(z * z + DDR_Z0 * DDR_Z0)


def _galactic_acceleration_los(distance_to_pulsar, l, b):
    """Vela ``_galactic_acceleration_los``. Returns ``(a_los, ok)``.

    The planar term divides by ``sin^2 l + beta^2``, which a sampled ``PX``
    can drive to zero on the Galactic-centre line of sight.
    """
    sin_b, cos_b = vm.sincos(b)
    sin_l, cos_l = vm.sincos(l)
    beta = (distance_to_pulsar / DDR_R0) * cos_b - cos_l
    denominator = sin_l * sin_l + beta * beta
    ok = vm.value(denominator) > 0.0
    denominator = _safe(ok, denominator, 1.0)
    a_planar = (
        -cos_b
        * (DDR_THETA0_OVER_C * DDR_THETA0_OVER_C / DDR_R0)
        * (cos_l + beta / denominator)
    )

    z_psr = DDR_ZSUN + distance_to_pulsar * sin_b
    a_vertical = (_a_z(z_psr) - _a_z(DDR_ZSUN)) * sin_b
    return a_planar + a_vertical, ok


def _compose_p(
    config,
    n,
    mp,
    mc,
    E2,
    *,
    PB,
    PBDOT,
    XPBDOT,
    PX,
    mu_alpha,
    mu_delta,
    gal_longitude,
    gal_latitude,
):
    """Vela ``_compose_p``: ``(total, p_shk, p_gal, p_gw, ok)``.

    The FBX chart never calls this; it encodes the period derivative in the
    series.
    """
    p_gw = 0.0
    if config.pbdot_kinematic:
        p_gw = pbdot_gw(n, mp, mc, E2)
        total = p_gw + XPBDOT
    else:
        total = PBDOT

    p_shk = 0.0
    p_gal = 0.0
    ok = True
    if config.use_kine:
        mu2 = mu_alpha * mu_alpha + mu_delta * mu_delta
        p_shk = PB * mu2 / PX
        acceleration, ok = _galactic_acceleration_los(
            1.0 / PX, gal_longitude, gal_latitude
        )
        p_gal = PB * acceleration
        total = total + p_shk + p_gal
    return total, p_shk, p_gal, p_gw, ok


# --- projector, Shapiro and the delays -------------------------------------


def _roemer(x, c, s, I, J, X, Y):
    """Vela ``_roemer``."""
    a = x / s
    Z = vm.sqrt(1.0 + I * I + J * J)
    return (x * Y + a * c * I * Y + a * J * X) / Z


def _shapiro_B_S_squared_norm(c_e, X, Y, c, s, I, J, omega):
    """Vela ``_shapiro_B_S_squared_norm``."""
    rho = 1.0 - c_e
    sin_omega, cos_omega = vm.sincos(omega)
    Rx = (X * cos_omega - Y * c * sin_omega) / rho
    Ry = (X * sin_omega + Y * c * cos_omega) / rho
    Rz = Y * s / rho
    Z = vm.sqrt(1.0 + I * I + J * J)
    v_I = -I * sin_omega + J * cos_omega
    v_J = I * cos_omega + J * sin_omega
    dx = v_I / Z - Rx
    dy = v_J / Z - Ry
    dz = 1.0 / Z - Rz
    return 0.5 * rho * (dx * dx + dy * dy + dz * dz)


def romer_einstein_delay(state: DDRState):
    """Vela ``rømer_einstein_delay``."""
    return (
        _roemer(state.x, state.c, state.s, state.I, state.J, state.X, state.Y)
        + state.g_gamma * state.s_e
    )


def d_romer_einstein_delay_d_F(state: DDRState):
    """Vela ``d_rømer_einstein_delay_d_F``."""
    return (
        _roemer(state.x, state.c, state.s, state.I, state.J, state.dX, state.dY)
        + state.g_gamma * state.c_e
    )


def d2_romer_einstein_delay_d_F2(state: DDRState):
    """Vela ``d2_rømer_einstein_delay_d_F2``."""
    return (
        _roemer(state.x, state.c, state.s, state.I, state.J, state.d2X, state.d2Y)
        - state.g_gamma * state.s_e
    )


def shapiro_delay(state: DDRState):
    """Vela ``shapiro_delay``."""
    return -2.0 * state.m2 * vm.log(state.B_S)


# --- state construction -----------------------------------------------------


def ddr_state(frozen, corr: Correction, p, *, config: DDRConfig) -> DDRState:
    """Vela's ``DDRState`` constructor, in Vela's order.

    The order is load-bearing twice over: it stops an evolved ``x`` being used
    where the reference ``x_star`` is required, and it keeps every domain
    predicate upstream of the substitution it guards.
    """
    fb = frozen.binary

    # 1. the binary time argument, reduced (R4.5).
    tcorr = corrected_time(frozen, corr)
    dt = tcorr - p.TASC
    dt_red = fb.dt_red - corr.delay - p.dTASC
    ok = _all_valid(dt, dt_red)
    valid = ok
    dt = _safe(ok, dt, 0.0)
    dt_red = _safe(ok, dt_red, 0.0)

    # 2. the Laplace-Lagrange shape and the inclination cosine.
    h = p.EPS1
    k = p.EPS2
    c = p.COSI
    E2 = h * h + k * k
    ok = _all_valid(h, k, c, E2) & (abs(vm.value(c)) < 1.0) & (vm.value(E2) <= 0.99**2)
    valid = valid & ok
    h = _safe(ok, h, 0.0)
    k = _safe(ok, k, 0.0)
    c = _safe(ok, c, 0.5)
    E2 = h * h + k * k
    s = sini_from_cosi(c)

    # 3. the projected axis, its rate, and the companion mass.
    x_star = p.A1
    m2 = p.M2
    a1dot = p.A1DOT
    x = x_star + dt * a1dot
    need_mass = config.use_pk or config.pbdot_kinematic
    ok = _all_valid(x_star, m2, a1dot, x, s)
    if need_mass:
        ok = ok & (vm.value(x_star) > 0.0) & (vm.value(x) > 0.0) & (vm.value(m2) > 0.0)
    else:
        ok = (
            ok
            & (vm.value(x_star) >= 0.0)
            & (vm.value(x) >= 0.0)
            & (vm.value(m2) >= 0.0)
        )
    valid = valid & ok
    x_star = _safe(ok, x_star, 1.0)
    m2 = _safe(ok, m2, M_SUN)
    a1dot = _safe(ok, a1dot, 0.0)
    x = _safe(ok, x, 1.0)
    mc = m2 / M_SUN

    # 4. the orbital chart, and the reference mean motion it implies.
    if config.use_fbx:
        FB = p.FB
        ok = _all_valid(*FB) & (vm.value(FB[0]) > 0.0)
        valid = valid & ok
        FB = (_safe(ok, FB[0], 1.0 / fb.period_ref_s),) + tuple(
            _safe(ok, member, 0.0) for member in FB[1:]
        )
        n = TWO_PI * FB[0]
        PB = dPB = pbdot = None
    else:
        PB = p.PB
        dPB = p.dPB
        ok = _all_valid(PB, dPB) & (vm.value(PB) > 0.0)
        valid = valid & ok
        PB = _safe(ok, PB, fb.period_ref_s)
        dPB = _safe(ok, dPB, 0.0)
        n = TWO_PI / PB
        FB = None

    # 5. the inferred pulsar mass (mass-mode only).
    if need_mass:
        mp, s = pulsar_mass(n, x_star, mc, c)
        ok = _all_valid(mp, s) & (vm.value(mp) > 0.0)
        valid = valid & ok
        mp = _safe(ok, mp, 1.0)
        s = _safe(ok, s, sini_from_cosi(c))
    else:
        mp = 0.0

    # 6. the Einstein coefficient and the periastron advance.
    if config.use_pk:
        g_gamma = g_gamma_gr(n, mp, mc)
        kappa = kappa_gr(x_star, mc, s, E2)
    else:
        g_gamma = p.GGAMMA
        omdot = p.OMDOT
        ok = _all_valid(g_gamma, omdot)
        valid = valid & ok
        g_gamma = _safe(ok, g_gamma, 0.0)
        omdot = _safe(ok, omdot, 0.0)
        kappa = omdot / n
    ok = _all_valid(g_gamma, kappa)
    valid = valid & ok
    g_gamma = _safe(ok, g_gamma, 0.0)
    kappa = _safe(ok, kappa, 0.0)

    # 7a. the frozen TGEO triad, once, for both kine and geo.
    astrometry = None
    if config.use_kine or config.use_geo:
        astrometry = _astrometry(p, config)
        valid = valid & astrometry[-1]

    # 7. the period derivative (PB chart only; FBX encodes it in the series).
    if not config.use_fbx:
        pbdot_raw = p.XPBDOT if config.pbdot_kinematic else p.PBDOT
        ok = _valid_number(pbdot_raw)
        valid = valid & ok
        pbdot_raw = _safe(ok, pbdot_raw, 0.0)

        alpha = delta = mu_alpha = mu_delta = 0.0
        px = 1.0
        gal_l = gal_b = 0.0
        if config.use_kine:
            alpha, delta, mu_alpha, mu_delta, px, tgeo, posepoch, _ = astrometry
            gal_l, gal_b = _galactic_direction(
                config, alpha, delta, mu_alpha, mu_delta, tgeo, posepoch
            )
            ok = _all_valid(gal_l, gal_b)
            valid = valid & ok
            gal_l = _safe(ok, gal_l, 0.0)
            gal_b = _safe(ok, gal_b, 0.0)

        pbdot, _, _, _, ok = _compose_p(
            config,
            n,
            mp,
            mc,
            E2,
            PB=PB,
            PBDOT=pbdot_raw,
            XPBDOT=pbdot_raw,
            PX=px,
            mu_alpha=mu_alpha,
            mu_delta=mu_delta,
            gal_longitude=gal_l,
            gal_latitude=gal_b,
        )
        valid = valid & ok
        ok = _valid_number(pbdot)
        valid = valid & ok
        pbdot = _safe(ok, pbdot, 0.0)

    # 8. the mean longitude and the phase slope.
    if config.use_fbx:
        lam, lamdot = fbx_mean_longitude(fb, dt, dt_red, FB)
    else:
        lam, lamdot = mean_longitude(fb, dt, dt_red, PB, dPB, pbdot)
    ok = _all_valid(lam, lamdot) & (vm.value(lamdot) > 0.0)
    valid = valid & ok
    lam = _safe(ok, lam, 0.0)
    lamdot = _safe(ok, lamdot, TWO_PI / fb.period_ref_s)

    # 9. precession is secular, so it reads the *unwrapped* longitude (R4.5b).
    lam_secular = lam + TWO_PI * fb.n_orb

    # 10. the regular Kepler solve.
    F, _, c_e, s_e, converged = solve_F(lam, h, k)
    ok = converged & _all_valid(F, c_e, s_e)
    valid = valid & ok
    F = _safe(ok, F, 0.0)
    c_e = _safe(ok, c_e, 0.0)
    s_e = _safe(ok, s_e, 0.0)

    # 11-12. the static projections, rotated by the accumulated precession.
    X0, Y0, dX0, dY0, d2X0, d2Y0 = static_XY(F, h, k)
    q = q_nu_minus_M(c_e, s_e, E2)
    q_star, q_star_converged = q_at_tasc(h, k)
    delta_prec = precession_delta(lam_secular, q, q_star, kappa)
    X, Y = rotate_XY(X0, Y0, delta_prec)
    dX, dY = rotate_XY(dX0, dY0, delta_prec)
    d2X, d2Y = rotate_XY(d2X0, d2Y0, delta_prec)
    ok = q_star_converged & _all_valid(X, Y, dX, dY, d2X, d2Y, q, q_star, delta_prec)
    valid = valid & ok
    X = _safe(ok, X, 0.0)
    Y = _safe(ok, Y, 0.0)
    dX = _safe(ok, dX, 0.0)
    dY = _safe(ok, dY, 0.0)
    d2X = _safe(ok, d2X, 0.0)
    d2Y = _safe(ok, d2Y, 0.0)

    # 13. the Cartesian viewing projector.
    if config.use_geo:
        alpha, delta, mu_alpha, mu_delta, px, tgeo, posepoch, _ = astrometry
        kom = p.KOM
        ok = _valid_number(kom)
        valid = valid & ok
        kom = _safe(ok, kom, 0.0)
        I, J = _ddr_geometry(
            config,
            frozen.ssb_obs_pos,
            tcorr,
            alpha,
            delta,
            mu_alpha,
            mu_delta,
            px,
            tgeo,
            posepoch,
            kom,
        )
        ok = _all_valid(I, J)
        valid = valid & ok
        I = _safe(ok, I, 0.0)
        J = _safe(ok, J, 0.0)
        omega = kom
    else:
        I = 0.0
        J = 0.0
        omega = 0.0

    # 14. the Shapiro argument.
    B_S = _shapiro_B_S_squared_norm(c_e, X, Y, c, s, I, J, omega)
    ok = _valid_number(B_S) & (vm.value(B_S) > 0.0)
    valid = valid & ok
    B_S = _safe(ok, B_S, 1.0)

    return DDRState(
        x=x,
        n=n,
        c=c,
        s=s,
        c_e=c_e,
        s_e=s_e,
        g_gamma=g_gamma,
        X=X,
        Y=Y,
        dX=dX,
        dY=dY,
        d2X=d2X,
        d2Y=d2Y,
        I=I,
        J=J,
        B_S=B_S,
        m2=m2,
        valid=valid,
    )


def _astrometry(p, config: DDRConfig):
    """Vela ``_native_astrometry`` plus geometry-domain checks.

    Returns ``(alpha, delta, mu_alpha, mu_delta, px, tgeo, posepoch, ok)``.
    Substitutes keep ``rho`` and ``1/PX`` finite on an invalid row.
    """
    if config.ecliptic_coordinates:
        alpha, delta = p.ELONG, p.ELAT
        mu_alpha, mu_delta = p.PMELONG, p.PMELAT
    else:
        alpha, delta = p.RAJ, p.DECJ
        mu_alpha, mu_delta = p.PMRA, p.PMDEC
    px = p.PX
    tgeo = p.TGEO
    posepoch = p.POSEPOCH

    ok = _all_valid(alpha, delta, mu_alpha, mu_delta, px, tgeo, posepoch)
    ok = ok & (vm.value(px) > 0.0)
    # A pole-on direction makes the east/north triad singular; the equatorial
    # rho = sqrt(n0x^2 + n0y^2) is cos(delta) before proper motion.
    ok = ok & (abs(abs(vm.value(delta)) - 0.5 * PI) > 1e-12)
    return (
        _safe(ok, alpha, 0.0),
        _safe(ok, delta, 0.0),
        _safe(ok, mu_alpha, 0.0),
        _safe(ok, mu_delta, 0.0),
        _safe(ok, px, 1.0),
        _safe(ok, tgeo, 0.0),
        _safe(ok, posepoch, 0.0),
        ok,
    )


def binary_ddr(frozen, corr: Correction, p, *, config: DDRConfig) -> Correction:
    """Vela ``correct_toa``: the inverse timing formula, then Shapiro.

    The inversion uses ``n / (1 - c_e)``, not ``lamdot`` (which carries the
    period derivative). Invalid points are NaN in both channels.
    """
    state = ddr_state(frozen, corr, p, config=config)
    d = romer_einstein_delay(state)
    dp = d_romer_einstein_delay_d_F(state)
    dpp = d2_romer_einstein_delay_d_F2(state)
    nhat = state.n / (1.0 - state.c_e)
    d_inv = d * (
        1.0
        - nhat * dp
        + (nhat * dp) * (nhat * dp)
        + 0.5 * (nhat * nhat) * d * dpp
        - 0.5 * state.s_e / (1.0 - state.c_e) * (nhat * nhat) * d * dp
    )
    delay = d_inv + shapiro_delay(state)
    doppler = nhat * dp
    return corr.add_delay(
        vm.nan_where(state.valid, delay),
        vm.nan_where(state.valid, doppler),
    )
