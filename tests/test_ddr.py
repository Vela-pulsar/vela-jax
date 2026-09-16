"""The Damour-Deruelle-Regular family (SPEC §12, T4/T7/T13 for DDR).

The pure-formula half is ported from Vela.jl's ``test/test_ddr.jl``: the same
helper identities and the same hard-coded delay anchors, so the two projects
gate on the same numbers rather than on each other's rounding. PINT is a
second, independent oracle and owns the one place the two deliberately differ
-- DDR geometry rotates with the par's resolved ``ECL``, where Vela hard-codes
IERS2010.
"""

from __future__ import annotations

import io
import math

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vela_jax import Engine
from vela_jax import numerics as vm
from vela_jax.binary import ddr as D
from vela_jax.binary.ddr import (
    DDRConfig,
    _solve_F_value,
    binary_ddr,
    ddr_state,
    reduce_longitude,
    solve_F,
)
from vela_jax.constants import AU_LS, M_SUN, OBL
from vela_jax.correction import Correction
from vela_jax.freeze import FrozenBinary
from vela_jax.params import Params
from vela_jax.perturbative.dual import Pert
from vela_jax.precision import reduce_orbits
from vela_jax.taylor import taylor_horner_integral

TWO_PI = 2.0 * np.pi
MAS = 1e-3 * np.pi / (180 * 3600)

#: Vela ``test_ddr.jl`` ``base_params``: the mass-valid anchor every delay
#: table below is evaluated at. ``PB`` is one day and ``TASC`` is the time
#: origin, so "seconds since TASC" is the TOA's own coordinate.
PB_S = 86400.0
ANCHOR_DT = np.array([0.0, 21600.0, 43200.0])


# --- scalar harness ---------------------------------------------------------
#
# Vela's DDR tests are scalar and build a TOA by hand. The stage here is a
# row-vector function of a frozen record, so the harness is the smallest frozen
# record that carries the same information: barycentred rows (a zero observer
# vector), and the orbit reduction the real freeze would have computed.


class _Frozen:
    def __init__(self, tau, period=PB_S, epoch_rel=0.0, obs_pos=None):
        self.tau = jnp.asarray(tau, dtype=float)
        reduction = reduce_orbits(
            np.asarray(tau, dtype=np.longdouble), epoch_rel, period
        )
        self.binary = FrozenBinary(
            jnp.asarray(reduction.dt_red),
            jnp.asarray(reduction.n_orb),
            reduction.period_ref_s,
        )
        zero = jnp.zeros_like(self.tau)
        self.ssb_obs_pos = (zero, zero, zero) if obs_pos is None else obs_pos
        self.n_rows = int(self.tau.shape[0])


def _params(**overrides):
    values = dict(
        TASC=0.0,
        dTASC=0.0,
        PB=PB_S,
        dPB=0.0,
        PBDOT=0.0,
        XPBDOT=0.0,
        A1=5.0,
        A1DOT=0.0,
        EPS1=0.02,
        EPS2=-0.03,
        M2=0.8 * M_SUN,
        COSI=0.5,
        GGAMMA=0.0,
        OMDOT=0.0,
        PX=0.0,
        TGEO=0.0,
        POSEPOCH=0.0,
        KOM=0.0,
        RAJ=0.0,
        DECJ=0.0,
        PMRA=0.0,
        PMDEC=0.0,
        ELONG=0.0,
        ELAT=0.0,
        PMELONG=0.0,
        PMELAT=0.0,
    )
    values.update(overrides)
    return Params(values)


def _config(**overrides):
    flags = dict(
        use_fbx=False,
        ecliptic_coordinates=False,
        use_pk=True,
        pbdot_kinematic=False,
        use_geo=False,
        use_kine=False,
        obliquity=OBL,
    )
    flags.update(overrides)
    return DDRConfig(**flags)


def _delay(frozen, params, config):
    corr = Correction.initial(frozen.n_rows)
    return np.asarray(
        binary_ddr(frozen, corr, params, config=config).delay, dtype=float
    )


def _doppler(frozen, params, config):
    corr = Correction.initial(frozen.n_rows)
    out = binary_ddr(frozen, corr, params, config=config)
    return np.asarray(out.doppler, dtype=float)


def _array(x):
    return np.asarray(x, dtype=float)


# ============================================================================
# 12.1 -- pure unit tests (Vela ``regular Kepler helpers``)
# ============================================================================


@pytest.mark.unit
def test_circular_solve_f_returns_the_reduced_longitude():
    """Vela: a circular orbit solves to its own reduced mean longitude."""
    lam = np.array([-7 * np.pi / 3, -np.pi / 4, 0.0, np.pi / 4, 7 * np.pi / 3])
    F, D, c_e, s_e, converged = solve_F(jnp.asarray(lam), 0.0, 0.0)
    lam_red, _ = reduce_longitude(lam)
    assert np.allclose(_array(F), lam_red, atol=5e-15, rtol=0)
    assert np.allclose(_array(D), 1.0, atol=1e-15, rtol=0)
    assert np.allclose(_array(c_e), 0.0, atol=1e-15, rtol=0)
    assert np.allclose(_array(s_e), 0.0, atol=1e-15, rtol=0)
    assert bool(np.all(np.asarray(converged)))


@pytest.mark.unit
def test_solve_f_inverts_the_regular_kepler_equation():
    """Vela: h=0.18, k=0.24, F=1.1 recovered from its own lambda."""
    h, k, F0 = 0.18, 0.24, 1.1
    lam = F0 - k * math.sin(F0) + h * math.cos(F0)
    F, D, c_e, s_e, converged = solve_F(jnp.asarray([lam]), h, k)
    assert float(_array(F)[0]) == pytest.approx(F0, abs=2e-15)
    residual = float(_array(F - k * vm.sin(F) + h * vm.cos(F))[0]) - lam
    assert abs(residual) == pytest.approx(0.0, abs=2e-15)
    assert float(_array(D)[0]) == pytest.approx(
        1.0 - k * math.cos(F0) - h * math.sin(F0), abs=1e-14
    )
    assert bool(converged[0])


@pytest.mark.unit
def test_the_pure_solver_meets_its_own_tolerance():
    """The gate is Vela's own ``4 eps max(1, |lambda_red|)``, not bit identity.

    A bit-for-bit assertion would be a statement about the host's libm, which
    varies between CPU architectures; the residual bound is a statement about
    the algorithm.
    """
    rng = np.random.default_rng(11)
    radius = 0.99 * np.sqrt(rng.random(4096))
    angle = rng.uniform(-np.pi, np.pi, radius.size)
    h, k = radius * np.sin(angle), radius * np.cos(angle)
    lam = rng.uniform(-20.0, 20.0, radius.size)

    F = np.asarray(_solve_F_value(jnp.asarray(lam), jnp.asarray(h), jnp.asarray(k)))
    lam_red, _ = reduce_longitude(lam)
    residual = F - k * np.sin(F) + h * np.cos(F) - lam_red
    bound = 4.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(lam_red))
    assert np.all(np.abs(residual) <= bound)


@pytest.mark.unit
def test_static_xy_lies_on_the_regular_orbit():
    """Vela: ``X^2 + Y^2 == (1 - c_e)^2``, the regular radius identity."""
    h, k = 0.18, 0.24
    F, _, c_e, s_e, _ = solve_F(jnp.asarray([1.1, -2.4, 0.3]), h, k)
    X, Y, *_ = D.static_XY(F, h, k)
    assert np.allclose(
        _array(X * X + Y * Y), _array((1.0 - c_e) ** 2), atol=2e-15, rtol=0
    )


@pytest.mark.unit
def test_q_at_tasc_is_q_of_the_lambda_zero_solution():
    """Vela: ``q_at_tasc`` is just ``q_nu_minus_M`` at ``lambda = 0``."""
    h, k = 0.18, 0.24
    _, _, c_e, s_e, _ = solve_F(jnp.asarray([0.0]), h, k)
    direct = D.q_nu_minus_M(c_e, s_e, h * h + k * k)
    q_star, converged = D.q_at_tasc(jnp.asarray(h), jnp.asarray(k))
    assert float(_array(direct)[0]) == pytest.approx(float(_array(q_star)), abs=2e-15)
    assert bool(np.all(np.asarray(converged)))


@pytest.mark.unit
@pytest.mark.parametrize("dt", [0.0, 21600.0, 86400.0 * 365.25])
@pytest.mark.parametrize("pbdot", [0.0, -1e-12, 1e-12])
def test_mean_longitude_matches_velas_unreduced_form(dt, pbdot):
    """Vela ``mean_longitude``, once the frozen orbit count is added back.

    Vela computes the absolute phase; this package divides out the integer
    orbit count at build time (R4.5) and keeps the ``n_orb*dPB`` term. With a
    frozen ``PB`` the two must agree exactly on the unwrapped longitude.
    """
    frozen = _Frozen(np.array([dt]))
    lam, lamdot = D.mean_longitude(
        frozen.binary, dt, np.asarray(frozen.binary.dt_red), PB_S, 0.0, pbdot
    )
    # Vela's assertion is on the longitude itself; the comparison has to be on
    # the *reduced* one, because Vela's unreduced value at a year is ~2300 rad
    # and a 1e-14 absolute gate there would sit below the float64 ulp.
    phase = np.longdouble(dt) / np.longdouble(PB_S)
    want = np.longdouble(TWO_PI) * (phase - np.longdouble(0.5) * pbdot * phase * phase)
    want = want - np.longdouble(TWO_PI) * np.longdouble(
        np.asarray(frozen.binary.n_orb)[0]
    )
    assert float(_array(lam)[0]) == pytest.approx(float(want), abs=1e-14)
    assert float(_array(lamdot)) == pytest.approx(
        TWO_PI * (1.0 / PB_S - pbdot * dt / (PB_S * PB_S)), abs=1e-20
    )


@pytest.mark.unit
@pytest.mark.parametrize("n_orb", [-(10**4), 10**4])
@pytest.mark.parametrize("d_pb", [0.0, 3e-9])
@pytest.mark.parametrize("d_tasc", [0.0, 1e-4])
def test_pb_reduced_longitude_holds_at_ten_thousand_orbits(n_orb, d_pb, d_tasc):
    """R4.5 on DDR's longitude: the floor stays far below a picosecond.

    The oracle is longdouble and *unreduced*: it forms the absolute phase and
    subtracts the integer orbits before rounding, which is exactly the
    operation float64 cannot do and the build-time reduction replaces. A live
    ``PB`` and a live ``TASC`` are both perturbed, because a frozen ``TASC``
    makes ``dTASC`` zero and would hide a missing term.

    ``(PB, dPB)`` are supplied as the *consistent* pair ``PB = P* + dPB``. The
    layout's own pair is consistent only to the rounding of that sum, and
    ``n_orb`` multiplies exactly that discrepancy -- a shared property of this
    reduction and DD/ELL1's ``mean_anomaly``, worth about 10 ps at 1e4 orbits,
    and a statement about float64 addition rather than about R4.5. This test is
    the R4.5 question: does building the phase from ``(dt_red, n_orb, dPB)``
    lose anything against evaluating the same parameters exactly?
    """
    pb_ref = PB_S
    tau = np.array([n_orb * pb_ref + 1234.5], dtype=np.longdouble)
    frozen = _Frozen(np.asarray(tau, dtype=float), period=pb_ref)

    pb_exact = np.longdouble(pb_ref) + np.longdouble(d_pb)
    dt = np.asarray(tau - np.longdouble(d_tasc), dtype=float)
    dt_red = np.asarray(frozen.binary.dt_red) - d_tasc
    lam, _ = D.mean_longitude(frozen.binary, dt, dt_red, float(pb_exact), d_pb, 0.0)

    exact = (tau - np.longdouble(d_tasc)) / pb_exact
    exact = exact - np.longdouble(np.asarray(frozen.binary.n_orb))
    want = np.longdouble(TWO_PI) * exact
    error = abs(float(np.longdouble(_array(lam)[0]) - want[0]))
    # 5 lt-s of projected axis turns a longitude error into a Roemer delay.
    assert error * 5.0 < 5e-14

    # And the reduction is what buys that: the naive absolute-phase form this
    # replaces loses three orders of magnitude more at the same row.
    naive = TWO_PI * (float(dt[0]) / float(pb_exact))
    naive = naive - TWO_PI * float(np.asarray(frozen.binary.n_orb)[0])
    assert abs(naive - float(want[0])) > 50.0 * max(error, 1e-18)


@pytest.mark.unit
@pytest.mark.parametrize("n_orb", [-(10**4), 10**4])
@pytest.mark.parametrize("d_fb0", [0.0, 4e-17])
def test_fbx_reduced_longitude_floor_is_one_named_rounding(n_orb, d_fb0):
    """The same gate on the FBX chart, which has one floor the PB chart lacks.

    ``fbx_mean_longitude`` retains the integer orbits through
    ``n_orb * (FB0*P* - 1)``, and ``P*`` is the float64 ``1/FB0*``: the product
    rounds to exactly 1.0, so the bracket evaluates to zero while its true
    value is ~5e-17. ``n_orb`` multiplies that, which is ~5e-13 turns -- about
    16 ps of equivalent Roemer delay -- at 1e4 orbits.

    This is a property of the reduction identity itself and is shared verbatim
    with ``binary.orbit.mean_anomaly``, which every FBX family already uses; it
    is not something DDR introduces. So rather than loosen a tolerance, this
    asserts that the error **is** that one term and nothing else, which is what
    would catch a genuine regression.

    Unlike the PB chart, the reduction wins only about a factor of two here at
    1e4 orbits -- the retained-integer term and the absolute phase it replaces
    run out of float64 at a similar scale -- so no "much better than naive"
    claim is made.
    """
    fb0_ref = 1.0 / PB_S
    period_ref = 1.0 / fb0_ref
    tau = np.array([n_orb * period_ref - 987.25], dtype=np.longdouble)
    frozen = _Frozen(np.asarray(tau, dtype=float), period=period_ref)

    fb0 = np.longdouble(fb0_ref) + np.longdouble(d_fb0)
    fb = (fb0, np.longdouble("2.5e-22"))
    dt = np.asarray(tau, dtype=float)
    lam, _ = D.fbx_mean_longitude(
        frozen.binary, dt, np.asarray(frozen.binary.dt_red), tuple(float(c) for c in fb)
    )

    exact = taylor_horner_integral(tau, fb)
    exact = exact - np.longdouble(np.asarray(frozen.binary.n_orb))
    want = np.longdouble(TWO_PI) * exact
    error = float(np.longdouble(_array(lam)[0]) - want[0])

    rounding = np.float64(float(fb0) * period_ref) - np.longdouble(fb0) * np.longdouble(
        period_ref
    )
    predicted = float(
        np.longdouble(TWO_PI)
        * np.longdouble(np.asarray(frozen.binary.n_orb)[0])
        * rounding
    )
    assert error == pytest.approx(predicted, rel=1e-6, abs=1e-16)
    # ~16 ps of Roemer delay at 1e4 orbits, three orders under the Vela budget.
    assert abs(error) * 5.0 < 1e-10


@pytest.mark.unit
def test_precession_reads_the_unwrapped_longitude():
    """R4.5b for DDR: periapsis advance is secular and must see every orbit.

    Dropping the restoration is not a rounding difference, which is why this
    asserts the *measured* shift rather than a loose bound: at 1e4 orbits the
    precession angle moves by ``kappa*2*pi*n_orb`` and the Roemer delay by
    most of a second.
    """
    n_orb = 10**4
    tau = np.array([n_orb * PB_S + 1234.5])
    frozen = _Frozen(tau)
    params = _params()
    state = ddr_state(frozen, Correction.initial(1), params, config=_config())

    kappa = float(
        _array(D.kappa_gr(5.0, 0.8, D.sini_from_cosi(0.5), 0.02**2 + 0.03**2))
    )
    assert kappa == pytest.approx(2.0501533373e-6, rel=1e-9)

    lam, _ = D.mean_longitude(
        frozen.binary, np.asarray(tau), np.asarray(frozen.binary.dt_red), PB_S, 0.0, 0.0
    )
    F, _, c_e, s_e, _ = solve_F(lam, params.EPS1, params.EPS2)
    q = D.q_nu_minus_M(c_e, s_e, 0.02**2 + 0.03**2)
    q_star, _ = D.q_at_tasc(params.EPS1, params.EPS2)

    wrapped = D.precession_delta(lam, q, q_star, kappa)
    secular = D.precession_delta(lam + TWO_PI * n_orb, q, q_star, kappa)
    shift = float(_array(secular)[0] - _array(wrapped)[0])
    assert shift == pytest.approx(kappa * TWO_PI * n_orb, rel=1e-12)
    assert shift == pytest.approx(0.129, abs=0.001)

    X0, Y0, *_ = D.static_XY(F, params.EPS1, params.EPS2)
    delays = []
    for angle in (wrapped, secular):
        X, Y = D.rotate_XY(X0, Y0, angle)
        delays.append(
            float(_array(D._roemer(state.x, state.c, state.s, 0.0, 0.0, X, Y))[0])
        )
    assert abs(delays[1] - delays[0]) == pytest.approx(0.64, abs=0.02)


@pytest.mark.unit
def test_the_solver_jvp_is_the_implicit_derivative():
    """The custom JVP, against ``dF = (dlam + sinF dk - cosF dh)/D``."""
    lam = jnp.asarray([0.7, -2.1, 4.9])
    h, k = jnp.asarray(0.18), jnp.asarray(0.24)

    for index, seed in enumerate(
        (
            (jnp.ones_like(lam), jnp.zeros(()), jnp.zeros(())),
            (jnp.zeros_like(lam), jnp.ones(()), jnp.zeros(())),
            (jnp.zeros_like(lam), jnp.zeros(()), jnp.ones(())),
        )
    ):
        F, dF = jax.jvp(_solve_F_value, (lam, h, k), seed)
        sin_F, cos_F = np.sin(_array(F)), np.cos(_array(F))
        denominator = 1.0 - float(k) * cos_F - float(h) * sin_F
        want = (
            np.asarray(seed[0], dtype=float)
            + sin_F * float(seed[2])
            - cos_F * float(seed[1])
        ) / denominator
        assert np.allclose(_array(dF), want, rtol=1e-12, atol=0), index


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_pert_regular_kepler_solves_the_difference_equation(dtype):
    """The perturbation must satisfy the equation, not merely be small.

    Checked by substitution, never by subtracting two absolute longitudes --
    which is the cancellation the difference formulation exists to avoid.
    """
    lam_ref = np.array([0.1, 1.9, -3.3, 12.4])
    d_lam = np.array([1e-8, -5e-9, 2e-9, 7e-10])
    h_ref, d_h = 0.18, 3e-9
    k_ref, d_k = 0.24, -2e-9

    lam = Pert(jnp.asarray(lam_ref), jnp.asarray(d_lam, dtype=dtype), dtype)
    h = Pert(jnp.asarray(h_ref), jnp.asarray(d_h, dtype=dtype), dtype)
    k = Pert(jnp.asarray(k_ref), jnp.asarray(d_k, dtype=dtype), dtype)
    F = lam.regular_kepler(h, k, _solve_F_value)

    solved = np.asarray(
        _solve_F_value(jnp.asarray(lam_ref), h_ref, k_ref), dtype=np.longdouble
    ) + np.asarray(F.delta, dtype=np.longdouble)
    lam_ld = np.asarray(lam_ref, dtype=np.longdouble)
    lam_red = lam_ld - np.round(lam_ld / np.longdouble(TWO_PI)) * np.longdouble(TWO_PI)
    residual = (
        solved
        - (k_ref + d_k) * np.sin(solved)
        + (h_ref + d_h) * np.cos(solved)
        - (lam_red + np.asarray(d_lam, dtype=np.longdouble))
    )
    assert np.max(np.abs(residual)) < (1e-14 if dtype is jnp.float64 else 1e-9)


@pytest.mark.unit
def test_pert_regular_kepler_survives_a_branch_crossing():
    """A perturbation may cross the principal-2pi cut; only periodicity matters.

    ``_solve_F_value`` reduces the *reference* longitude, so the difference
    solve can land one whole turn away from a fresh principal-branch solve.
    That is harmless precisely because every downstream use of ``F`` is
    2pi-periodic -- which is what this compares, rather than ``F`` itself.
    """
    lam_ref = np.array([np.pi - 1e-9])
    d_lam = np.array([2e-9])
    h, k = 0.18, 0.24

    F = Pert(jnp.asarray(lam_ref), jnp.asarray(d_lam)).regular_kepler(
        h, k, _solve_F_value
    )
    moved = np.asarray(F.ref + F.delta, dtype=float)
    fresh = np.asarray(_solve_F_value(jnp.asarray(lam_ref + d_lam), h, k), dtype=float)

    turns = (moved - fresh) / TWO_PI
    assert np.allclose(turns, np.round(turns), atol=1e-12)
    assert np.allclose(np.sin(moved), np.sin(fresh), atol=1e-12)
    assert np.allclose(np.cos(moved), np.cos(fresh), atol=1e-12)


@pytest.mark.unit
def test_pert_cbrt_is_a_difference_identity():
    """``cbrt(a+da) - cbrt(a)``, against a longdouble oracle."""
    ref = np.array([0.3, 1.7, 12.9], dtype=np.longdouble)
    step = np.array([1e-9, -3e-10, 5e-11], dtype=np.longdouble)
    got = np.asarray(
        Pert(jnp.asarray(ref, dtype=float), jnp.asarray(step, dtype=float))
        .cbrt()
        .delta,
        dtype=np.longdouble,
    )
    want = np.cbrt(ref + step) - np.cbrt(ref)
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-14


@pytest.mark.unit
def test_nan_where_poisons_both_pert_channels():
    """Lifting a scalar NaN would give it a zero delta; ``nan_where`` must not."""
    x = Pert(jnp.asarray([1.0, 2.0]), jnp.asarray([1e-6, 2e-6]))
    masked = vm.nan_where(jnp.asarray([True, False]), x)
    assert np.isfinite(float(masked.ref[0])) and np.isfinite(float(masked.delta[0]))
    assert np.isnan(float(masked.ref[1])) and np.isnan(float(masked.delta[1]))
    plain = vm.nan_where(jnp.asarray([True, False]), jnp.asarray([1.0, 2.0]))
    assert np.isnan(float(plain[1]))


# ============================================================================
# 12.2 -- direct Vela delay anchors
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,config_kwargs,extra,expected",
    [
        (
            "pk",
            {},
            {},
            (-0.19858979711316324, 4.8951043709516995, -0.2014760001122599),
        ),
        (
            "phenomenological",
            {"use_pk": False},
            {"GGAMMA": 0.0, "OMDOT": 0.0},
            (-0.1984501848750591, 4.8953235689803245, -0.2015923196527405),
        ),
        (
            "fbx",
            {"use_fbx": True},
            {"FB": (1.0 / 86400.0,)},
            (-0.19858979711316324, 4.8951043709516995, -0.2014760001122591),
        ),
    ],
)
def test_delay_matches_velas_anchors(label, config_kwargs, extra, expected):
    """Vela ``test_ddr.jl`` ``delay against PINT``, at 1 ps."""
    frozen = _Frozen(ANCHOR_DT)
    got = _delay(frozen, _params(**extra), _config(**config_kwargs))
    assert np.max(np.abs(got - np.array(expected))) < 1e-12, label
    assert np.all(
        np.isfinite(_doppler(frozen, _params(**extra), _config(**config_kwargs)))
    )


@pytest.mark.unit
def test_the_auxiliary_gr_quantities_match_vela():
    """``mp``, ``g_gamma`` and the derived ``kappa`` at the anchor."""
    frozen = _Frozen(ANCHOR_DT)
    state = ddr_state(frozen, Correction.initial(3), _params(), config=_config())
    n = float(_array(state.n))
    mp, s = D.pulsar_mass(n, 5.0, 0.8, 0.5)
    assert float(_array(mp)) == pytest.approx(0.7741081166, rel=1e-10)
    assert float(_array(state.g_gamma).ravel()[0]) == pytest.approx(
        0.0071937506, rel=1e-8
    )
    kappa = D.kappa_gr(5.0, 0.8, s, 0.02**2 + 0.03**2)
    assert float(_array(kappa)) == pytest.approx(2.0501533373e-6, rel=1e-9)


# ============================================================================
# 12.3 -- injected geometry
# ============================================================================


@pytest.mark.unit
def test_injected_projector_matches_velas_geometry_anchors():
    """Vela's hand-injected ``(I, J)`` state, at 1 ps.

    These anchors inject the projector rather than deriving it, so they pin
    ``_roemer``/``_shapiro_B_S_squared_norm`` under a non-zero ``(I, J)``;
    ``_ddr_geometry`` itself is gated against PINT below.
    """
    frozen = _Frozen(ANCHOR_DT)
    state = ddr_state(frozen, Correction.initial(3), _params(), config=_config())
    kom = np.pi / 6
    I, J = D.IJ_from_v(-MAS, 0.0, kom)
    B_S = D._shapiro_B_S_squared_norm(
        state.c_e, state.X, state.Y, state.c, state.s, I, J, kom
    )
    geo = state._replace(I=I, J=J, B_S=B_S)

    d = D.romer_einstein_delay(geo)
    dp = D.d_romer_einstein_delay_d_F(geo)
    dpp = D.d2_romer_einstein_delay_d_F2(geo)
    nhat = geo.n / (1.0 - geo.c_e)
    d_inv = d * (
        1.0
        - nhat * dp
        + (nhat * dp) * (nhat * dp)
        + 0.5 * (nhat * nhat) * d * dpp
        - 0.5 * geo.s_e / (1.0 - geo.c_e) * (nhat * nhat) * d * dp
    )
    got = _array(d_inv + D.shapiro_delay(geo))
    want = np.array([-0.19858982234067224, 4.895104376332743, -0.20147597688237923])
    assert np.max(np.abs(got - want)) < 1e-12


# ============================================================================
# 12.4 -- the kinematic period derivative
# ============================================================================


def _icrs_from_galactic(l_gal, b_gal):
    direction = np.array(
        [
            np.cos(b_gal) * np.cos(l_gal),
            np.cos(b_gal) * np.sin(l_gal),
            np.sin(b_gal),
        ]
    )
    icrs = np.array(D.ICRS_TO_GAL).T @ direction
    return (
        float(np.arctan2(icrs[1], icrs[0])),
        float(np.arctan2(icrs[2], np.hypot(icrs[0], icrs[1]))),
        icrs,
    )


@pytest.mark.unit
def test_kinematic_period_derivative_matches_vela():
    """Vela ``kinematic period derivative``, piecewise and composed.

    ``mean_longitude`` is what proves the composed value reaches phase with the
    right sign and factor 1/2; this pins the three pieces and their sum.
    """
    px = MAS / AU_LS
    pm = 10 * MAS / (365.25 * 86400.0)
    n = TWO_PI / PB_S
    mc = 0.8
    mp, _ = D.pulsar_mass(n, 5.0, mc, 0.5)
    e2 = 0.02**2 + 0.03**2

    p_gw = D.pbdot_gw(n, mp, mc, e2)
    p_shk = PB_S * pm * pm / px
    acceleration, ok = D._galactic_acceleration_los(1.0 / px, 1.0, 0.2)
    p_gal = PB_S * acceleration
    assert bool(np.all(np.asarray(ok)))

    assert float(_array(p_gw)) == pytest.approx(-1.170160493104432e-14, rel=1e-10)
    assert float(p_shk) == pytest.approx(2.0988692470710383e-14, rel=1e-10)
    assert float(_array(p_gal)) == pytest.approx(-4.722805781109057e-15, rel=1e-10)
    assert float(_array(p_gw + p_shk + p_gal)) == pytest.approx(
        4.564281758557005e-15, rel=1e-10
    )

    raj, decj, _ = _icrs_from_galactic(1.0, 0.2)
    config = _config(pbdot_kinematic=True, use_kine=True)
    gal_l, gal_b = D._galactic_direction(config, raj, decj, pm, 0.0, 0.0, 0.0)
    assert float(_array(gal_l)) == pytest.approx(1.0, abs=2e-15)
    assert float(_array(gal_b)) == pytest.approx(0.2, abs=2e-15)

    total, shk, gal, gw, ok = D._compose_p(
        config,
        n,
        mp,
        mc,
        e2,
        PB=PB_S,
        PBDOT=0.0,
        XPBDOT=0.0,
        PX=px,
        mu_alpha=pm,
        mu_delta=0.0,
        gal_longitude=gal_l,
        gal_latitude=gal_b,
    )
    assert bool(np.all(np.asarray(ok)))
    assert float(_array(gw)) == pytest.approx(float(_array(p_gw)), rel=1e-13)
    assert float(_array(shk)) == pytest.approx(float(p_shk), rel=1e-13)
    assert float(_array(gal)) == pytest.approx(float(_array(p_gal)), rel=1e-13)
    assert float(_array(total)) == pytest.approx(4.564281758557005e-15, rel=1e-10)


@pytest.mark.unit
def test_the_galactic_direction_is_frame_independent():
    """An ecliptic model reaches the same ``(l, b)`` as its equatorial twin."""
    from vela_jax.astrometry import equatorial_to_ecliptic

    raj, decj, icrs = _icrs_from_galactic(1.0, 0.2)
    ecliptic = equatorial_to_ecliptic(tuple(icrs), OBL)
    elong = float(np.arctan2(ecliptic[1], ecliptic[0]))
    elat = float(np.arctan2(ecliptic[2], np.hypot(ecliptic[0], ecliptic[1])))

    equatorial = D._galactic_direction(
        _config(use_kine=True), raj, decj, 0.0, 0.0, 0.0, 0.0
    )
    rotated = D._galactic_direction(
        _config(ecliptic_coordinates=True, use_kine=True),
        elong,
        elat,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert float(_array(rotated[0])) == pytest.approx(
        float(_array(equatorial[0])), abs=2e-15
    )
    assert float(_array(rotated[1])) == pytest.approx(
        float(_array(equatorial[1])), abs=2e-15
    )


# ============================================================================
# 12.5 -- the invalid sampled domain (scalar half)
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,params_kwargs,config_kwargs",
    [
        # Vela's own invalid case: M2 = 0.2 makes the inferred pulsar mass
        # negative at this (PB, A1, COSI).
        ("negative pulsar mass", {"M2": 0.2 * M_SUN}, {}),
        ("|COSI| >= 1", {"COSI": 1.0}, {}),
        ("|COSI| > 1", {"COSI": -1.5}, {}),
        ("e > 0.99", {"EPS1": 0.8, "EPS2": 0.7}, {}),
        ("non-positive lamdot", {"PB": -PB_S}, {}),
        ("non-positive A1 in mass mode", {"A1": 0.0}, {}),
        ("non-positive M2 in mass mode", {"M2": 0.0}, {}),
        ("NaN EPS1", {"EPS1": float("nan")}, {}),
    ],
)
def test_an_invalid_sampled_point_returns_nan(label, params_kwargs, config_kwargs):
    """Vela returns an invalid state; a traced chain returns NaN, never raises."""
    frozen = _Frozen(ANCHOR_DT)
    config = _config(**config_kwargs)
    params = _params(**params_kwargs)
    delay = _delay(frozen, params, config)
    doppler = _doppler(frozen, params, config)
    assert np.all(np.isnan(delay)), label
    assert np.all(np.isnan(doppler)), label


@pytest.mark.unit
def test_an_evolved_a1_crosses_the_domain_row_by_row():
    """``x = A1 + dt*A1DOT`` is per-row, and so is the mask.

    The domain is not a property of the parameter vector alone: with a
    negative ``A1DOT`` the first row is still inside it while later rows are
    not, which is exactly why ``valid`` is an R-row array rather than a scalar.
    """
    frozen = _Frozen(ANCHOR_DT)
    # x(0) = 5 lt-s, x(21600 s) < 0.
    delay = _delay(frozen, _params(A1DOT=-1.0), _config())
    assert np.isfinite(delay[0])
    assert np.all(np.isnan(delay[1:]))


@pytest.mark.unit
def test_the_shapiro_argument_stays_positive_through_conjunction():
    """Vela's conjunction case, and the guard that stands behind it.

    ``B_S = 0.5*rho*|n_a - R|^2`` is half a non-negative radius times a sum of
    three squares, so it cannot go negative for finite inputs and reaches zero
    only at an exact edge-on superior conjunction. The ``B_S > 0`` mask in
    ``ddr_state`` is therefore a backstop against a degenerate intermediate,
    not a domain a sampler wanders across -- so what is worth gating is the
    near-singular case: Vela's ``COSI = 1e-10`` conjunction sits at
    ``B_S ~ 5e-21``, twenty orders below one, and both the delay and its
    derivative have to survive it.
    """
    frozen = _Frozen(np.array([21600.0]))
    config = _config(use_pk=False)
    conjunction = _params(EPS1=0.0, EPS2=0.0, COSI=1e-10, GGAMMA=0.0, OMDOT=0.0)
    state = ddr_state(frozen, Correction.initial(1), conjunction, config=config)
    assert bool(np.all(np.asarray(state.valid)))
    assert float(_array(state.B_S)[0]) == pytest.approx(5e-21, rel=1e-9)
    assert np.all(np.isfinite(_delay(frozen, conjunction, config)))

    # Exact alignment: positive, ~1e-33, and still differentiable.
    def delay(cosi):
        params = _params(EPS1=0.0, EPS2=0.0, COSI=cosi, GGAMMA=0.0, OMDOT=0.0)
        return binary_ddr(frozen, Correction.initial(1), params, config=config).delay

    aligned = ddr_state(
        frozen,
        Correction.initial(1),
        _params(EPS1=0.0, EPS2=0.0, COSI=0.0, GGAMMA=0.0, OMDOT=0.0),
        config=config,
    )
    assert float(_array(aligned.B_S)[0]) > 0.0
    assert float(_array(aligned.B_S)[0]) < 1e-30
    assert np.all(np.isfinite(np.asarray(delay(0.0))))
    assert np.all(np.isfinite(np.asarray(jax.jacfwd(delay)(1e-10))))


@pytest.mark.unit
def test_a_valid_point_stays_differentiable_next_to_the_boundary():
    """``jacfwd`` at a valid reference must not see the substituted branch."""
    frozen = _Frozen(ANCHOR_DT)

    def delay(cosi):
        return binary_ddr(
            frozen, Correction.initial(3), _params(COSI=cosi), config=_config()
        ).delay

    # COSI ~ 0.84 is where this (PB, A1, M2) stops implying a positive pulsar
    # mass, so the valid points below are chosen inside that, not near |c| = 1.
    for cosi in (0.0, 0.5, 0.8):
        assert np.all(np.isfinite(np.asarray(delay(cosi)))), cosi
        assert np.all(np.isfinite(np.asarray(jax.jacfwd(delay)(cosi)))), cosi
    assert np.all(np.isnan(np.asarray(delay(0.9))))
    # Crossing the boundary must not poison the derivative on the valid side.
    assert np.all(np.isfinite(np.asarray(jax.jacfwd(delay)(0.8))))


# ============================================================================
# ingest, dispatch and the full engine (12.6)
# ============================================================================


@pytest.fixture(scope="module")
def ddr_engine(ddr_fixture):
    return Engine.from_files(*ddr_fixture)


def test_the_fixture_resolves_to_the_ddr_family(ddr_engine):
    assert ddr_engine.chain.family == "DDR"
    assert "binary.DDR" in ddr_engine.stages
    facts = ddr_engine.binary_chart_facts()
    assert facts.family == "DDR"
    assert facts.kepler_convention == "ddr"
    assert facts.shapiro == "m2_cosi"
    assert facts.epoch_shift_exact is False
    assert facts.supports_domain is False


def test_ddr_does_not_claim_the_kepler_laplace_chart(ddr_engine):
    """nltiming must not apply its DD polar-to-Laplace chart to DDR."""
    from vela_jax.backend import VelaJaxTimingEngine

    capability = VelaJaxTimingEngine(ddr_engine).binary_chart_capability(
        "kepler_laplace", ""
    )
    assert capability.kepler_convention == "ddr"
    assert capability.supports_domain is False
    assert capability.epoch_shift_exact is False


def test_the_residuals_are_small_and_jit_stable(ddr_engine):
    residuals = ddr_engine.residuals()
    assert np.all(np.isfinite(residuals))
    # A zero-noise fixture: what is left is the longdouble spin-phase floor
    # over a 27-year baseline, not model error.
    assert np.max(np.abs(residuals)) < 1e-8

    delta = np.zeros(len(ddr_engine.param_names))
    delta[ddr_engine.param_names.index("A1")] = 1e-9
    once = ddr_engine.residual_delta(delta)
    assert np.allclose(once, np.asarray(ddr_engine.residual_delta_jax(delta)), rtol=0)
    assert np.all(np.isfinite(once))
    assert np.max(np.abs(once)) > 0.0


def test_every_free_axis_has_a_finite_jacobian_column(ddr_engine):
    jacobian = ddr_engine.residual_jacobian()
    assert np.all(np.isfinite(jacobian))
    for index, name in enumerate(ddr_engine.param_names):
        assert np.any(jacobian[:, index] != 0.0), name


@pytest.mark.parametrize("name", ["A1", "EPS1", "EPS2", "M2", "COSI", "TASC", "PB"])
def test_central_differences_agree_with_the_jacobian(ddr_engine, name):
    """M1: the analytic column against a central difference of the engine."""
    index = ddr_engine.param_names.index(name)
    column = ddr_engine.residual_jacobian()[:, index]
    # Absolute steps in PINT units. A step scaled by the parameter's own value
    # would put TASC (an MJD) 460 s from its reference and measure curvature.
    step = {
        "A1": 1e-7,
        "EPS1": 1e-8,
        "EPS2": 1e-8,
        "M2": 1e-5,
        "COSI": 1e-5,
        "TASC": 1e-9,
        "PB": 1e-11,
    }[name]
    delta = np.zeros(len(ddr_engine.param_names))
    delta[index] = step
    numeric = (ddr_engine.residual_delta(delta) - ddr_engine.residual_delta(-delta)) / (
        2.0 * step
    )
    scale = max(np.max(np.abs(column)), 1e-300)
    assert np.max(np.abs(numeric - column)) / scale < 1e-6, name


def test_an_invalid_sampled_theta_nans_the_residuals_without_raising(ddr_engine):
    """D13: the engine returns NaN; converting that to -inf is the sampler's job."""
    theta = ddr_engine.reference_theta().copy()
    theta[ddr_engine.param_names.index("M2")] = 0.2
    residuals = ddr_engine.residuals(theta)
    assert np.all(np.isnan(residuals))

    delta = ddr_engine.layout.delta_from_theta(theta)
    assert np.all(np.isnan(np.asarray(ddr_engine.residual_delta_jax(delta))))
    # The reference point is still valid and still differentiable.
    assert np.all(np.isfinite(ddr_engine.residual_jacobian()))


def test_moving_toward_the_boundary_stays_finite_until_it_is_crossed(ddr_engine):
    index = ddr_engine.param_names.index("M2")
    theta = ddr_engine.reference_theta().copy()
    for m2 in (0.8, 0.7, 0.6, 0.55):
        theta[index] = m2
        assert np.all(np.isfinite(ddr_engine.residuals(theta))), m2
    theta[index] = 0.2
    assert np.all(np.isnan(ddr_engine.residuals(theta)))


@pytest.mark.unit
def test_an_invalid_tzr_row_nans_every_residual():
    """D3: ``valid`` is an R-row mask, and the TZR row is one of the R.

    With a nonzero ``A1DOT`` the TZR row's evolved ``x = A1 + dt_tzr*A1DOT``
    crosses zero at a different parameter value than any science TOA. Here the
    science rows are strictly inside the mass-mode domain and only the TZR row
    is outside, so an all-NaN residual is attributable to the TZR row alone.
    """
    from vela_jax.pipeline import form_residuals

    # Science rows just after TASC, TZR row far in the past.
    tau = np.array([0.0, 21600.0, 43200.0, -4.0 * PB_S])
    frozen = _Frozen(tau)
    a1dot = 5.0 / (2.0 * PB_S)  # x(-2 PB) = 0: only the TZR row is outside
    params = _params(A1DOT=a1dot)
    config = _config()

    state = ddr_state(frozen, Correction.initial(4), params, config=config)
    valid = np.asarray(state.valid)
    assert valid[:3].all(), "the science rows must still be inside the domain"
    assert not valid[3], "the TZR row must be the one that crossed"

    delay = _delay(frozen, params, config)
    assert np.all(np.isfinite(delay[:3]))
    assert np.isnan(delay[3])

    spin = jnp.full(4, 250.0)
    corr = Correction.initial(4)._replace(phase=jnp.asarray(delay), spin_frequency=spin)
    assert np.all(np.isnan(np.asarray(form_residuals(frozen, corr))))


# ============================================================================
# 12.3 -- geometry against PINT, and the resolved-ECL deviation
# ============================================================================
#
# These build their own par/tim pairs: the committed fixture is barycentric and
# geometry-off (§10), which is what makes its scalar delay gate clean, and a
# zero observer vector cannot exercise a projector whose whole content is
# ``PX * (observer . east/north)``.

_GEOMETRY_PAR = """\
PSR              SIMDDRGEO
EPHEM            DE440
CLOCK            TT(BIPM2021)
UNITS            TDB
{astrometry}
PX               1.2                       0
F0               250.0                     1  1.0e-12
F1               -1.0e-15                  1  1.0e-22
PEPOCH           55000.0
POSEPOCH         55000.0
PLANET_SHAPIRO   N
BINARY           DDR
PB               1.0                       0
A1               5.0                       1  1.0e-7
TASC             55000.0                   1  1.0e-9
EPS1             0.02                      1  1.0e-7
EPS2             -0.03                     1  1.0e-7
M2               0.8                       1  1.0e-4
COSI             0.5                       1  1.0e-4
KOM              30.0                      1  1.0e-2
DDRPK            {ddrpk}
DDRPBDOT         kinematic
DDRGEO           Y
DDRKINE          Y
{extra}
TZRMJD           55000.0
TZRFRQ           1400.0
TZRSITE          gbt
"""

_EQUATORIAL_ASTROMETRY = """\
RAJ              18:00:00.00000000         0
DECJ             -20:00:00.0000000         0
PMRA             3.0                       0
PMDEC            -5.0                      0"""

_ECLIPTIC_ASTROMETRY = """\
ELONG            270.0                     0
ELAT             3.5                       0
PMELONG          3.0                       0
PMELAT           -5.0                      0
ECL              {ecl}"""


def _geometry_engine(tmp_path, tag, astrometry, *, span, ddrpk="Y", extra=""):
    """A geometry+kinematics DDR engine on simulated topocentric TOAs."""
    from pint.models import get_model
    from pint.simulation import make_fake_toas_uniform

    text = _GEOMETRY_PAR.format(astrometry=astrometry, ddrpk=ddrpk, extra=extra)
    model = get_model(io.StringIO(text))
    toas = make_fake_toas_uniform(
        55000.0,
        55000.0 + span,
        24,
        model,
        obs="gbt",
        add_noise=False,
        freq=1400.0 * u.MHz,
    )
    toas.compute_pulse_numbers(model)
    par, tim = tmp_path / f"{tag}.par", tmp_path / f"{tag}.tim"
    par.write_text(text)
    toas.write_TOA_file(str(tim), format="tempo2", include_pn=True, include_info=False)
    return Engine.from_files(par, tim)


def _pre_binary(engine):
    """``(correction, params)`` accumulated up to -- and excluding -- the binary."""
    params = engine.layout.build(np.zeros(len(engine.param_names)))
    corr = Correction.initial(engine.frozen.n_rows)
    for name, stage in engine.chain.stages:
        if name.startswith("binary."):
            break
        corr = stage(engine.frozen, corr, params)
    return corr, params


def _our_binary_delay(engine, corr, params):
    stage = dict(engine.chain.stages)["binary.DDR"]
    out = stage(engine.frozen, corr, params)
    return np.asarray(out.delay - corr.delay, dtype=float)[:-1]


def _pint_binary_delay(engine, corr):
    acc = np.asarray(corr.delay, dtype=float)[:-1] * u.s
    component = engine.pint_model.components["BinaryDDR"]
    return np.asarray(
        component.binarymodel_delay(engine.pint_toas, acc).to_value(u.s), dtype=float
    )


@pytest.fixture(scope="module")
def geometry_engines(tmp_path_factory):
    """The three geometry-on engines these tests share; each costs a PINT sim."""
    tmp = tmp_path_factory.mktemp("ddr_geometry")
    return {
        "equatorial": _geometry_engine(tmp, "eq", _EQUATORIAL_ASTROMETRY, span=100.0),
        "ecl_2003": _geometry_engine(
            tmp,
            "ecl2003",
            _ECLIPTIC_ASTROMETRY.format(ecl="IERS2003"),
            span=1095.0,
        ),
        "ecl_1992": _geometry_engine(
            tmp,
            "ecl1992",
            _ECLIPTIC_ASTROMETRY.format(ecl="IERS1992"),
            span=1095.0,
        ),
    }


def test_the_real_tgeo_triad_matches_pint(geometry_engines):
    """The derived ``(I, J)`` projector against PINT's own DDR kernel, at 1 ps.

    Vela supplies no real-TGEO anchor -- its geometry test injects ``(I, J)``
    by hand -- so PINT is the oracle for the triad itself. The span is 100 days
    deliberately: see the ``M2`` constant test below for the linear-in-time
    term that would otherwise set the floor, and which has nothing to do with
    geometry.
    """
    engine = geometry_engines["equatorial"]
    corr, params = _pre_binary(engine)
    difference = _our_binary_delay(engine, corr, params) - _pint_binary_delay(
        engine, corr
    )
    assert np.max(np.abs(difference)) < 1e-12


def test_the_geometry_uses_the_resolved_obliquity_not_velas(geometry_engines):
    """D8: DDR rotates with the par's ``ECL``, and PINT is the authority.

    Vela's DDR hard-codes IERS2010. This package resolves ``ECL`` everywhere,
    which is a *physics* deviation, so it needs an oracle that also honours
    ``ECL`` -- PINT.

    The comparison is a **movement**: two otherwise identical ecliptic models
    with different obliquities, evaluated on one frozen pre-binary correction
    that is passed to both engines and, as the same numeric accumulated delay,
    to both PINT components. Re-running each model's solar-system stage would
    change ``corrected_time`` too and measure that instead.
    """
    a, b = geometry_engines["ecl_2003"], geometry_engines["ecl_1992"]
    assert a.obliquity != b.obliquity

    corr, params_a = _pre_binary(a)
    params_b = b.layout.build(np.zeros(len(b.param_names)))
    moved_ours = _our_binary_delay(b, corr, params_b) - _our_binary_delay(
        a, corr, params_a
    )
    moved_pint = _pint_binary_delay(b, corr) - _pint_binary_delay(a, corr)

    # The fixture has to actually move, or the agreement below is vacuous.
    assert np.max(np.abs(moved_ours)) > 1e-11
    assert np.max(np.abs(moved_ours - moved_pint)) < 1e-12


def test_solar_masses_come_from_pints_constant_not_velas():
    """A measured Vela/PINT constant mismatch DDR is the first family to feel.

    ``M2`` enters the layout in seconds through PINT's own
    ``tcb2tdb_scale_factor`` (``GMsun/c^3 = 4.9254909476412675e-06 s``), and
    DDR recovers dimensionless solar masses as ``m2 / M_SUN`` with Vela's
    hard-coded literal ``4.92549094830932e-06``. The two constants differ by
    ``-1.36e-10`` relative, so ``mc`` is not exactly ``0.8`` for ``M2 0.8``.

    DD/ELL1 never see it: they use ``m2`` in seconds and the constant cancels.
    DDR's GR maps do, and ``kappa`` carries it into a *secular* precession
    angle, so against PINT the delay drifts linearly -- about 10 ps over three
    years on the geometry fixture, and ~90 ps at the committed fixture's 1e4
    orbits. That is inside every Vela and PINT budget here, and this package
    reproduces Vela.jl by construction (:mod:`vela_jax.binary.ddr` is a
    translation), so the behaviour is deliberate and pinned rather than fixed.

    This test exists so the mismatch cannot change size unnoticed.
    """
    from pint.models import get_model

    from vela_jax.units import reference_internal

    text = _GEOMETRY_PAR.format(astrometry=_EQUATORIAL_ASTROMETRY, ddrpk="Y", extra="")
    model = get_model(io.StringIO(text))
    m2_internal = reference_internal(model["M2"], 55000.0)
    mc = m2_internal / M_SUN
    assert mc == pytest.approx(0.8, rel=1e-9)
    assert mc != 0.8
    assert mc / 0.8 - 1.0 == pytest.approx(-1.3563161704865934e-10, rel=1e-6)


# ============================================================================
# 12.7 -- the perturbative engine's own DDR obligations
# ============================================================================


def test_the_live_binary_mode_keeps_the_ddr_axes_nonlinear(ddr_engine):
    """``nonlinear_params="binary"`` must not drop DDR's own axes.

    ``COSI`` is the one DDR always has; ``GGAMMA`` and ``XPBDOT`` are read only
    in the modes that select them, so they are checked on the registry rather
    than on this fixture.
    """
    from vela_jax.perturbative import BINARY_AXES

    assert {"COSI", "GGAMMA", "XPBDOT"} <= set(BINARY_AXES)
    live = ddr_engine.perturbative("binary").live_nonlinear
    assert "COSI" in live
    for name in ("A1", "M2", "TASC", "EPS1", "EPS2", "PB"):
        assert name in live, name


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_the_reference_channel_convergence_survives_float32(ddr_engine, dtype):
    """R11.3-0: convergence is a property of the float64 reference root.

    The difference solve runs in the working dtype and could not meet a
    float64-epsilon residual bound; reading the flag off ``Pert.value`` instead
    of ``Pert.reference`` would mark every float32 row invalid and NaN the
    whole engine.
    """
    engine = ddr_engine.perturbative("binary+astrometry", dtype=dtype)
    delta = np.zeros(len(ddr_engine.param_names))
    delta[ddr_engine.param_names.index("COSI")] = 1e-6
    delta[ddr_engine.param_names.index("A1")] = 1e-8
    out = np.asarray(engine.residual_delta(delta), dtype=float)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0.0


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_a_perturbation_across_the_domain_boundary_is_nan_not_zero(ddr_engine, dtype):
    """D4: ``nan_where`` must poison both channels.

    The trap this guards is specific. ``select(valid, delay, nan)`` lifts the
    scalar NaN with a *zero* perturbation, so an invalid sampled point would
    come back from ``delay_delta`` as "no change at all" -- a finite, plausible,
    wrong answer, where NaN is the honest one.
    """
    engine = ddr_engine.perturbative("binary+astrometry", dtype=dtype)
    index = ddr_engine.param_names.index("M2")

    inside = np.zeros(len(ddr_engine.param_names))
    inside[index] = -0.05
    assert np.all(np.isfinite(np.asarray(engine.delay_delta(inside))))

    outside = np.zeros(len(ddr_engine.param_names))
    outside[index] = -0.6  # M2 0.8 -> 0.2: the inferred pulsar mass goes negative
    crossed = np.asarray(engine.delay_delta(outside), dtype=float)
    assert np.all(np.isnan(crossed))
    assert np.all(np.isnan(np.asarray(engine.residual_delta(outside), dtype=float)))
