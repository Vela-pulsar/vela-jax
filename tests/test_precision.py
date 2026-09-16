"""The build-time reductions: what would go wrong without them (SPEC §4, T13)."""

import numpy as np

from vela_jax.precision import (
    reduce_orbits,
    reference_phase,
    seconds_since_epoch,
    spin_coefficients,
)
from vela_jax.taylor import taylor_horner_integral


def test_seconds_since_epoch_keeps_longdouble():
    mjd = np.array([54999.5, 55000.0, 55000.5], dtype=np.longdouble)
    tau = seconds_since_epoch(mjd, 55000.0)
    assert tau.dtype == np.longdouble
    assert np.allclose(np.asarray(tau, dtype=float), [-43200.0, 0.0, 43200.0])


def test_reference_phase_keeps_only_the_fractional_part():
    """It must be of order F0*D, not F0*tau: that is the whole point.

    Feeding it the exact pulse numbers leaves nothing but the fractional
    phase, plus the (constant) TZR phase it subtracts.
    """
    f = (np.longdouble("100.0000000000000039"), np.longdouble("-1e-15"))
    tau = np.linspace(-3.2e8, 3.2e8, 501).astype(np.longdouble)
    pulse_number = np.round(taylor_horner_integral(tau, f))
    phi_ref = reference_phase(tau, pulse_number, tau[0], f)
    assert np.ptp(phi_ref) < 2.0


def test_spin_coefficients_reproduce_the_shifted_series():
    """phi(tau + xi) - phi(tau) == c1 xi + c2 xi^2/2 + ..., exactly."""
    f = (np.longdouble(100.0), np.longdouble(-4.2e-8), np.longdouble(1e-18))
    tau = np.array([-2.3e7, 0.0, 4.4e6], dtype=np.longdouble)
    coeffs = spin_coefficients(tau, f)
    xi = np.array([-256.0, 500.0, -33.0])
    ours = taylor_horner_integral(xi, coeffs)
    exact = taylor_horner_integral(
        tau + xi.astype(np.longdouble), f
    ) - taylor_horner_integral(tau, f)
    assert np.max(np.abs(ours - np.asarray(exact, dtype=float))) < 1e-9


def test_float64_spin_phase_would_have_lost_nanoseconds():
    """The reduction is not decoration: name the error it removes."""
    f0 = np.longdouble("61.485476554371304592")
    tau_ld = np.longdouble(3.2e8)
    naive = float(f0) * float(tau_ld)
    exact = float(f0 * tau_ld)
    assert abs(naive - exact) / float(f0) > 1e-9  # more than 1 ns of rounding


def test_orbit_reduction_removes_the_sawtooth():
    """T13. At 1e4 orbits the unreduced float64 phase carries a ps sawtooth.

    Both the reduced and the unreduced orbital phase are compared against the
    longdouble truth; only the fractional turn matters, since the integer part
    drops out of every trigonometric function downstream.
    """
    period = 86400.0 * 0.08
    tau = np.linspace(0.0, 1e4 * period, 2000).astype(np.longdouble)
    reduction = reduce_orbits(tau, 0.0, period)

    def wrap(x):
        return np.abs((np.asarray(x, dtype=float) + 0.5) % 1 - 0.5)

    exact = (tau / np.longdouble(period)) % 1
    naive = (np.asarray(tau, dtype=float) / period) % 1
    reduced = (reduction.dt_red / period) % 1

    reduced_error = np.max(wrap(reduced - exact))
    naive_error = np.max(wrap(naive - exact))
    # A couple of float64 ulps of a turn. Stated in ulps rather than as a
    # literal because the reduction's own floor is the longdouble subtraction
    # `dt - n*P`, and `numpy.longdouble` is 113-bit on aarch64 but 80-bit x87
    # on x86-64 -- a hard-coded budget tuned on one is a platform trap on the
    # other. What the gate is really for is the ratio below.
    assert reduced_error < 4 * np.finfo(float).eps
    assert naive_error > 100 * reduced_error


def test_worst_case_f0_and_span_lose_hundreds_of_ns_without_the_reduction():
    """Unreduced float64 ``F0*dt`` at F0~800 Hz over ~100 yr is a ~700 ns error.

    ``F0*D`` with ``|D| <= 1000 s`` is already float64-safe (~0.2 ps). The
    longdouble reduction that removes the first number has its own floor
    (~0.34 ns on 80-bit x87, negligible on 113-bit aarch64) and must stay
    below a nanosecond wherever ``require_longdouble`` lets the engine build.
    Every bound below is therefore in ulps of the type that sets it; a literal
    tuned on one platform is a trap on the other.

    A single round span such as ``100*365.25*86400`` multiplied by ``800.0``
    is an exactly representable float64 product and shows no error at all, so
    the witness is a set of non-integer spans and an F0 with fractional
    digits. Over 2000 draws the worst unreduced error is ~6.5e-7 s and the
    median ~1.8e-7 s.
    """
    f0 = np.longdouble("800.123456789012345678")
    rng = np.random.default_rng(0)
    years = np.longdouble(90.0) + rng.random(2000).astype(np.longdouble) * 10.0
    tau_ld = years * np.longdouble(365.25) * np.longdouble(86400.0)

    naive = np.float64(f0) * tau_ld.astype(np.float64)
    exact = f0 * tau_ld
    unreduced_s = np.abs(naive.astype(np.longdouble) - exact) / f0
    assert float(np.max(unreduced_s)) > 1e-7

    delay = 1000.0
    f0_d_err = abs(np.float64(f0) * delay - float(f0 * np.longdouble(delay))) / float(
        f0
    )
    assert f0_d_err < 1e-12

    longdouble_floor_s = float(np.finfo(np.longdouble).eps * np.max(tau_ld))
    assert longdouble_floor_s < 1e-9

    # The reduced path against the unreduced one, in turns. The bound is in
    # ulps of the reduced phase and not a literal, because *both* sides are
    # platform-dependent: `truth` differences two ~2.5e12-turn longdoubles,
    # which is 113-bit on aarch64 and 64-bit x87 on x86-64. Measured 0.65 ulp
    # on aarch64 and 24 ulps on x86-64 -- 64 leaves margin over the wider of
    # the two while still catching a lost reduction, which would be ~3e6 ulps.
    # 64 ulps here is 1.4e-11 turns, i.e. 14 ps at this F0.
    tau_one = np.array([tau_ld[0]], dtype=np.longdouble)
    coeffs = spin_coefficients(tau_one, (f0,))
    ours = taylor_horner_integral(np.array([-delay]), coeffs)
    truth = taylor_horner_integral(
        tau_one - np.longdouble(delay), (f0,)
    ) - taylor_horner_integral(tau_one, (f0,))
    phase_ulp = float(np.spacing(np.float64(f0) * delay))
    residual_turns = float(np.max(np.abs(ours - np.asarray(truth, dtype=float))))
    assert residual_turns < 64 * phase_ulp
