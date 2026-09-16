"""The delta-formulated engine (SPEC §12: T14, T15)."""

import jax.numpy as jnp
import numpy as np
import pytest

#: T15's "full fixture set": every binary family the engine supports, with the
#: DDS fixture ``J2302+4442`` included by name. That one was v1's known fp32
#: gap -- ``SHAPMAX ~ 9`` means ``sin i = 0.99989`` and the Shapiro log
#: argument sits at 1e-4 after a violent cancellation, which a working-dtype
#: reference channel cannot survive. R11.3 rule 1 is what moves it into the
#: passing set, so it is a parametrisation, not a skip.
CASES = [
    pytest.param(*case, marks=marks)
    for case, marks in [
        # The default four: one of each thing the dual can get wrong.
        (("sim_dd", "binary+astrometry"), ()),  # DD: the Kepler difference solve
        (("J2302+4442.sim", "binary+astrometry"), ()),  # DDS: R11.3-1's fixture
        (("sim_ell1k", "binary+astrometry"), ()),  # ELL1k: the validity domain
        # DDR: the *regular* Kepler difference solve and `Pert.cbrt`, neither
        # of which any other family exercises. Its fixture freezes astrometry,
        # so the live set is the binary one; geometry/kinematics are certified
        # on their own model in `tests/test_ddr.py`.
        (("sim_ddr", "binary+astrometry"), ()),
        (("NGC6440E", "astrometry"), ()),  # isolated: the astrometry axes alone
        # Breadth over the remaining families.
        (("sim_ddk", "binary+astrometry"), pytest.mark.slow),  # DDK
        (("J1802-2124.sim", "binary+astrometry"), pytest.mark.slow),  # ELL1
        (("J1227-6208.sim", "binary+astrometry"), pytest.mark.slow),  # ELL1H
        (("J0453+1559.sim", "binary+astrometry"), pytest.mark.slow),  # DDH
        (("sim_sw", "astrometry"), pytest.mark.slow),  # solar-wind clip of cos rho
    ]
]


@pytest.fixture(scope="module")
def perturbative(engine_factory):
    """One perturbative engine per (fixture, mode, dtype), for the whole file.

    The three gates below all want the same two engines, and building one
    costs a ``jacfwd`` of the parent plus its own compile. Rebuilding them per
    test was a third of this file's wall clock.
    """
    cache = {}

    def build(name, mode, dtype):
        key = (name, mode, jnp.dtype(dtype).name)
        if key not in cache:
            cache[key] = engine_factory(name).perturbative(mode, dtype=dtype)
        return cache[key]

    return build


#: How much of the delta suite a default run uses. The gates are about whether
#: the difference *algebra* is right, which every case answers; sweeping all
#: four sigma multiples and 8 joint draws is breadth, and breadth is what the
#: ``slow`` tier is for. ``certify`` still reports over whatever it is given.
DEFAULT_CERTIFY = dict(n_random=2, scales=(-3.0, 3.0))
FULL_CERTIFY = dict(n_random=8, scales=(-3.0, -1.0, 1.0, 3.0))


@pytest.fixture(scope="module")
def certify_kwargs(request):
    full = request.config.getoption("--certify") == "full"
    return FULL_CERTIFY if full else DEFAULT_CERTIFY


#: The residual amplitude this gate caps at. R11.6 measures the dropped
#: second-order term at ~4e-4 per second, i.e. a picosecond at 50 microseconds
#: -- exactly the T14 budget, so 50 would put the gate *on* the boundary and
#: measure the assembly rather than the algebra. 20 microseconds leaves the
#: cross term at ~1.5e-13 s and is still far beyond any step a posterior takes.
POSTERIOR_SCALE_S = 2e-5


@pytest.mark.parametrize("name,mode", CASES)
def test_float64_delta_formulation_is_exact(perturbative, certify_kwargs, name, mode):
    """T14. Same physics as the full engine, written as differences.

    Capped at the residual amplitude a posterior visits: above that, what
    limits the comparison is the *assembly's* dropped ``Delta doppler *
    Delta D`` term, not the difference algebra this test is about. The next
    test measures that term instead of hiding it.
    """
    engine = perturbative(name, mode, jnp.float64)
    report = engine.certify(
        rtol=1e-5, atol=1e-12, residual_cap=POSTERIOR_SCALE_S, **certify_kwargs
    )
    assert report.max_abs < 1e-12, report
    assert report.jacobian_rel < 1e-6, report


@pytest.mark.parametrize("name,mode", CASES)
def test_the_dropped_cross_term_stays_second_order(
    perturbative, certify_kwargs, name, mode
):
    """R11.6. The assembly drops ``Delta doppler * Delta D``; bound it.

    Uncapped, so the fixtures whose quoted uncertainties move the residual by
    milliseconds are exactly the ones that measure this. The coefficient must
    stay at the ~4e-4 per second the spec names -- if it grew, the residual
    would be too large for a first-order assembly and *that*, not the delta
    formulation, would be the finding.
    """
    engine = perturbative(name, mode, jnp.float64)
    report = engine.certify(rtol=1e-5, atol=1e-12, **certify_kwargs)
    assert report.max_abs < 1e-12 + 1e-3 * report.max_scale**2, report
    if report.max_abs > 1e-11:  # above the float64 floor, the ratio means something
        assert report.quadratic_coefficient < 1e-3, report


@pytest.mark.parametrize("name,mode", CASES)
def test_float32_stays_within_the_gate(perturbative, certify_kwargs, name, mode):
    """T15. float32 on the *delta*, validated against the float64 engine."""
    engine = perturbative(name, mode, jnp.float32)
    report = engine.certify(rtol=1e-5, atol=1e-12, **certify_kwargs)
    assert report.passed, report


def test_linear_axes_come_from_the_design_matrix(engine_factory):
    engine = engine_factory("sim_dd")
    perturbative = engine.perturbative("binary")
    index = engine.param_names.index("DM")
    delta = np.zeros(len(engine.param_names))
    delta[index] = 1e-6
    assert np.allclose(
        perturbative.residual_delta(delta), -engine.design_matrix()[:, index] * 1e-6
    )


def test_requesting_a_non_perturbative_axis_is_refused(engine_factory):
    from vela_jax.errors import UnsupportedModelError

    engine = engine_factory("sim_dd")
    with pytest.raises(UnsupportedModelError):
        engine.perturbative(["F0"])


@pytest.mark.unit
def test_dual_identities_are_exact():
    """Each rule is a difference identity, not a first-order expansion.

    The oracle is longdouble, because the naive float64 difference
    ``f(x+h) - f(x)`` is exactly the cancellation the identities exist to
    avoid: at ``h ~ 1e-7`` it is only good to ~1e-16 absolute, while the
    identities are good to ~1e-23.
    """
    from vela_jax.perturbative.dual import Pert

    ref = np.array([0.3, -1.7, 2.9], dtype=np.longdouble)
    step = np.array([1e-7, -3e-8, 5e-9], dtype=np.longdouble)
    x = Pert(jnp.asarray(ref, dtype=float), jnp.asarray(step, dtype=float))
    for op, exact in [
        (lambda v: v.sin(), np.sin),
        (lambda v: v.cos(), np.cos),
        (lambda v: v.exp(), np.exp),
        (lambda v: (v * v + 4.0).sqrt(), lambda t: np.sqrt(t * t + 4.0)),
        (lambda v: (v * v + 4.0).log(), lambda t: np.log(t * t + 4.0)),
        (lambda v: 1.0 / (v + 5.0), lambda t: 1.0 / (t + 5.0)),
        (lambda v: v.clip(-2.0, 2.0), lambda t: np.clip(t, -2.0, 2.0)),
    ]:
        got = np.asarray(op(x).delta, dtype=np.longdouble)
        want = exact(ref + step) - exact(ref)
        # The oracle is itself a difference of two longdoubles, so its own
        # relative accuracy is `eps_longdouble * |f| / |Delta f|`. With a
        # 113-bit longdouble (aarch64) that is ~1e-27 and the 1e-14 gate is
        # about the identities; with 80-bit x87 (x86-64) it is ~1e-12, and
        # gating below it would be measuring the oracle.
        floor = (
            float(np.finfo(np.longdouble).eps)
            * float(np.max(np.abs(exact(ref))))
            / float(np.max(np.abs(want)))
        )
        tolerance = max(1e-14, 20.0 * floor)
        assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < tolerance


@pytest.mark.unit
def test_dual_kepler_solves_the_difference_equation():
    from vela_jax.binary.orbit import mikkola
    from vela_jax.perturbative.dual import Pert

    l_ref = jnp.asarray([0.1, 1.9, -3.3, 12.4])
    dl = jnp.asarray([1e-8, -5e-9, 2e-9, 7e-10])
    e_ref, de = 0.42, 3e-9
    u = Pert(l_ref, dl).kepler(Pert(jnp.asarray(e_ref), jnp.asarray(de)), mikkola)
    # u' - e' sin u' = l' is the equation the perturbation must satisfy; check
    # it directly rather than differencing two float64 solutions.
    solved = np.asarray(mikkola(l_ref, e_ref), dtype=np.longdouble) + np.asarray(
        u.delta, dtype=np.longdouble
    )
    residual = (
        solved
        - (e_ref + de) * np.sin(solved)
        - np.asarray(l_ref + dl, dtype=np.longdouble)
    )
    assert np.max(np.abs(residual)) < 1e-15


@pytest.mark.unit
def test_dual_clip_zeros_a_perturbation_that_stays_outside():
    """The solar-wind path: unit-vector rounding puts |cos rho*| above 1.

    Keeping the unclipped perturbation after clipping the reference left
    arccos with c*=1 and a nonzero dc, which is Inf.
    """
    from vela_jax.perturbative.dual import Pert

    x = Pert(jnp.asarray([1.0000001]), jnp.asarray([1e-9]))
    clipped = x.clip(-1.0, 1.0)
    assert float(clipped.ref[0]) == 1.0
    assert abs(float(clipped.delta[0])) == 0.0
    angle = clipped.arccos()
    assert np.isfinite(float(angle.ref[0]))
    assert np.isfinite(float(angle.delta[0]))
    assert float(angle.ref[0]) == 0.0
    assert abs(float(angle.delta[0])) == 0.0

    crossing = Pert(jnp.asarray([0.99]), jnp.asarray([0.02])).clip(-1.0, 1.0)
    assert float(crossing.ref[0]) == pytest.approx(0.99)
    assert float(crossing.delta[0]) == pytest.approx(0.01)
