"""Engine-level gates (SPEC §12: T2, T3, T8, T9, T10)."""

import numpy as np
import pytest

from conftest import ENGINES, engine_params

NAMES = engine_params()
PINT_BUDGET = {name: budget for name, _, budget in ENGINES}


@pytest.mark.parametrize("name", NAMES)
def test_zero_delta_is_exactly_zero(engine_factory, name):
    """T2. The reference and the perturbed call are the same graph."""
    engine = engine_factory(name)
    delta = np.zeros(len(engine.param_names))
    assert np.array_equal(
        np.asarray(engine.residual_delta_jax(delta)), np.zeros(engine.toa_count)
    )


@pytest.mark.parametrize("name", NAMES)
def test_residuals_track_pint(engine_factory, name):
    """T3. PINT divides the phase residual by the constant F0, Vela by the
    doppler-shifted instantaneous spin frequency, so the two differ by a
    relative ~1e-4 of the residual plus a gauge constant. What must match is
    the *shape*, to well inside a nanosecond."""
    from pint.residuals import Residuals

    engine = engine_factory(name)
    ours = engine.residuals()
    theirs = Residuals(
        engine.pint_toas, engine.pint_model, subtract_mean=False
    ).time_resids.to_value("s")
    difference = ours - theirs
    assert np.max(np.abs(difference - difference.mean())) < PINT_BUDGET[name]


@pytest.mark.parametrize("name", NAMES)
def test_design_matrix_is_minus_the_jacobian(engine_factory, name):
    """M1/R9.3. The product matrix, against central differences, every column.

    ``design_matrix()`` *is* ``-residual_jacobian()`` by construction, so this
    gates the Jacobian on the posterior scale -- the matrix Enterprise and
    Discovery marginalize and nltiming's analytic route reads.
    """
    engine = engine_factory(name)
    design = engine.design_matrix()
    jacobian = engine.residual_jacobian()
    assert np.array_equal(design, -jacobian)
    assert np.all(np.isfinite(design))

    for column, param in enumerate(engine.param_names):
        scale = np.max(np.abs(design[:, column]))
        if scale == 0.0:
            continue
        assert _finite_difference_error(engine, design, column) < 1e-6, param


def _finite_difference_error(engine, design, column) -> float:
    """Best five-point difference-quotient disagreement with ``design``.

    Both ends of the step range are real: the residual's own float64 floor is
    ~1e-13 s, so a step that moves it by less than ~1e-8 s measures roundoff,
    while a step large enough to be roundoff-free walks a strongly nonlinear
    axis (``SINI`` near 1, ``KIN``) into curvature -- or, at 1e-4 s, straight
    out of the physical domain and into NaN. The gate is therefore that
    *some* well-conditioned step reproduces the column, which is the honest
    statement: a wrong column is wrong at every step.
    """
    scale = np.max(np.abs(design[:, column]))
    delta = np.zeros(len(engine.param_names))
    errors = []
    # Two steps, not three: 1e-7 s is the sweet spot on every fixture measured
    # and 1e-6 s is the escape hatch for an axis whose curvature bites there
    # (SINI near 1, KIN, SHAPMAX). A third only cost a third of this file.
    for target in (1e-6, 1e-7):
        step = target / scale

        def moved(scaling, step=step):
            delta[column] = scaling * step
            out = engine.residual_delta(delta)
            delta[column] = 0.0
            return out

        numeric = -(8 * (moved(1) - moved(-1)) - (moved(2) - moved(-2))) / (12 * step)
        error = np.max(np.abs(numeric - design[:, column])) / scale
        if np.isfinite(error):
            errors.append(float(error))
    assert errors, "every finite-difference step left the physical domain"
    return min(errors)


@pytest.mark.parametrize("name", NAMES)
def test_jacobian_is_finite_and_matches_finite_differences(engine_factory, name):
    """T8. Forward-mode against central differences of the same function.

    The step is chosen per column so that the residual actually moves by
    ~3e-8 s: much smaller and the comparison measures the float64 noise floor
    of the residual itself (~1e-13 s, since the accumulated phase is ~1e5
    turns), much larger and it measures the curvature of a nonlinear axis.
    """
    engine = engine_factory(name)
    jacobian = engine.residual_jacobian()
    assert np.all(np.isfinite(jacobian))

    for column, param in enumerate(engine.param_names):
        scale = np.max(np.abs(jacobian[:, column]))
        if scale == 0.0:
            continue
        step = 3e-8 / scale
        delta = np.zeros(len(engine.param_names))
        delta[column] = step
        numeric = (engine.residual_delta(delta) - engine.residual_delta(-delta)) / (
            2 * step
        )
        assert np.max(np.abs(numeric - jacobian[:, column])) / scale < 1e-4, param


@pytest.mark.parametrize("name", NAMES)
def test_design_matrix_matches_the_jacobian_on_linear_axes(engine_factory, name):
    """T9. PINT's analytic linearisation against forward-mode autodiff.

    The comparison is in *phase*, not seconds, because the two divide by
    different things: PINT by the constant ``F0``, Vela by the instantaneous
    doppler-shifted spin frequency. Once that is undone, the two differ by one
    constant per column -- PINT's design matrix ignores the TZR TOA's own
    dependence on the parameters, and the PHOFF gauge absorbs it.

    The budget is 1e-3 rather than machine precision because PINT's analytic
    derivative also omits the feedback of one component's delay into the
    corrected time later components see, which autodiff carries. That term is
    bounded by the total ``|dD/dt|`` -- the orbital plus solar-system Doppler
    factor -- which reaches ~3e-4 on a compact binary such as J0453+1559.

    PINT's matrix is the *oracle* here (R9.4), never the product: the product
    is ``-J``, gated column by column against finite differences by
    ``test_design_matrix_is_minus_the_jacobian``.
    """
    engine = engine_factory(name)
    linear = engine.identically_linear_params()
    if not linear:
        pytest.skip("no identically-linear free parameters")
    design = engine.design_matrix(source="pint")
    jacobian = engine.residual_jacobian()
    spin = engine.reference_spin_frequency()
    f0 = float(engine.pint_model.F0.quantity.to_value("Hz"))

    for name_ in linear:
        column = engine.param_names.index(name_)
        difference = jacobian[:, column] * spin + design[:, column] * f0
        scale = max(np.max(np.abs(design[:, column] * f0)), 1e-30)
        assert np.max(np.abs(difference - difference.mean())) / scale < 1e-3, name_


def test_theta_round_trip_goes_through_decimal(engine_factory):
    """R9.2: the delta is formed in Decimal against the exact strings."""
    from decimal import Decimal, localcontext

    engine = engine_factory("sim_dd")
    exact = engine.reference_theta_exact()
    delta = np.zeros(len(engine.param_names))
    delta[engine.param_names.index("A1")] = 1e-7

    with localcontext() as ctx:
        ctx.prec = 60
        theta = np.array(
            [
                float(Decimal(exact[name]) + Decimal(float(step)))
                for name, step in zip(engine.param_names, delta)
            ]
        )
    assert np.allclose(
        engine.residuals(theta) - engine.residuals(), engine.residual_delta(delta)
    )


def test_the_trace_never_calls_pint(engine_factory):
    """T10. PINT is the timing package; it must not appear inside the jitted graph."""
    import pint.models.timing_model as tm

    engine = engine_factory("NGC6440E")
    engine.residual_delta(np.zeros(len(engine.param_names)))  # warm the cache

    original = tm.TimingModel.__getattribute__

    def forbidden(self, item):
        raise AssertionError(f"the trace touched PINT: {item}")

    tm.TimingModel.__getattribute__ = forbidden
    try:
        delta = np.zeros(len(engine.param_names))
        delta[0] = 1e-9
        engine.residual_delta(delta)
    finally:
        tm.TimingModel.__getattribute__ = original


def test_precision_critical_and_exact_reference(engine_factory):
    engine = engine_factory("sim_dd")
    critical = engine.precision_critical_params()
    assert {"F0", "T0", "PB"} <= critical
    exact = engine.reference_theta_exact()
    # The epoch keeps more digits than float64 can hold.
    assert len(exact["T0"].split(".")[-1]) > 10


def test_binary_facts(engine_factory):
    engine = engine_factory("sim_dd")
    facts = engine.binary_chart_facts()
    assert facts.family == "DD"
    assert facts.kepler_convention == "dd"
    assert facts.epoch_shift_exact
    assert engine_factory("NGC6440E").binary_chart_facts() is None


def test_the_reference_channel_is_evaluated_once(engine_factory):
    """One chain run at theta*, however many consumers ask for it.

    The full correction, the barycentric cutoff and the topocentric spin
    frequency are three views of the same pass; building one pulsar record
    asks for all three. theta* does not move, so the answer cannot change
    between calls and re-running the chain would be pure waste.
    """
    engine = engine_factory("sim_dmx")

    engine.reference_correction()
    engine.reference_barycentric()
    engine.reference_spin_frequency()
    engine.pulsar_data()

    assert engine._reference_pass_count == 1
