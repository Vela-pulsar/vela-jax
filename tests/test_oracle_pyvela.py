"""Parity against Vela.jl itself, through pyvela (SPEC §12: T4, T7).

Skipped unless the ``oracle`` extra is installed (pyvela + a Julia runtime).
Both engines are built from the *same* PINT model object: SPNTA refits PHOFF
for its cheat priors, and that would otherwise show up as a gauge offset with
nothing to do with the physics under test.

Vela's rows are PINT's table rows, and so are this engine's -- nothing here
reorders TOAs -- so the two arrays are compared as they come.

One conversion IS needed. Vela divides the phase residual by the
doppler-shifted spin frequency; this engine divides by the pulsar-frame one,
to match PINT and tempo2 (SPEC §8, G2). So ``r_ours = r_vela * (1 + doppler)``,
and :func:`_as_ours` applies that before comparing. Without it these tests
measure ``r * (v/c)`` -- ~1e-4 of the residual -- and nothing about the delay
chain, which is what they exist to check.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from conftest import engine_params, resolve_pair
from vela_jax import Engine
from vela_jax.freeze import strip_noise_lines

pytestmark = pytest.mark.oracle


def _as_ours(engine, residuals):
    """Vela's residual in this engine's convention: ``r * (1 + doppler)``.

    The divisor is the only place the two deliberately differ (SPEC §8, G2);
    everything else -- delays, phase, TZR -- is meant to agree to the budgets
    below.
    """
    doppler = np.asarray(engine.reference_correction().doppler, dtype=float)[:-1]
    return np.asarray(residuals, dtype=float) * (1.0 + doppler)


def _spnta():
    """``pyvela.SPNTA``, imported on first use.

    Deliberately not a module-level ``importorskip``: importing pyvela boots a
    Julia runtime through juliacall, and at module scope that happens during
    *collection* -- 4 seconds added to every run of this package's tests,
    including runs that deselect this file entirely.
    """
    return pytest.importorskip("pyvela").SPNTA


def _require_vela_ddr():
    """DDR needs Vela.jl's own ``feat/ddr-model``; say which half is missing.

    An older Vela.jl imports and runs perfectly well for the other seven
    families, so the generic ``importorskip`` above would let this case fail
    deep inside pyvela's component dispatch instead of skipping.
    """
    _spnta()  # the ordinary pyvela/Julia skip first
    try:
        from juliacall import Main as jl

        jl.seval("using Vela")
        has_ddr = hasattr(jl.Vela, "BinaryDDR")
    except Exception as exc:  # pragma: no cover - environment probe
        pytest.skip(f"Vela.jl is not usable here: {exc}")
    if not has_ddr:
        pytest.skip(
            "the installed Vela.jl has no BinaryDDR; the DDR oracle needs "
            "Vela.jl feat/ddr-model (fcf7134)"
        )


CASES = engine_params(
    [
        "NGC6440E",
        "sim1",
        "sim2",
        "sim_dd",
        "sim_ddk",
        "sim_dmx",
        "sim_fd",
        "sim_jump",
        "sim_jump_ex",
        "sim_sw",
        "sim_ell1k",
        "pure_rotator",
        "J0453+1559.sim",
        "J0955-6150.sim",
        "J1227-6208.sim",
        "J1802-2124.sim",
        "J1856-3754.sim",
        "J2302+4442.sim",
        # The eighth family. Vela.jl ships no DDR par/tim, so this one is
        # this repository's own fixture, resolved per file (`resolve_pair`).
        "sim_ddr",
    ]
)


#: One ``SPNTA`` per fixture for the whole session. Building it boots a Julia
#: runtime and re-reads the par/tim, and the three gates below all want the
#: same one -- rebuilding per test was 60% of the entire suite's wall clock.
_PAIRS: dict[str, tuple] = {}


#: DDR needs one custom prior to reach Vela at all, and the reason is upstream.
#: pyvela's default-prior path has a ``TASC``/``T0`` branch that calls PINT's
#: ``PulsarBinary.pb()``, and ``pb()`` reads ``T0`` for every model whose name
#: does not start with ``ELL1``. ``BinaryDDR`` is ``TASC``-based and exposes no
#: ``T0``, so building an ``SPNTA`` raises ``AttributeError`` before any physics
#: runs. Supplying *any* explicit ``TASC`` prior takes the earlier
#: ``custom_prior_dists`` branch and steps around it. The bound below is
#: arbitrary: these gates compare residuals, never posteriors.
_DDR_PRIORS = {"TASC": {"distribution": "Uniform", "args": [53999.5, 54000.5]}}


def _pair(examples, name):
    if name not in _PAIRS:
        pair = resolve_pair(examples, name)
        if pair is None:
            pytest.skip(f"fixture {name} not available")
        par, tim = pair
        kwargs = {}
        if name == "sim_ddr":
            _require_vela_ddr()
            kwargs["custom_priors"] = _DDR_PRIORS
        stripped = pathlib.Path(tempfile.mkdtemp()) / par.name
        stripped.write_text(strip_noise_lines(par.read_text()))
        spnta = _spnta()(
            str(stripped), str(tim), center_epochs=False, check=False, **kwargs
        )
        _PAIRS[name] = (
            Engine.from_pint(spnta.model_pint_modified, spnta.toas_pint),
            spnta,
        )
    return _PAIRS[name]


@pytest.mark.parametrize("name", CASES)
def test_absolute_residuals_match_vela(examples, name):
    """T4: RMS <= 1 ns, max <= 10 ns."""
    engine, spnta = _pair(examples, name)
    theirs = _as_ours(engine, spnta.time_residuals(np.asarray(spnta.default_params)))
    difference = engine.residuals() - theirs
    assert np.std(difference) < 1e-9
    assert np.max(np.abs(difference)) < 1e-8


@pytest.mark.parametrize("name", CASES)
def test_residual_deltas_match_vela(examples, name):
    """T7: posterior-scale deltas on every free axis, RMS <= 1 ns."""
    engine, spnta = _pair(examples, name)
    theta0 = np.asarray(spnta.default_params, dtype=float)
    scale = np.asarray(spnta.scale_factors, dtype=float)
    vela_names = [str(n) for n in spnta.param_names]
    permutation = np.array([vela_names.index(n) for n in engine.param_names])
    reference = _as_ours(engine, spnta.time_residuals(theta0))

    sigma = np.array(
        [
            float(engine.pint_model[n].uncertainty_value or 0.0)
            for n in engine.param_names
        ]
    )
    rng = np.random.default_rng(0)
    for _ in range(4):
        delta = rng.normal(size=len(sigma)) * sigma
        theta = theta0.copy()
        theta[permutation] += delta * scale[permutation]
        theirs = _as_ours(engine, spnta.time_residuals(theta)) - reference
        assert np.max(np.abs(engine.residual_delta(delta) - theirs)) < 1e-8


@pytest.mark.parametrize("name", ["sim_dd", "J1802-2124.sim"])
def test_a_live_binary_epoch_matches_vela(examples, name):
    """T12. The epoch axes are the ones the build-time reduction touches: a live
    ``T0``/``TASC`` shifts the frozen orbit-count remainder, and a live ``PB``
    moves late-time orbital phase through the retained integer. Both have to
    reproduce Vela at deltas far below a par-file uncertainty."""
    engine, spnta = _pair(examples, name)
    theta0 = np.asarray(spnta.default_params, dtype=float)
    scale = np.asarray(spnta.scale_factors, dtype=float)
    vela_names = [str(n) for n in spnta.param_names]
    reference = _as_ours(engine, spnta.time_residuals(theta0))

    epoch = "TASC" if "TASC" in engine.param_names else "T0"
    for name_, step in ((epoch, 1e-6), (epoch, 1e-9), ("PB", 1e-9)):
        if name_ not in engine.param_names:
            continue
        delta = np.zeros(len(engine.param_names))
        delta[engine.param_names.index(name_)] = step
        theta = theta0.copy()
        theta[vela_names.index(name_)] += step * scale[vela_names.index(name_)]
        theirs = _as_ours(engine, spnta.time_residuals(theta)) - reference
        assert np.max(np.abs(engine.residual_delta(delta) - theirs)) < 1e-9, name_
