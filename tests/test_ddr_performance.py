"""DDR solver convergence and cost (SPEC §12.8).

Two separate questions, and the second one is the one that nearly went
unasked:

* **Does the fixed 16-pass unroll always converge?** Vela breaks out of a
  64-step scalar loop; a traced array cannot break, so the count is a
  constant and has to be justified by measurement rather than by assertion.
* **Is DDR affordable next to the closed-form families?** The 16-vs-64
  comparison cannot answer that -- it only says 16 passes cost less than 64.
  The gate that matters is DDR against DD/ELL1's single Mikkola solve on the
  *same* number of TOAs.

Wall-clock numbers are informational (CI hardware varies); the ratios are the
gates, and ``docs/PARITY.md`` records what was measured and where.
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vela_jax.binary.dd import binary_dd
from vela_jax.binary.ddr import (
    DDRConfig,
    _solve_F_value,
    binary_ddr,
    reduce_longitude,
    solve_F,
)
from vela_jax.binary.orbit import mikkola
from vela_jax.constants import M_SUN, OBL
from vela_jax.correction import Correction
from vela_jax.freeze import FrozenBinary
from vela_jax.params import Params
from vela_jax.precision import reduce_orbits

TWO_PI = 2.0 * np.pi

#: The production loop count. Everything below either justifies it or prices it.
PRODUCTION_LOOPS = 16


# --- an instrumented numpy twin of the kernel ------------------------------


def first_converged_index(lam, h, k, *, loops=64):
    """0-based index of the first pass whose *incoming* ``F`` already converged."""
    return _numpy_solve(lam, h, k, loops=loops)[1]


def _numpy_solve(lam, h, k, *, loops=64):
    """``(F, first_converged_index)`` from a numpy transcription of the kernel.

    A transcription of :func:`vela_jax.binary.ddr._solve_F_value` that records
    what the traced kernel cannot return. It is not taken on trust:
    :func:`test_the_numpy_twin_tracks_the_traced_kernel` pins both its root and
    its convergence against the real kernel on the same points.

    An index of ``-1`` means the point never converged within ``loops``.
    """
    n_orb = np.round(lam / TWO_PI)
    lam_red = lam - n_orb * TWO_PI
    lo = lam_red - 1.0
    hi = lam_red + 1.0
    F = np.clip(lam_red + k * np.sin(lam_red) - h * np.cos(lam_red), lo, hi)
    atol = 4.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(lam_red))
    done = np.zeros_like(F, dtype=bool)
    index = np.full(F.shape, -1, dtype=np.int64)

    for i in range(loops):
        sin_F, cos_F = np.sin(F), np.cos(F)
        residual = F - k * sin_F + h * cos_F - lam_red
        converged = np.abs(residual) <= atol
        index = np.where(converged & ~done, i, index)
        D = 1.0 - k * cos_F - h * sin_F
        with np.errstate(invalid="ignore", divide="ignore"):
            trial = F - residual / D
        good = np.isfinite(trial) & (lo <= trial) & (trial <= hi)
        sin_lo, cos_lo = np.sin(lo), np.cos(lo)
        residual_lo = lo - k * sin_lo + h * cos_lo - lam_red
        go_hi = residual_lo * residual > 0.0
        lo_new = np.where(go_hi, F, lo)
        hi_new = np.where(go_hi, hi, F)
        next_F = np.where(good, trial, 0.5 * (lo_new + hi_new))
        next_lo = np.where(good, lo, lo_new)
        next_hi = np.where(good, hi, hi_new)
        done = done | converged
        F = np.where(done, F, next_F)
        lo = np.where(done, lo, next_lo)
        hi = np.where(done, hi, next_hi)
    return F, index


def _numpy_root(lam, h, k, *, loops=64):
    """The twin's own root, for comparing algorithms rather than libms."""
    return _numpy_solve(lam, h, k, loops=loops)[0]


def _converged(lam, h, k):
    """The production predicate, evaluated the way ``ddr_state`` evaluates it.

    Deliberately ``solve_F`` itself rather than a numpy re-check of its output.
    The tolerance is ``4 eps max(1, |lambda_red|)`` -- about four ulp -- and
    numpy's and XLA's ``sin``/``cos`` differ by an ulp or two, so re-evaluating
    a JAX-produced root with numpy's libm reports spurious non-convergence on a
    grid this size. The engine folds *this* flag into ``valid``, so this is
    also the thing worth gating.
    """
    return np.asarray(solve_F(jnp.asarray(lam), jnp.asarray(h), jnp.asarray(k))[4])


# --- the deterministic random probe (§12.8) --------------------------------


def _probe(n=2_000_000):
    """SPEC §12.8's fixed-seed disk probe, verbatim."""
    rng = np.random.default_rng(20260916)
    radius = 0.99 * np.sqrt(rng.random(n))
    angle = rng.uniform(-np.pi, np.pi, radius.size)
    return (
        rng.uniform(-np.pi, np.pi, radius.size),
        radius * np.sin(angle),
        radius * np.cos(angle),
    )


#: The 22 points of the two-million-point probe whose first-converged index is
#: 8 -- the 9th pass, and the worst the probe finds. Retained as hexadecimal
#: float64 so the grid cannot drift with a NumPy RNG change, and so the gate
#: survives even if the probe itself is ever reseeded.
WORST_CASES = (
    ("0x1.492616d908b3cp+0", "0x1.e78463265cd75p-1", "0x1.13757b0d78c16p-2"),
    ("-0x1.81c4a62b1fd19p+1", "-0x1.b7c7836294bfbp-4", "-0x1.f4100c3bf2874p-1"),
    ("0x1.0dafacc709f60p-4", "0x1.a07ddedf9ff86p-5", "0x1.f7b7be374b2fbp-1"),
    ("-0x1.3887783e21890p+1", "-0x1.49c3f794e3135p-1", "-0x1.802c77d713203p-1"),
    ("0x1.94284cb7ed7d0p+0", "0x1.f9c7c6cc1663ap-1", "0x1.265a727054cf9p-8"),
    ("0x1.a14dca163c444p-1", "0x1.6b4552072b579p-1", "0x1.5f0b38da8c3eep-1"),
    ("-0x1.98796f87b01fbp+0", "-0x1.f9fa4c95194b9p-1", "-0x1.a36a026efa981p-7"),
    ("0x1.8285d02601390p-3", "0x1.685fbcd23d228p-3", "0x1.f1f2c747527d3p-1"),
    ("0x1.8f76e30745740p+0", "0x1.f990f82c0c9a4p-1", "0x1.74ca9064d96a0p-6"),
    ("-0x1.66d475bab985ap+1", "-0x1.4305c162c1419p-2", "-0x1.deabd7cffdaa8p-1"),
    ("0x1.82bf4acc1ae08p+0", "0x1.f7fa746166ef5p-1", "0x1.25087c776a4e4p-4"),
    ("-0x1.26ab7e56bb81ep+0", "-0x1.c7753e2d0878cp-1", "0x1.aa83749bd32e8p-2"),
    ("-0x1.6206994f0dc6cp-1", "-0x1.4785d1290b4dfp-1", "0x1.7f6b9152e492fp-1"),
    ("-0x1.4abf97b342e8cp+0", "-0x1.e3e895e4ca876p-1", "0x1.21bb5eb4b76fbp-2"),
    ("0x1.767d2d62dec50p-1", "0x1.4cc7ae2987390p-1", "0x1.7c51c98d17b21p-1"),
    ("0x1.f0d97333be1e0p-3", "0x1.0229ac74d4813p-2", "0x1.e6fd28671a2a5p-1"),
    ("0x1.13a580e19a20cp-1", "0x1.ff1cc90f61588p-2", "0x1.b5a75784e74c2p-1"),
    ("0x1.5b40984106428p+1", "0x1.ad56e0df8681cp-2", "-0x1.caf5e672ddd42p-1"),
    ("-0x1.e76c9a6f4af5dp+0", "-0x1.dabe588c13514p-1", "-0x1.57f4ad7507861p-2"),
    ("0x1.b235507c7efa8p-2", "0x1.aceb8dc15a583p-2", "0x1.c7b87d87465b2p-1"),
    ("-0x1.448c409d4d345p+1", "-0x1.173f78520dfefp-1", "-0x1.a2a5824cc64bdp-1"),
    ("-0x1.79eebfead8792p+0", "-0x1.f5153ba8e9eb6p-1", "0x1.b7207195240ecp-4"),
)


def _worst_case_arrays():
    columns = np.array([[float.fromhex(value) for value in row] for row in WORST_CASES])
    return columns[:, 0], columns[:, 1], columns[:, 2]


@pytest.mark.unit
def test_the_numpy_twin_tracks_the_traced_kernel():
    """The instrumented copy has to be the same algorithm, or it measures itself."""
    lam, h, k = _probe(20_000)
    index = first_converged_index(lam, h, k, loops=PRODUCTION_LOOPS)
    assert np.all(index >= 0)
    assert np.all(_converged(lam, h, k))

    # Same algorithm, so the roots agree far inside the solver's own tolerance.
    ours = np.asarray(_solve_F_value(jnp.asarray(lam), jnp.asarray(h), jnp.asarray(k)))
    twin = _numpy_root(lam, h, k, loops=PRODUCTION_LOOPS)
    assert np.max(np.abs(ours - twin)) < 1e-13


@pytest.mark.unit
def test_the_retained_worst_cases_still_converge_on_the_ninth_pass():
    """0-based loop *index*, not a 1-based pass count -- an off-by-one here
    would silently weaken the justification for 16."""
    lam, h, k = _worst_case_arrays()
    index = first_converged_index(lam, h, k, loops=64)
    assert index.tolist() == [8] * len(WORST_CASES)
    assert np.all(index <= PRODUCTION_LOOPS - 1)

    assert np.all(_converged(lam, h, k))


@pytest.mark.slow
def test_the_deterministic_stress_grid_converges_in_sixteen():
    """The committed grid of SPEC §12.8, including both sides of the +-pi cut."""
    radii = np.concatenate(
        (
            np.array([0.0]),
            np.geomspace(2.0**-40, 1e-3, 16),
            np.linspace(0.01, 0.95, 32),
            0.99 - np.geomspace(1e-15, 0.03, 15),
            np.array([0.99]),
        )
    )
    angles = np.linspace(-np.pi, np.pi, 128, endpoint=False)
    base_lam = np.linspace(-np.pi, np.pi, 253)
    edge_lam = np.array(
        [
            np.nextafter(-np.pi, -np.inf),
            np.nextafter(-np.pi, 0.0),
            np.nextafter(np.pi, 0.0),
            np.nextafter(np.pi, np.inf),
        ]
    )
    lam = np.sort(np.concatenate((base_lam, edge_lam)))
    radius_grid, angle_grid, lam_grid = np.meshgrid(radii, angles, lam, indexing="ij")
    h = (radius_grid * np.sin(angle_grid)).ravel()
    k = (radius_grid * np.cos(angle_grid)).ravel()
    lam_flat = lam_grid.ravel()

    worst = -1
    for start in range(0, lam_flat.size, 500_000):
        stop = start + 500_000
        block = (lam_flat[start:stop], h[start:stop], k[start:stop])
        index = first_converged_index(*block, loops=PRODUCTION_LOOPS)
        assert np.all(index >= 0), "a grid point did not converge in 16 passes"
        worst = max(worst, int(index.max()))
        assert np.all(_converged(*block))
    assert worst <= PRODUCTION_LOOPS - 1


@pytest.mark.slow
def test_the_two_million_point_probe_converges_in_sixteen():
    """The probe the loop count was chosen from, re-run end to end."""
    lam, h, k = _probe()
    worst = -1
    for start in range(0, lam.size, 500_000):
        stop = start + 500_000
        index = first_converged_index(
            lam[start:stop], h[start:stop], k[start:stop], loops=64
        )
        assert np.all(index >= 0)
        worst = max(worst, int(index.max()))
    # Eight is the number ``docs/PARITY.md`` records; 16 keeps seven passes in
    # hand after it. If this ever rises, the loop count needs re-deriving.
    assert worst == 8


# --- cost ------------------------------------------------------------------


def _median_seconds(call, args, *, samples=25, repeats=None):
    """Cold call, five warm calls, then ``samples`` synchronised timings.

    ``repeats`` is chosen so one sample is at least 50 ms on this device;
    a shared count is passed to every kernel so the medians are comparable.
    """
    jax.block_until_ready(call(*args))
    for _ in range(5):
        jax.block_until_ready(call(*args))
    if repeats is None:
        repeats = 1
        while True:
            start = time.perf_counter()
            for _ in range(repeats):
                out = call(*args)
            jax.block_until_ready(out)
            if time.perf_counter() - start >= 0.05 or repeats >= 4096:
                break
            repeats *= 2
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(repeats):
            out = call(*args)
        jax.block_until_ready(out)
        timings.append((time.perf_counter() - start) / repeats)
    return statistics.median(timings), repeats


def _solve_F_loops(loops):
    """A test-local kernel identical to the production one but for the count."""

    def solve(lam, h, k):
        lam_red, _ = reduce_longitude(lam)
        lo = lam_red - 1.0
        hi = lam_red + 1.0
        F = jnp.clip(lam_red + k * jnp.sin(lam_red) - h * jnp.cos(lam_red), lo, hi)
        atol = 4.0 * jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, jnp.abs(lam_red))
        done = jnp.zeros_like(F, dtype=bool)
        for _ in range(loops):
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

    return solve


@pytest.mark.slow
def test_sixteen_passes_cost_what_sixteen_passes_should():
    """A loop-count sanity check, and explicitly *not* the affordability gate.

    16/64 is 25% of the transcendental work; the ceiling is 35% to leave room
    for fixed overhead. Both kernels must also land on the same root -- a
    faster kernel that stopped early would pass a timing gate and fail the
    physics.
    """
    lam, h, k = (jnp.asarray(a[:20_000]) for a in _probe())
    production = jax.jit(_solve_F_value)
    long_loop = jax.jit(_solve_F_loops(64))

    assert np.allclose(
        np.asarray(production(lam, h, k)),
        np.asarray(long_loop(lam, h, k)),
        rtol=0,
        atol=1e-13,
    )

    fast, repeats = _median_seconds(production, (lam, h, k))
    slow, _ = _median_seconds(long_loop, (lam, h, k), repeats=repeats)
    print(
        f"\n16-step {fast * 1e3:.3f} ms  64-step {slow * 1e3:.3f} ms  "
        f"ratio {fast / slow:.3f} (repeats={repeats})"
    )
    assert fast / slow <= 0.35


@pytest.mark.slow
def test_the_implicit_jvp_costs_one_primal_solve():
    """The derivative is a closed form, so it must not cost a second solve.

    This replaces the obvious "16-step JVP vs 64-step JVP" comparison, which
    cannot be made honestly. With the implicit rule the differentiated cost is
    *independent of the loop count* -- the rule evaluates the primal once and
    applies a formula -- so a 16-vs-64 gradient ratio would only be measuring
    whichever primal the rule happened to call. And a 64-step twin *without* a
    custom rule is not a usable reference either: forward-mode through 64
    nested ``where``/bracket levels did not finish compiling in six minutes on
    this device, where the primal compiles in two seconds. That is the measured
    justification for the custom JVP, and it is recorded in ``docs/PARITY.md``
    rather than paid for on every slow run.

    What is left is the gate that matters: the rule adds essentially nothing.
    """
    lam, h, k = (jnp.asarray(a[:20_000]) for a in _probe())
    production = jax.jit(_solve_F_value)
    ones = tuple(jnp.ones_like(x) for x in (lam, h, k))
    gradient = jax.jit(lambda *a: jax.jvp(_solve_F_value, a, ones)[1])

    primal, repeats = _median_seconds(production, (lam, h, k))
    derivative, _ = _median_seconds(gradient, (lam, h, k), repeats=repeats)
    print(
        f"\nprimal {primal * 1e3:.3f} ms  jvp {derivative * 1e3:.3f} ms  "
        f"ratio {derivative / primal:.3f} (repeats={repeats})"
    )
    assert derivative / primal <= 2.0


@pytest.mark.slow
def test_the_solver_costs_no_more_than_ten_mikkolas():
    """DDR's 16 Newton passes against DD/ELL1's one closed-form solve."""
    lam, h, k = (a[:20_000] for a in _probe())
    e = np.sqrt(h * h + k * k)
    production = jax.jit(_solve_F_value)
    closed_form = jax.jit(mikkola)

    ours, repeats = _median_seconds(
        production, (jnp.asarray(lam), jnp.asarray(h), jnp.asarray(k))
    )
    theirs, _ = _median_seconds(
        closed_form, (jnp.asarray(lam), jnp.asarray(e)), repeats=repeats
    )
    print(
        f"\nsolver: DDR {ours * 1e3:.3f} ms  mikkola {theirs * 1e3:.3f} ms  "
        f"ratio {ours / theirs:.2f} (repeats={repeats})"
    )
    assert ours / theirs <= 10.0


def _twenty_thousand_rows(period=86400.0):
    tau = np.linspace(0.0, 2.0e4 * period, 20_000)
    reduction = reduce_orbits(np.asarray(tau, dtype=np.longdouble), 0.0, period)

    class Frozen:
        pass

    frozen = Frozen()
    frozen.tau = jnp.asarray(tau)
    frozen.binary = FrozenBinary(
        jnp.asarray(reduction.dt_red),
        jnp.asarray(reduction.n_orb),
        reduction.period_ref_s,
    )
    zero = jnp.zeros_like(frozen.tau)
    frozen.ssb_obs_pos = (zero, zero, zero)
    frozen.n_rows = tau.size
    return frozen


@pytest.mark.slow
def test_the_ddr_stage_costs_no_more_than_fifteen_dd_stages():
    """The gate that actually prices DDR: whole stage against whole stage.

    "DDR is 20x the cost of DD" fails here even if the loop-count ratio passes,
    which is the failure mode that test cannot see.

    The traced input is the **orbital epoch**, and that is load-bearing. With
    only, say, ``A1`` traced, every input to the Kepler solve is a compile-time
    constant, XLA folds the whole solve away, and the "measurement" is the
    Roemer algebra alone -- which reported a 1.28x ratio on a stage whose
    solver takes longer than the entire measured time. A live epoch is also
    what a sampler actually moves.
    """
    frozen = _twenty_thousand_rows()
    corr = Correction.initial(frozen.n_rows)

    ddr_values = dict(
        TASC=0.0,
        dTASC=0.0,
        PB=86400.0,
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
    )
    config = DDRConfig(False, False, True, False, False, False, OBL)
    dd_values = dict(
        T0=0.0,
        dT0=0.0,
        PB=86400.0,
        dPB=0.0,
        PBDOT=0.0,
        A1=5.0,
        A1DOT=0.0,
        ECC=0.036,
        EDOT=0.0,
        OM=2.5535,
        OMDOT=0.0,
        GAMMA=0.0,
        DR=0.0,
        DTH=0.0,
        M2=0.8 * M_SUN,
        SINI=0.8660254037844386,
    )

    def ddr_stage(epoch):
        params = Params(dict(ddr_values, TASC=epoch, dTASC=epoch))
        return binary_ddr(frozen, corr, params, config=config).delay

    def dd_stage(epoch):
        params = Params(dict(dd_values, T0=epoch, dT0=epoch))
        return binary_dd(
            frozen, corr, params, family="DD", use_fbx=False, ecliptic=False
        ).delay

    ours_fn, theirs_fn = jax.jit(ddr_stage), jax.jit(dd_stage)
    epoch = jnp.asarray(0.0)
    assert np.all(np.isfinite(np.asarray(ours_fn(epoch))))
    assert np.all(np.isfinite(np.asarray(theirs_fn(epoch))))

    ours, repeats = _median_seconds(ours_fn, (epoch,))
    theirs, _ = _median_seconds(theirs_fn, (epoch,), repeats=repeats)
    print(
        f"\nstage: binary.DDR {ours * 1e3:.3f} ms  binary.DD {theirs * 1e3:.3f} ms  "
        f"ratio {ours / theirs:.2f} (repeats={repeats}, rows={frozen.n_rows})"
    )
    assert ours / theirs <= 15.0
