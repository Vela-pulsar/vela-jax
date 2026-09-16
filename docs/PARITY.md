# Measured parity

Every number here comes from [`examples/parity_report.py`](../examples/parity_report.py),
which regenerates the Vela.jl, perturbative, design-matrix, and H7 tables:

```bash
VELA_JAX_FIXTURES=/path/to/Vela.jl/pyvela/examples python examples/parity_report.py
```

The Vela.jl table needs the `oracle` extra (pyvela + a Julia runtime); the
PINT–tempo2 table needs the `tempo2` extra. The same comparisons run as ordinary
tests under `make oracle` (`tests/test_oracle_pyvela.py`), `make tempo2`
(`tests/test_read_tempo2.py`, `tests/test_tempo2_gates.py`) and `make fast`
(`tests/test_perturbative.py`, `tests/test_engine.py`).

Vela and PINT hand back arrays in the TOA table's order, and so does this
engine — nothing here reorders TOAs (SPEC R5.5) — so every comparison below is
row against row, with no permutation in between. That is the point of the rule:
a permuted comparison is the quietest way to manufacture a wrong number, and
the way not to have one is not to have a permutation.

## How the two timing packages enter these numbers

PINT or tempo2 reads the files; Vela's chain computes the delay. Those are
orthogonal to `binary_conventions`, which is the ELL1 Roemer truncation
only. The tables below are therefore different oracles, not four copies of
one comparison:

* **Against Vela.jl (T4, T7)** is a PINT read, by construction. pyvela
  builds from a PINT model object; the comparison holds the freeze fixed
  so only the physics differs. A tempo2 read compared to Vela.jl would mix
  a freeze difference with a physics difference.
* **The two timing packages (H7)** is the freeze comparison: the same
  observatory par/tim, both timing packages, same conventions, on
  mixed-engine-consistent files (clock coverage, shared ecliptic frame,
  `PLANET_SHAPIRO N`). Setting `ECL IERS2003` makes PINT use tempo2's
  default obliquity (tempo2 has no `ECL` keyword). It bounds freeze
  disagreement, not the delay chain. The load-bearing geometry check on
  a tempo2 read is the Roemer closure (A.2.3). Vela `sim_jump` and
  `sim_sw` are replaced by clock-covered / MetaPulsar-aligned files under
  `tests/data/`.
* **ELL1 conventions (S4)** is `binary_conventions`, against libstempo,
  on a discriminating eccentricity. Timing package is tempo2 on both
  sides of the flag.
* **Real data (J1853)** uses tempo2 for EPTA and PINT for NANOGrav, which
  is the intended pairing: each PTA's native timing package. The EPTA
  NUTS run is a tempo2 *read* under PINT *conventions* (the ELL1
  eccentricity is too small for S4 to bite).

Real PTA data is read by that PTA's native timing package. EPTA / IPTA /
InPTA → tempo2. NANOGrav and PINT-written simulations → PINT. The API
default `"pint"` matches pyvela and the Vela.jl fixtures; it is not a
claim about EPTA-family files.

## Against Vela.jl (SPEC §12, T4 and T7)

Both engines are built from **the same PINT model object** — pyvela's
`SPNTA.model_pint_modified`, after its own `fix_params` and its cheat-prior
refit. That matters: `SPNTA` refits `PHOFF`, and comparing against a separately
parsed model would show that as a gauge offset having nothing to do with the
physics under test.

* `‖r − r_Vela‖∞` is the worst absolute residual difference over all TOAs.
* `‖Δr − Δr_Vela‖∞` is the worst over four random ±1σ draws (par-file
  uncertainties) across *every* free axis, including the epochs.

Budget: RMS ≤ 1 ns, max ≤ 10 ns.

**These numbers are platform-dependent, and the table is aarch64's.** The one
build-time computation that sets the floor is the reference-phase reduction
`φ(τ) − N − φ(τ_tzr)` over ~10⁹ turns, in `numpy.longdouble` — which is
113-bit on aarch64 and 80-bit x87 on x86-64. Both platforms were measured
(the x86-64 run is the same script on the RTX 4070 Ti SUPER of the end-to-end
section below), and `‖r − r_Vela‖∞` moves by one to two orders:

| fixture | aarch64 | x86-64 |
|---|---:|---:|
| `NGC6440E` | 0.15 ps | 1.6 ps |
| `sim1` | 0.41 ps | 12 ps |
| `sim_dd` | 124 ps | 127 ps |
| `J1208-5936.sim` | 109 ps | 592 ps |

The delta column moves with it where the fixture is already near the floor:
`J1208-5936.sim` goes from 105 ps to 580 ps. It is entirely arithmetic width,
not physics — every fixture stays two orders inside the 10 ns budget on both
platforms, and the PINT/tempo2 two-package table below is identical to within
0.03 ns across the two. Two gates whose budgets had been tuned on aarch64 are
stated in ulps of the relevant type rather than as literals, for the same
reason (`test_precision.py`, `test_perturbative.py`).

| fixture | exercises | TOAs | RMS | ‖r − r_Vela‖∞ | ‖Δr − Δr_Vela‖∞ |
|---|---|---:|---:|---:|---:|
| `NGC6440E` | isolated, equatorial | 62 | 33333 ns | 0.15 ps | 0.22 ps |
| `sim1` | isolated + parallax | 2000 | 2595 ns | 0.41 ps | 0.44 ps |
| `sim2` | isolated, no DM | 2000 | 1895 ns | 0.17 ps | 0.35 ps |
| `pure_rotator` | no astrometry at all | 100 | 959 ns | 66 ps | 60 ps |
| `J1856-3754.sim` | isolated, ecliptic | 500 | 987 ns | 0.21 ps | 0.25 ps |
| `sim_dmx` | DMX | 1000 | 981 ns | 0.16 ps | 0.41 ps |
| `sim_fd` | FD | 2000 | 1336 ns | 0.19 ps | 0.44 ps |
| `sim_jump` | non-exclusive JUMPs | 999 | 1151 ns | 0.22 ps | 0.44 ps |
| `sim_jump_ex` | exclusive JUMPs | 2000 | 1291 ns | 0.3 ps | 0.43 ps |
| `sim_sw` | solar wind + planet Shapiro, ecliptic | 2000 | 1282 ns | 0.21 ps | 0.33 ps |
| `sim_dd` | DD | 1000 | 1000 ns | 124 ps | 45 ps |
| `J0955-6150.sim` | DD with OMDOT | 500 | 1006 ns | 2.4 ps | 3.5 ps |
| `sim_ddk` | DDK (Kopeikin) | 2000 | 1034 ns | 21 ps | 98 ps |
| `J0453+1559.sim` | DDH, infinite-frequency TZR | 2000 | 1308 ns | 16 ps | 23 ps |
| `J1208-5936.sim` | DDH | 500 | 972 ns | 109 ps | 92 ps |
| `J2302+4442.sim` | DDS, ecliptic | 500 | 1013 ns | 1.7 ps | 2.7 ps |
| `J1802-2124.sim` | ELL1 | 500 | 927 ns | 27 ps | 29 ps |
| `J1227-6208.sim` | ELL1H | 500 | 1030 ns | 12 ps | 12 ps |
| `sim_ell1k` | ELL1k | 500 | 1007 ns | 2.1 ps | 2.4 ps |

Median 1.7 ps, worst 124 ps — two orders inside the budget. Every remaining
fixture in `pyvela/examples` is refused by name (BT, `WaveX`, `DMWaveX`,
`CMWaveX`, `SolarWindDispersionX`, `FDJumpDM`, `DispersionJump`, glitches,
chromatic, wideband); `UnsupportedModelError` says which component and lists the
supported set.

### Where the picoseconds come from

The spread across fixtures is not random, and it is worth knowing which side it
is on.

* **The 0.2 ps floor** is `phi_ref`'s own: the accumulated phase is ~10⁵ turns,
  whose float64 ulp is ~1.5×10⁻¹³ s of residual. Nothing can do better in
  float64 without a second double-double, and Vela's `Double64` sits below it.
* **`sim_dd` (124 ps) and `J1208-5936` (109 ps)** are the fastest spin-down
  cases (`sim_dd` has `F1 = −4.2×10⁻⁸ Hz/s`). Here the remainder is *Vela's*
  float64, not this engine's: against PINT as a third opinion, vela-jax differs
  from PINT by 0.25 ns and Vela.jl by 0.33 ns on `sim_dd`.
* **`pure_rotator` (66 ps)** has no astrometry at all, so every row is
  barycentred and the residual is pure spin phase — again the phase floor,
  scaled by that fixture's `F1`.
* **The DD/DDK/ELL1 fixtures (12–100 ps)** carry the binary time argument
  through `corrected_time`, whose float64 representation is ~1.5×10⁻⁸ s; times
  the orbital velocity `2π·a1/PB` that is ~10⁻¹² s, and times the inverse
  timing formula's higher derivatives a little more.

None of these is a physics difference. A physics difference is millisecond-scale:
the unwrapped true anomaly `OMDOT` needs (2.7 ms on `J0955-6150`) or a truncated
reference spin series (0.87 ns on `sim_dd`, small only because that fixture has
no `F2`).

## The design matrix (SPEC §12, M1 / R9.3)

`design_matrix()` is `−residual_jacobian()` **by construction**, so M1 is a
gate on the Jacobian itself: every column against a five-point difference
quotient of `residual_delta`, at the posterior scale, on every fixture.

Both ends of the step range are real. The residual's own float64 floor is
~10⁻¹³ s, so a step that moves it by less than ~10⁻⁸ s measures roundoff; a step
large enough to be roundoff-free walks a strongly nonlinear axis (`SINI` near 1,
`KIN`, `SHAPMAX`) into curvature, or straight out of the physical domain and
into NaN. The gate is that *some* well-conditioned step reproduces the column —
a wrong column is wrong at every step. Budget: 10⁻⁶ relative.

| fixture | worst column | ‖(−M) − ΔQ‖∞ / max\|M\| |
|---|---|---:|
| `NGC6440E` | `RAJ` | 8.8e-08 |
| `sim1` | `DECJ` | 3.2e-07 |
| `sim2` | `RAJ` | 2.2e-07 |
| `pure_rotator` | `F1` | 9.3e-14 |
| `J1856-3754.sim` | `RAJ` | 5.0e-07 |
| `sim_dmx` | `DECJ` | 2.0e-07 |
| `sim_fd` | `RAJ` | 2.1e-07 |
| `sim_jump` | `RAJ` | 1.7e-07 |
| `sim_jump_ex` | `RAJ` | 2.3e-07 |
| `sim_sw` | `ELONG` | 1.5e-07 |
| `sim_dd` | `SINI` | 3.0e-07 |
| `J0955-6150.sim` | `PMRA` | 1.0e-07 |
| `sim_ddk` | `KIN` | 3.5e-07 |
| `J0453+1559.sim` | `STIGMA` | 7.2e-07 |
| `J1208-5936.sim` | `DECJ` | 1.7e-07 |
| `J2302+4442.sim` | `SHAPMAX` | 7.1e-07 |
| `J1802-2124.sim` | `SINI` | 3.8e-07 |
| `J1227-6208.sim` | `STIGMA` | 5.7e-07 |
| `sim_ell1k` | `OMDOT` | 7.5e-07 |

PINT's analytic matrix is kept as the R9.4 **oracle**, reachable as
`design_matrix(source="pint")` and compared in phase (test T9): its columns
ignore the TZR row's own parameter dependence and the feedback of one delay
into the corrected time later components see, and divide by the constant `F0`
rather than the doppler-shifted spin frequency. Substituting it into `Mmat`
would reintroduce the pulsar/engine disagreement this package exists to remove.

## The perturbative engine (SPEC §12, T14 and T15)

`certify()` compares the perturbative engine against its own fp64 parent on
±1σ and ±3σ per live axis plus eight random joint draws — never against PINT or
Vela, because what is under test is the *delta formulation*, not the physics.

Live set: `binary+astrometry` (binary family, sky, proper motion, parallax).
The float64 column caps the residual change at 20 µs (see the validity domain
below); the float32 column is a relative gate and runs uncapped.

| fixture | cases | float64 ‖Δ‖∞ | float64 Jacobian | float32 ‖Δ‖∞ | float32 Jacobian | float32 gate |
|---|---:|---:|---:|---:|---:|---|
| `sim_dd` | 52 | 2.2e-13 s | 1.1e-11 | 1.8e-11 s | 3.2e-07 | pass |
| `J0955-6150.sim` | 44 | 2.2e-13 s | 7.4e-12 | 1.5e-12 s | 2.9e-07 | pass |
| `sim_ddk` | 60 | 2.1e-13 s | 1.2e-11 | 5.9e-11 s | 3.1e-07 | pass |
| `J0453+1559.sim` | 56 | 3.2e-13 s | 8.0e-11 | 2.4e-12 s | 5.7e-07 | pass |
| `J1208-5936.sim` | 48 | 2.1e-13 s | 3.3e-10 | 2.7e-12 s | 5.0e-07 | pass |
| `J2302+4442.sim` | 56 | 2.9e-13 s | 1.8e-12 | 4.0e-10 s | 2.7e-07 | pass |
| `J1802-2124.sim` | 56 | 5.0e-13 s | 4.5e-10 | 1.9e-12 s | 1.8e-06 | pass |
| `J1227-6208.sim` | 44 | 3.0e-13 s | 5.0e-11 | 4.1e-13 s | 3.5e-06 | pass |
| `sim_ell1k` | 40 | 4.6e-13 s | 2.3e-09 | 7.4e-08 s | 3.0e-07 | pass |

**Every fixture passes both gates, `J2302+4442` included.** That fixture is a
DDS binary with `SHAPMAX = 9.18`, i.e. `sin i = 0.99989`, whose Shapiro
logarithm argument

```
1 − e cos u − (sin i / a1)·(α(cos u − e_r) + β sin u)
```

is a difference of O(1) terms landing at ~10⁻⁴ near conjunction. SPEC R11.3
rule 1 — the reference channel is float64 always, cast to the working dtype
only where it enters the perturbation channel — is implemented in
[`perturbative/dual.py`](../src/vela_jax/perturbative/dual.py). The reference
channel depends on nothing traced, so XLA constant-folds it under `jit` and
the extra precision is free at run time.

### The perturbative engine's validity domain

`sim_ell1k` reaches 7×10⁻⁸ s in **both** dtypes when the residual change is
uncapped, which is the tell: it is not a precision problem. The delta chain is
exact — the dual is a difference formulation, not a first-order expansion. The
*assembly* is first order:

```
Δr = −M·δ_lin − (ΔD − (F_tzr/F_i)·ΔD_tzr)
```

uses the **reference** Doppler factor, dropping `Δdoppler·ΔD`. That term is
second order in the residual change, and the diagnostic confirms it: scaling the
worst case down by 30×, the error falls by 900× while
`max_abs / max_scale²` stays flat at 3.7×10⁻⁴ s⁻¹.

| scale | ‖Δr‖∞ | error | error / ‖Δr‖² |
|---:|---:|---:|---:|
| ×0.03 | 4.0e-04 s | 6.1e-11 s | 3.76e-04 /s |
| ×0.10 | 1.3e-03 s | 6.8e-10 s | 3.74e-04 /s |
| ×0.30 | 4.1e-03 s | 6.2e-09 s | 3.72e-04 /s |
| ×1.00 | 1.4e-02 s | 7.1e-08 s | 3.65e-04 /s |

The offending case is a ±3σ step in `TASC` of 0.00165 d = **143 s** on a
`PB = 0.08 d = 6912 s` orbit — 2% of a period, moving the residual by 14
**milliseconds**. `sim_ell1k`'s par is simply not constrained in `TASC`; on any
real par this axis moves by nanoseconds.

So the rule, which `CertifyReport.quadratic_coefficient` reports:

> the perturbative assembly is good to ~4×10⁻⁴·(Δr)². For a 1 µs residual
> change that is 4×10⁻¹⁶ s; it reaches a picosecond only at Δr ≈ 50 µs, and a
> nanosecond at Δr ≈ 1.6 ms.

Well inside any posterior a sampler will visit. The T14 gate therefore caps at
20 µs — where the cross term is ~1.5×10⁻¹³ s, comfortably under the 10⁻¹² s
budget rather than sitting on it — and a separate test bounds the coefficient
itself on the uncapped suite. If a use case ever needs deltas that large, the
fix is to carry `Δdoppler` in the assembly: a few lines, since the chain
already computes it.

## DDR (SPEC §12, gates D1-D8)

The eighth family, added in v2.6. Measured on `tests/fixtures/sim_ddr.{par,tim}`
-- 20 barycentric zero-noise TOAs, `PB` one day, geometry off, the last two
straddling 10<sup>4</sup> reference orbits -- plus the scalar anchors Vela.jl's
own `test/test_ddr.jl` carries, and two geometry models `tests/test_ddr.py`
builds for itself (the committed fixture is barycentric, and a zero observer
vector cannot exercise a parallax projector).

### Against Vela.jl and against Vela's scalar anchors

| what | oracle | budget | measured |
|---|---|---|---|
| absolute residuals, `sim_ddr` (T4) | Vela.jl `SPNTA` | RMS ≤ 1 ns, max ≤ 10 ns | **RMS 8.0 ps, max 20 ps** |
| posterior-scale deltas, 4 joint draws (T7) | Vela.jl `SPNTA` | max ≤ 10 ns | **max 56 ps** |
| PK-chart delay at TASC + {0, 21600, 43200} s | `test_ddr.jl` table | 1 ps | **≤ 1·10<sup>-15</sup> s** |
| phenomenological chart (`DDRPK N`, `GGAMMA = OMDOT = 0`) | `test_ddr.jl` table | 1 ps | **≤ 1·10<sup>-15</sup> s** |
| FBX chart, `FB0 = 1/86400` | `test_ddr.jl` table | 1 ps | **≤ 1·10<sup>-15</sup> s** |
| injected-`(I, J)` geometry anchors | `test_ddr.jl` table | 1 ps | **≤ 3.1·10<sup>-14</sup> s** |
| `mp`, `g_gamma`, derived `κ` | `test_ddr.jl` | rel 10<sup>-8</sup>..10<sup>-10</sup> | 0.7741081166, 0.0071937506, 2.0501533373·10<sup>-6</sup> |
| kinematic `Pbdot`: `p`, `p_shk`, `p_gal`, `p_gw` | `test_ddr.jl` | rel 10<sup>-10</sup> | **rel ≤ 3·10<sup>-15</sup>** |

Reaching Vela at all needs one workaround, and it is upstream of this package:
pyvela's default-prior path has a `TASC`/`T0` branch that calls PINT's
`PulsarBinary.pb()`, and `pb()` reads `T0` for every model whose name does not
begin with `ELL1`. `BinaryDDR` is `TASC`-based and exposes no `T0`, so building
an `SPNTA` raises `AttributeError` before any physics runs. The oracle test
supplies an explicit `TASC` prior, which takes the earlier custom-prior branch
and steps around it; the bound is arbitrary, because these gates compare
residuals rather than posteriors.

### Against PINT

PINT is an independent oracle here, and the authority for one deliberate
deviation: Vela's DDR hard-codes the IERS2010 obliquity, this package resolves
the par's `ECL` (SPEC §7.3, §7.10b), and PINT honours `ECL` too.

| what | budget | measured |
|---|---|---|
| derived TGEO triad, real equatorial geometry, 100-day span (D3) | 1 ps | **0.89 ps** |
| `ECL` movement IERS2003 → IERS1992, isolated binary stage (D4) | moves materially; agreement 1 ps | **moves 615 ps; agreement 0.046 ps** |

Both comparisons freeze **one** pre-binary `Correction` and hand the same
object to each vela-jax evaluation and the same numeric accumulated delay to
each PINT component. Re-running each model's solar-system stage would change
`corrected_time` as well and would measure that instead of DDR geometry.

### One constant mismatch, pinned rather than fixed (D8)

The equatorial PINT comparison is gated at a **100-day** span deliberately. Over
three years it degrades to ~10 ps, and the excess is not geometry: it is a
linear-in-time drift with a single measured cause.

`M2` enters the layout in seconds through PINT's own `tcb2tdb_scale_factor`,
which is `GMsun/c³ = 4.9254909476412675·10⁻⁶ s`. DDR then recovers
dimensionless solar masses as `m2 / M_SUN` with Vela's hard-coded literal
`4.92549094830932·10⁻⁶`. The two differ by **−1.3563·10⁻¹⁰ relative**, so
`mc` is `0.7999999998914947` where the par said `M2 0.8`.

DD and ELL1 never see it: they use `m2` in seconds and the constant cancels out
of `−2 m2 log(...)`. DDR's GR maps do see it, and `κ` carries it into a
*secular* precession angle, so the disagreement with PINT grows linearly:

| span | drift vs PINT |
|---|---|
| 100 d | 0.89 ps |
| 1095 d | 9.8 ps |
| 10<sup>4</sup> orbits (the committed fixture) | ≈ 90 ps |

Ablation isolated it completely. With `κ = 0` (`DDRPK N`, `OMDOT 0`) the
agreement is 3·10⁻¹⁴ s and *flat* over the same 1095 days; a phenomenological
`OMDOT` six times smaller than `κ_GR` is also flat, so the drift is not simply
proportional to the precession rate. Field by field against PINT's own DDR
state, `x`, `n`, `c`, `s`, `c_e`, `s_e`, `I` and `J` all agree to ~10⁻¹⁶
relative; `X`/`Y` and their derivatives drift to 2·10⁻¹² by the last row; and
`g_gamma` is off by a constant 4.5·10⁻¹¹ relative -- the same mismatch seen
through a different power. Recovering the precession angle from the rotated
`(X, Y)` pair gives `Δδ/δ = −1.356·10⁻¹⁰` at every row, which is exactly
`mc/0.8 − 1`.

It is **not fixed**, for a stated reason. `binary/ddr.py` is a translation and
Vela.jl is the porting authority (SPEC §2); Vela.jl has exactly this
inconsistency, since pyvela also converts `M2` with PINT's scale factor and
then divides by Vela's literal. Changing it would move this engine away from
Vela.jl by the same 1.36·10⁻¹⁰ while nothing in the Vela or PINT budgets here
is threatened either way. `tests/test_ddr.py` pins the number instead, so it
cannot change size unnoticed, and `constants.M_SUN` cannot be edited without a
red test. **If an owner would rather the two agree, the one-line change is to
recover `mc` with the same constant the layout multiplied by** -- that is a
physics decision about which `T_⊙` the GR maps use, not a bug fix.

### What the consumer actually sees at an invalid point

vela-jax returns NaN and stops there (D13): turning a non-finite residual into
a rejected sample belongs to the sampler. The other half of that contract is
asserted in nltiming
(`tests/test_discovery_host_delay.py::test_an_engine_outside_its_domain_gives_a_non_finite_log_likelihood`),
on a stub engine that reproduces DDR's NaN boundary without needing a DDR par.

Measured: Discovery **propagates the NaN**, eagerly and under `jit`, so
`logL` is NaN -- *not* `-inf`, which is the word SPEC §12.5 uses. Both are
rejected by a Metropolis test (every comparison against NaN is false) and by
NumPyro's divergence handling, so nothing is wrong today. They are not
interchangeable for a sampler that branches on `isneginf` or that feeds the
value into adaptation, so the observed value is pinned separately from the
`not isfinite` gate and a change would be visible rather than silent.

### The regular-Kepler solver (D6)

The production loop count is **16**, chosen by measurement. The probe is
deterministic and re-run in `tests/test_ddr_performance.py`:

```python
rng = np.random.default_rng(20260916)
radius = 0.99 * np.sqrt(rng.random(2_000_000))
angle  = rng.uniform(-np.pi, np.pi, radius.size)
h, k   = radius * np.sin(angle), radius * np.cos(angle)
lam    = rng.uniform(-np.pi, np.pi, radius.size)
```

First-converged **0-based loop index** (the index of the pass whose *incoming*
`F` already met `|residual| ≤ 4 eps max(1, |λ_red|)`):

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| points | 0 | 124 | 65 849 | 768 793 | 928 014 | 176 723 | 59 771 | 704 | **22** |

Nothing fails within 64. The maximum is **8** -- the 9th pass -- so 16 retains
seven unused passes after it. The 22 index-8 points are committed as
hexadecimal float64 so the gate survives a NumPy RNG change. An assertion
written against a 1-based *pass count* would be off by one.

The committed stress grid (65 radii × 128 angles × 257 longitudes, including
both `nextafter` neighbours of ±π) converges within 16 at every point.

### The reduced longitude at 10<sup>4</sup> orbits (T13)

DDR's PB chart meets the 0.05 ps floor with room. Its **FBX** chart does not,
and the shortfall is one named rounding rather than an accumulation:
`fbx_mean_longitude` retains the integer orbits through `n_orb·(FB0·P★ − 1)`,
`P★` is the float64 `1/FB0★`, and `FB0·P★` rounds to **exactly 1.0** while its
true value differs by ~5·10⁻¹⁷. At 10⁴ orbits that is ~5·10⁻¹³ turns, about
**16 ps** of equivalent Roemer delay.

This is the reduction identity's own floor and is shared verbatim with
`binary/orbit.py::mean_anomaly`, which every FBX family already uses -- DDR did
not introduce it, and it is invisible on any fixture with a shorter baseline.
The test asserts that the error **is** that term to a relative 10⁻⁶ rather
than widening a tolerance, so a real regression still fails.

### Perturbative (T14, T15)

`sim_ddr`, mode `binary+astrometry` (the fixture freezes astrometry, so the
live set is `PB, A1, M2, TASC, EPS1, EPS2, COSI`):

| dtype | max abs | max rel | Jacobian rel | verdict |
|---|---|---|---|---|
| float64 | 4.23·10<sup>-15</sup> s | 2.11·10<sup>-10</sup> | 7.2·10<sup>-16</sup> | pass (budget 10<sup>-12</sup> s / 10<sup>-6</sup>) |
| float32 | 3.40·10<sup>-11</sup> s | 1.67·10<sup>-7</sup> | 3.4·10<sup>-7</sup> | pass (rtol 10<sup>-5</sup>, atol 10<sup>-12</sup> s) |

This is the first family to exercise `Pert.regular_kepler` and `Pert.cbrt`, and
the reference-channel convergence rule of R11.3-0: the difference solve runs in
the working dtype while convergence is read off the float64 reference channel.

### Cost (D7)

Measured with the protocol in `tests/test_ddr_performance.py`: one cold call
timed separately, five unmeasured warm calls, an inner repetition count chosen
so a sample is at least 50 ms and then shared by every kernel, 25 synchronised
samples in alternating order, medians compared. Absolute wall clock is
informational -- CI hardware varies -- and the **ratios** are the gate.

Device: CPU, aarch64 container, JAX 0.8.0, float64, 20 000 rows.

| comparison | median | ratio | ceiling |
|---|---|---|---|
| 16-step kernel | 3.83 ms | | |
| 64-step twin, same algorithm | 13.38 ms | **0.286** | ≤ 0.35 |
| production JVP (implicit rule) | 3.77 ms | **0.96** of one primal | ≤ 2 |
| `binary.orbit.mikkola` on the same rows | 1.24 ms | DDR/Mikkola **3.02** | ≤ 10 |
| `binary.DDR` stage | 3.40 ms | | |
| `binary.DD` stage, same 20 000 rows | 1.62 ms | **2.09** | ≤ 15 |

The two kernels return *identical* roots (max difference 0.0), so the loop-count
ratio is not a faster kernel stopping early.

Compile, once per shape: 0.62 s for the 16-step primal, 1.98 s for the 64-step
primal, 0.60 s for the production JVP.

**Two measurement traps, both hit before the numbers above were believed.**

*The stage comparison must trace the orbital epoch.* With only `A1` traced,
every input to the Kepler solve is a compile-time constant, XLA folds the whole
solve away, and the measurement reports **1.28** -- on a stage whose solver
alone takes longer than the entire time measured. With a live epoch (which is
also what a sampler moves) the ratio is 2.09.

*The "16-vs-64 gradient ratio" cannot be measured honestly.* With the implicit
custom JVP the differentiated cost is independent of the loop count, because
the rule evaluates the primal once and applies a closed form -- so such a ratio
only reports whichever primal the rule happened to call. And a 64-step twin
*without* a custom rule is not a usable reference: forward-mode through 64
nested `where`/bracket levels **did not finish compiling in six minutes** on
this device, against two seconds for the same kernel's primal. That is the
measured justification for the custom JVP, stronger than the design note it
replaces, and it is why the gradient gate is "the rule costs one primal solve"
rather than a loop-count ratio. The six-minute compile is recorded here rather
than paid for on every slow run.

## Frozen parameters the engine consumes or refuses (SPEC §12, gate P7)

`validate_model` allow-lists components and §6.1 refuses *free* parameters no
stage evaluates. R5.3b closes the remaining gap: a **frozen** parameter that
PINT applies must be consumed, known-inert, or refused by name.

`A0`, `B0` and `SWM 1` are refused by name. `ECL` is *read*: the obliquity is
resolved per engine from PINT's own table. On `sim_sw`, rewriting `ECL` from
`IERS2010` to `IERS1992` / `IAU1976` / `IERS2003` moves this engine by
1326.7 ns, 9286.7 ns and 22.1 ns — PINT's numbers to better than 0.1%.

`sim_dd` sets `A0`/`B0` at zero (the zero-skip accepts them); `sim_dmx` sets
`DMX 0.0`; no fixture sets a non-zero unimplemented term, and **every fixture
that sets `ECL` sets `IERS2010`** — the value Vela hard-codes. A fixture that
does not set a parameter looks identical whether the engine reads it or not,
so T3, T4 and T7 cannot see this class. Frozen non-zero bare `DMX` is inert
in PINT and here; a fitted bare `DMX` is refused as unconsumed.

Both the EPTA DR1 v2.2 and NANOGrav 9-year `J1853+1303` pars say
`ECL IERS2003`, as EPTA and IPTA release pars typically do. Against the
composite's own PINT pulsar, on the NANOGrav 9y leg (1369 TOAs), the engine
residual agrees to **0.4 ns RMS**.

## The two timing packages (SPEC A.6.1, gate H7)

A.6.1 is an absolute freeze floor: **RMS ≤ 100 ns, no relative term, no
cushion**, on every Vela.jl fixture tempo2 can open, with exclusions named
individually rather than absorbed into a wider bound.

This is **not** `PINT.Residuals` vs libstempo, and it is not a timing-model
test. `timing_package="pint"` and `"tempo2"` each freeze clocks, ephemeris,
and pulse numbers; `Engine.residuals()` is the same JAX delay chain on that
freeze. Identical freezes make the JAX residuals bit-identical even when the
two packages' own residual functions still differ. The load-bearing geometry
check on a tempo2 read is the Roemer closure of A.2.3 (≤ 1 ns required,
≤ 0.4 ps measured).

The files therefore have to be ones both packages can freeze the same way.
tempo2 has no `ECL` keyword. It always uses the IERS2003 obliquity
(`ECLIPTIC_OBLIQUITY_VAL` = 84381.4059″). PINT honours `ECL`, and Vela.jl's
ecliptic simulations say `IERS2010`. The gate sets `ECL IERS2003` on the
PINT copy so both packages use tempo2's default on the same ELONG/ELAT
numbers — a freeze pin, not a sky-coordinate transform. Without it the table
would report ~120 ns of Roemer from 0.1 mas of obliquity, a real PINT/tempo2
disagreement and nothing to do with clocks. Equatorial pars are a no-op.
Clock coverage, `PLANET_SHAPIRO`, solar-wind derivatives, and `TIMEEPH` are
the same class of freeze pin; A.6.1 names them.

| fixture | TOAs | pulse numbers | RMS(r_tempo2 − r_pint) | max |
|---|---:|---|---:|---:|
| `NGC6440E` | 62 | model | 1.34 ns | 3.72 ns |
| `sim1` | 2000 | model | 1.77 ns | 6.67 ns |
| `sim2` | 2000 | model | 1.77 ns | 6.33 ns |
| `pure_rotator` | 100 | model | 1.44 ns | 5.24 ns |
| `J1856-3754.sim` | 500 | model | 1.64 ns | 5.15 ns |
| `sim_dmx` | 1000 | model | 1.61 ns | 5.23 ns |
| `sim_fd` | 2000 | model | 1.53 ns | 5.82 ns |
| `sim_jump_clk` | 999 | model | 1.58 ns | 4.81 ns |
| `sim_jump_ex` | 2000 | model | 1.56 ns | 5.76 ns |
| `sim_sw_aligned` | 2000 | model | 1.20 ns | 5.38 ns |
| `sim_dd` | 1000 | model | 1.50 ns | 4.59 ns |
| `J0955-6150.sim` | 500 | model | 1.46 ns | 4.20 ns |
| `sim_ddk` | 2000 | model | 1.40 ns | 5.46 ns |
| `J0453+1559.sim` | 2000 | model | 1.52 ns | 5.23 ns |
| `J1208-5936.sim` | 500 | model | 1.49 ns | 4.54 ns |
| `J2302+4442.sim` | 500 | model | 1.54 ns | 4.98 ns |
| `J1802-2124.sim` | 500 | model | 1.69 ns | 4.66 ns |
| `J1227-6208.sim` | 500 | model | 1.53 ns | 4.54 ns |
| `sim_ell1k` | 500 | model | 1.57 ns | 4.76 ns |

The measured floor is **1.5 ns** on every row (solar-wind 1.20 ns on the
MetaPulsar-aligned files).

### Freeze fields the tempo2 reader must match

**Observatory velocity includes Earth's rotation.** tempo2 stores the site
*position* in `observatory_earth[0:3]` and leaves `[3:6]` at zero; the site
velocity is a separate field, and tempo2 itself forms the observatory
velocity as `earth_ssb[3:6] + siteVel` (`dm_delays.C:99`). Using the zero
half shortens `ssb_obs_vel` by the site's rotational velocity — 1.3% of the
total. It is invisible in the Roemer closure, which is a *position* check,
and it reaches the residual through the doppler-corrected observing
frequency: ~2·D_disp·Δdoppler, i.e. **234–900 ns** on these fixtures and
~100 ns on a real MSP at 1400 MHz. The tempo2 reader exposes tempo2's
`siteVel` property for it.

**PINT writes `CLOCK`, tempo2 reads only `CLK`.** PINT names the parameter
`CLOCK` with `CLK` as an alias; `readParfile.C` knows only `CLK` and silently
falls back to its own default realisation for anything else. A PINT-written
par that keeps `CLOCK` pins the clock chain for one timing package and not
the other — a flat 234 ns across every fixture whose par says
`CLOCK TT(BIPM2023)`. `read_tempo2` respells it for the tempo2 copy of the
par text, the mirror of the `FDJUMPn`→`FDnJUMP` respelling for PINT's copy
(A.2.8).

### Two Vela fixtures that are not this table

**`sim_jump`** is not H7. Its TOAs begin at MJD 50000, before GBT's clock
file. PINT extrapolates and warns; tempo2 applies no correction. The
replacement is `tests/data/sim_jump_clk`: the same equatorial JUMP
structure with SAT and MJD JUMP windows shifted +4000 d into clock
coverage (1.58 ns).

**`sim_sw`** is not H7. Native `PLANET_SHAPIRO Y` puts planet positions into
the freeze; PINT and tempo2 interpolate those independently. Native
`ECL IERS2010` and PINT-only `NE_SW1`/`SWEPOCH` are further package
disagreements. The replacement is `tests/data/sim_sw_aligned`: MetaPulsar's
mixed-engine shared par on the original TOAs — `ECL IERS2003` with the
numeric ELONG/ELAT transform, `PLANET_SHAPIRO N`, `NE_SW1`/`SWEPOCH`
stripped, `TIMEEPH FB90` (1.20 ns).

### Identical freezes are bit-identical JAX residuals

That is the J1909-sim case: barycentric `bat` TOAs make both freezes inject
`ssb_obs_pos = 0` and the same `tau`, so vela-jax agrees with itself at
0 ps. The packages' own residual functions on those files are 0.31 ns RMS:

| fixture | TOAs | RMS(PINT − libstempo) | max |
|---|---:|---:|---:|
| `J1909-3744-sim` | 100 | 0.31 ns | 0.62 ns |

That barycentric pair does not replace the observatory H7 table.

## The phase connection (SPEC A.6.2, gate H8)

A.6.2 asks for a fixture where `pulse_number_source == "tempo2"` is *required*,
constructed so that a silently dropped connection is at least half a period
wrong. The fixture is built in the test from `sim_dd`, whose tim already carries
`-pn` flags: the par gets `TRACK -2`, and two tims are written that differ only
in shifting the last quarter of the `-pn` values by one whole turn. Nothing in
the timing model distinguishes them, so a reader that ignored the flags would
return the same residuals twice; honouring them puts one spin period (10 ms) on
a quarter of the data.

### A bare `-pn` flag is not a phase connection

Bumping a `-pn` value by one turn on a tim with no `TRACK` changes tempo2's
residuals by **nothing at all** — `formResiduals.C:2263` reads the flags only
inside the `TRACK -2` branch. Most of Vela.jl's own fixtures carry decorative
`-pn` flags and no `TRACK`. The predicate is `TRACK -2` **and** `-pn`; a test
pins both halves.

## The binary conventions (SPEC A.6.3, gate S4)

A.6.3 asks for one ELL1 fixture measured against libstempo on matched clocks:
RMS ≤ 50 ns under `binary_conventions="tempo2"`, and the fixture must
*discriminate* — the same comparison under `"pint"` must fail.

The fixture is `J1227-6208.sim` (`e = 1.15×10⁻³`) reduced to a plain ELL1, built
in the test with three stated edits:

| edit | why |
|---|---|
| `BINARY ELL1H → ELL1`, drop `H3`/`NHARMS` | Vela and tempo2 also disagree about the ELL1H *Shapiro* harmonics — a separate finding, below — which would swamp the gate |
| `NE_SW 0` | mixed-engine pin: tempo2 defaults the solar-wind density to 4 cm⁻³ and PINT to 0 (~500 ns), nothing to do with the binary |
| `CORRECT_TROPOSPHERE N` | mixed-engine pin: tempo2 models it and this engine does not (SPEC §1.2); ~3 µs peak to peak |

The eccentricity is untouched: it is what makes the fixture discriminating.

| conventions | RMS(r − r_libstempo) | verdict |
|---|---:|---|
| `"tempo2"` | **3.3 ns** | passes the 50 ns gate |
| `"pint"` | **12 072 ns** | fails it by 240× |

The gap is the 22.9 µs peak that A.4 predicts for the ELL1 Roemer truncation at
this eccentricity. A near-circular EPTA ELL1 would pass 50 ns with the flag off
and prove nothing; this one cannot.

With the same three edits applied, `J1802-2124.sim` (a real ELL1, `e ≈ 2×10⁻⁶`)
lands at **0.26 ns** under `"tempo2"` and 1.8 ns under `"pint"` — inside the
gate either way, which is exactly why it is not the gate's fixture.

### Deviation-ledger finding: ELL1H's Shapiro harmonics

On the unmodified `J1227-6208.sim` (ELL1H), the same comparison gives 1.7 µs
under `"tempo2"` and 12.2 µs under `"pint"`. The conventions flag still
discriminates, but the floor is not the Roemer truncation — it is the
orthometric Shapiro parametrisation: Vela subtracts the `a0 + b1 sinΦ + a2 cos2Φ`
harmonics analytically (`binary_ell1h.jl`) where tempo2 does something else.
SPEC §16.6 asks Vela.jl whether tempo2's ELL1 truncation is meant to apply to
ELL1H at all. Per A.6.3 this is recorded as a finding, **not** taken as a
licence to port more `formBats`.

## float32 end to end (SPEC §12, gate T16)

`examples/nuts_fp32.py` runs the same pulsar twice through NUTS -- once with
the float64 engine in a float64 Discovery likelihood, once with the float32
perturbative engine in a likelihood whose linear algebra is
`working=float32` -- and compares the posteriors in units of their *combined*
Monte Carlo error, which is the only scale that means anything: two chains of
the same posterior differ by O(1) of it.

`certify()` already gates the delta formulation against the float64 parent on
posterior-scale deltas; this asks the harder question, whether the posterior a
sampler actually explores is the same one.

On `J1853+1303`, 500 warmup + 2000 samples, one chain, on an RTX 4070 Ti
SUPER:

| leg | TOAs | fp64 | fp32 | worst watched axis | worst overall | fp32 divergences |
|---|---:|---:|---:|---:|---:|---:|
| NANOGrav 9y, PINT | 1369 | 51.7 s | 99.7 s | `A1` at 1.13 | 1.13 | 0 |
| EPTA DR1 v2.2, tempo2 | 101 | 44.3 s | 62.7 s | `A1DOT` at 0.45 | 0.71 | 0 |

Both are inside one Monte Carlo error on every axis, which is where two chains
of the *same* posterior sit. The NANOGrav leg is the harder of the two — a
14× larger data set and an ECORR block — and it is the one that would show a
float32 delta channel drifting if it did.

x64 stays enabled throughout: Discovery's float32 mode is a working dtype for
the expensive factorizations, not a global switch, and nltiming refuses to
sample timing deltas without float64 available. A MetaPulsar composite is
refused by the script rather than half-supported: each leg would need its own
perturbative engine, which is MetaPulsar's dispatch to make.

## End to end, on real data

Beyond the fixtures, `examples/nuts_j1853.py` runs NUTS through Discovery's
likelihood on `J1853+1303` from the public EPTA DR1 v2.2 and NANOGrav
9-year releases, with per-backend white noise and a power-law red-noise GP held at the
dictionary values and the timing block sampled through nltiming's dynamic
decentering. 300 warmup + 800 samples, one chain:

300 warmup + 800 samples, one chain, on an RTX 4070 Ti SUPER:

| configuration | TOAs | fitpars | sampling | divergences | worst \|z\| vs the par | ESS |
|---|---:|---:|---:|---:|---:|---:|
| EPTA DR1 v2.2, tempo2 | 101 | 19 | 29.2 s | 0 | 1.4σ | 960–2220 |
| NANOGrav 9y, PINT (with ECORR) | 1369 | 18 | 33.7 s | 0 | 1.5σ | 780–2120 |
| both, combined by MetaPulsar, one vela-jax leg each | 1470 | 37 | 55.9 s | 0 | 1.7σ | 370–1640 |

The combined run is the sharpest check available without a second oracle: the
`per_pta` strategy gives each leg its own copy of `A1`, `PB`, `TASC`, `EPS1/2`,
and the two legs — read by *different codes*, on disjoint TOAs — agree with each
other and with the par to well inside their widths. `A1` is the clearest case:
40.769522 50 ± 6.8×10⁻⁷ from the tempo2-read leg against
40.769522 56 ± 7.4×10⁻⁸ from the PINT-read one, agreeing to a tenth of the
narrower width. `PB` agrees to 0.3 of it.

Since the legs are built by the timing side (`engines="vela_jax"`), there is
no second read to align: the record's `Mmat` *is* the leg engine's own `−J`,
and MetaPulsar checks that by block equality on every product leg
(`validate_composite_against_pulsar`). Neither the engine nor MetaPulsar
permutes rows (R5.5), so nothing has to be un-permuted either. The
`check_row_alignment` guard remains on the from-par/tim path, where a second
read does happen and a misaligned leg would line up against the wrong rows of
the composite design matrix and give a plausible, wrong posterior.
