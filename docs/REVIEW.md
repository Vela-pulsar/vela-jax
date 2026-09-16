# Reviewing vela-jax

This is Vela.jl's deterministic delay chain — `correct_toa` over the component
list, `form_residuals`, the TZR TOA, no mean subtraction — evaluated as a JAX
function of the timing parameters. PINT or tempo2 still opens the par and tim:
clocks, TT→TDB, the JPL ephemeris, pulse numbers. The residual that comes out
is `jit`-able and `jacfwd`-able, so a PTA likelihood can move nonlinear timing
together with the noise and GW model.

The physics is Vela's. The interesting work is everything JAX and the two
timing packages force: the build-time reductions that stand in for
`Double64`, the tempo2 reader, and the frozen pulsar record that Enterprise,
Discovery and nltiming consume. [`SPEC.md`](../SPEC.md) is normative;
[`CONFORMANCE.md`](CONFORMANCE.md) maps every requirement to a test;
[`PARITY.md`](PARITY.md) has the numbers.

Cite the first Vela.jl paper when using this package (Susobhanan 2025, ApJ
980, 165; [`CITATION.cff`](../CITATION.cff)). The license is GPL-3.0-or-later.

---

## Where to start

Twenty minutes:

1. [`precision.py`](../src/vela_jax/precision.py) — the spin-phase and orbital
   reductions. A line-by-line port of Vela would lose nanoseconds here: JAX
   has no `Double64`.
2. [`binary/ell1.py`](../src/vela_jax/binary/ell1.py) or
   [`binary/dd.py`](../src/vela_jax/binary/dd.py) — the binary delay, whichever
   family is more familiar. The coefficients come from Vela.jl, not from a
   re-derivation.
3. [`perturbative/dual.py`](../src/vela_jax/perturbative/dual.py) — how a
   float32 working dtype still keeps a float64 reference channel. Every
   place a reference quantity enters the perturbation is an explicit `_cast`.

An hour: add [`pipeline.py`](../src/vela_jax/pipeline.py) (Vela's component
order), [`freeze.py`](../src/vela_jax/freeze.py)
(`prepare_model` is `fix_params`; `freeze` is `pint_toa_to_vela`,
columnar), [`PARITY.md`](PARITY.md), and
[`pulsar_data.py`](../src/vela_jax/pulsar_data.py). The load-bearing cutoff
in the record is that `toas` is the barycentric arrival *before* the binary,
not the fully corrected TOA — those differ by seconds on any binary.

Safe to skip: `units.py` (Vela's `get_scale_factor`), `taylor.py`,
`constants.py` (Vela's literals, each with its source line), and
`backend.py` / `pulsar.py` (no delays in them).

---

## Where this differs from Vela.jl

### Spin phase and orbital phase

Vela carries the spin phase in `Double64`. JAX does not have that type. At
build time the whole reference spin series is evaluated at the undelayed TDB
in `numpy.longdouble`, and the trace sees only the Taylor re-expansion in
the delay:

```
phi_ref = φ(τ) − N − φ(τ_tzr)
ψ       = phi_ref + c₁ξ + c₂ξ²/2 + …  +  Σ δF_k t^(k+1)/(k+1)!
```

with `ξ = −delay`. Nothing of size `F0·(t − PEPOCH)` is ever multiplied
inside the graph. Folding the full series in — not just `F0` — is what keeps
a fast spinning-down pulsar at Vela's own float64 floor (`sim_dd` has
`F1 = −4.2×10⁻⁸ Hz/s`; the remainder against Vela is 0.12 ns).

The orbital phase `2π(t − T0)/PB` is reduced the same way: an exact integer
orbit count `n` is frozen at build time, and a live `PB` stays exact as
`Φ = 2π[Δt_red/PB − n·δPB/PB]`. The integer is put back in one place, the
unwrapped true anomaly that drives `OMDOT` (`v = v_u + u + 2π n`). That
term is secular, not periodic; without it `J0955-6150` is wrong by
milliseconds.

### Ingest

`Correction` drops Vela's `efac` / `equad2` fields. White noise belongs to
Discovery or Enterprise; this package never sees it. `TNRED*`, `RNAMP` /
`RNIDX` and `PLREDFREQ` are stripped from the par before PINT loads it, so
`prepare_model` is `fix_params` without the noise half. Five extra names sit
on the zeroable list (`EDOT`, `OMDOT`, `GAMMA`, `DR`, `DTH`) so a par that
omits them does not arrive in the DD state as a missing key.

### Tracing

Mikkola's `e == 0 || l == 0 → l` short-circuit, and every other Vela `if`
that guards a singular expression, becomes *substitute a safe value, then
select*. Both sides of a traced `where` run, and a NaN in the unused branch
poisons `jacfwd` even when the residual is right. The rule lives in
`binary/orbit.py` and is used throughout.

Proper motion's `pmα == pmδ == 0` short-circuit is dropped: the general
formula already returns the reference direction.

### Infinite frequency

A TOA — usually the TZR row, if `TZRFRQ` is absent — can have PINT's
infinite observing frequency. `DM/ν²` then has the right value and an
`inf · 0` derivative, which NaNs every Jacobian column. The frozen
frequency is finite everywhere; dispersive terms are selected off through
`correction.inverse_freq_sqr`. Vela never hits this because it does not
differentiate. `J0453+1559.sim` does, immediately.

### Troposphere and a par with no sky

`CORRECT_TROPOSPHERE Y` is accepted with a warning and not modelled. The
delay is ~10 ns and nearly constant; it cancels out of every residual
*difference*. Vela leaves the troposphere commented out as well, and
refusing the flag would block most EPTA and IPTA files.

Astrometry is optional. A `pure_rotator` builds, as it does in
`pint_components_to_vela`.

---

## Reading the files

PINT is the default reader. For an EPTA or IPTA product that is often the
wrong one: the file wants tempo2's `INCLUDE` handling, its clock chain,
`TRACK -2` pulse numbers, and its site and ephemeris vectors.
`Engine.from_tempo2` lets tempo2 do that job and freezes *those* arrays into
the same component chain. PINT still parses the TDB par for the BINARY
family, the masks and the units.

In tempo2's own frame, `psrPos · R − ½ PX R_⊥²` has to reproduce tempo2's
`roemer`. It does, to ≤ 0.4 ps. That closure is a units-and-composition
check, not a timing-model test.

A few things the reader has to get right, because the obvious substitute is
quietly wrong:

* **Observatory velocity.** tempo2 keeps the site *position* in
  `observatory_earth[0:3]` and leaves `[3:6]` at zero. The velocity is
  `siteVel`; tempo2 forms the observatory velocity as
  `earth_ssb[3:6] + siteVel`. Using the zero half drops the Earth's
  rotation — 1.3% of the velocity, ~100 ns of dispersion through the
  Doppler-corrected observing frequency. The Roemer closure cannot see it:
  that is a position check.
* **Clock keyword.** PINT writes `CLOCK`. `readParfile.C` knows `CLK` and
  otherwise falls back to its default realisation, so a PINT-written par
  would pin the clock chain for one timing package and not the other. The
  tempo2 copy is respelled, the mirror of `FDJUMPn` → `FDnJUMP`.
* **Phase connection.** tempo2 reads `-pn` flags only inside the
  `TRACK -2` branch (`formResiduals.C`). A decorative `-pn` without
  `TRACK` does not connect the data. Pulse numbers come from tempo2 only
  when both are present, and they are re-referenced onto PINT's fiducial.

With mixed-engine-consistent files (clock coverage, shared ecliptic frame,
``PLANET_SHAPIRO N``) and ``ECL`` set to IERS2003 so PINT uses tempo2's
default obliquity, the two timing packages agree to 1.5 ns RMS on every H7
row, against a 100 ns gate. Vela ``sim_jump`` and ``sim_sw`` are replaced;
see [`PARITY.md`](PARITY.md#two-vela-fixtures-that-are-not-this-table).

A tempo2-fitted ELL1 also has to be evaluated under tempo2's ELL1 Roemer
truncation (`ELL1model.C`). That is the `binary_conventions` flag, not a
second physics chain.

### Ecliptic frame

Vela hard-codes `OBL = 84381.406″` (IERS2010). A par's `ECL` keyword
selects the realisation its `ELONG` / `ELAT` were defined in, and PINT
honours it. `ECL IERS2003` (84381.4059″) is 0.1 mas away — about 100 ns
RMS of Roemer delay on a millisecond pulsar. EPTA and IPTA release pars
typically say exactly that; every Vela fixture that sets `ECL` sets
`IERS2010`, so a fixture suite that only compares to Vela cannot see the
keyword.

The obliquity is resolved from `ECL` against PINT's table and recorded on
the engine. On the NANOGrav 9-year `J1853+1303` leg the engine and the
composite's own PINT pulsar then agree to 0.4 ns RMS; evaluating that par
at Vela's hard-coded IERS2010 is 98 ns out.

tempo2 has no `ECL` keyword. It has already rotated every ephemeris vector
with `ECLIPTIC_OBLIQUITY_VAL` (IERS2003), so the line-of-sight rotation
back to ICRS has to use the same constant or the composition is not the
identity.

### Parameters the chain does not consume

A par may only set a parameter this engine consumes, knows to be inert
(with a stated reason), pins to one value (`SWM 0`), or refuses by name.
`A0 = 1e-5` moves PINT by 14 µs; silently dropping it would move this
engine by nothing. Wideband TOAs, `SWM 1`, and the same class of unused
frozen parameters raise rather than build a narrowband residual and
pretend. Bare `DMX` is inert (PINT info line; delay is `DMX_NNNN`).
`TIMEEPH`/`T2CMETHOD` are inert ingest flags, not pinned conventions.

`INERT_PARAMS` is a reading of what each parameter does, not a proof.
`RM`, `NHARMS` and the read-side settings are the entries most worth a
second look.

### Row order

The freeze does not reorder TOAs. Residuals, the design matrix and the
feather are in the timing package's storage order. Discovery's `quantize`
and Enterprise's `create_quantization_matrix` argsort internally and write
ECORR membership back in the caller's order anyway — they group by value,
not adjacency.

When a MetaPulsar leg builds the engine from a second read of the same
par/tim, site arrival times are compared against the composite's rows and
the leg is refused if they differ.

---

## Residual, design matrix, gauge

The design matrix is `−J`, the Jacobian of this residual, not PINT's
analytic matrix. That is the matrix a sampler should marginalize: the
tangent of the residual it samples.

Vela forms the residual by dividing phase by the Doppler-shifted
*instantaneous* spin frequency. PINT divides by the constant `F0`. The two
design-matrix columns therefore differ by that ratio before anything else,
and PINT's analytic derivative also omits the TZR row's own parameter
dependence and the feedback of one delay into the corrected time later
components see. Linear-column agreement with PINT is therefore compared in
*phase*, at 10⁻³, not in seconds. Against Vela, `J` itself is gated by
finite differences (M1).

The same instantaneous-frequency residual means a phase offset moves TOA
`i` by `1/F_i`, not by a constant. nltiming's gauge check accepts a
declared `gauge_direction()` for that; without it the constant-direction
test fails (a part in 10⁴ on a real MSP, 7% on `sim_dd`). `PHOFF` is kept
free because that column *is* the gauge, and without it a `TimingPulsar`
is rejected before the likelihood is assembled.

Agreement with PINT `Residuals` is a coarse sanity check (10⁻⁷ s, 10⁻⁶
for ELL1H/DDH). PINT and Vela disagree on the orthometric Shapiro
parametrisation — Vela subtracts the `a0/b1/a2` harmonics analytically,
PINT truncates a series at `NHARMS`. On `J1227-6208` that is 0.6 µs. Vela
is the oracle there (12 ps).

---

## The pulsar record

`PulsarData` / `TimingPulsar` is a frozen record of named arrays from one
`Engine`: residuals, `−J`, and the fitpar list are the same objects. It is
nltiming's pulsar, an Enterprise pulsar in the `FeatherPulsar` sense, and
the in-memory form of a versioned feather file. PINT's physics methods are
monkeypatched to raise while the record is built, so a second evaluation
cannot sneak in.

Neither module imports nltiming. The protocols are structural; a
mandatory CI job installs nltiming and checks the shape.

---

## Single precision

The default engine is float64 and refuses to build without `jax_enable_x64`.
A PTA likelihood is dominated by the `O(n_par² n_TOA)` timing
marginalisation and the red-noise / GW matrix work, not by the `O(n_TOA)`
residual. The recommended path is this engine plus a cast of `r` and `M`
at the likelihood boundary, which JAX differentiates.

`engine.perturbative(mode, dtype=jnp.float32)` is the other case: a
Discovery kernel whose working dtype is already float32 and which wants
the live nonlinear axes in the same kernel. The component chain is written
against [`numerics.py`](../src/vela_jax/numerics.py) rather than `jnp`, and
the dual value `(reference, perturbation)` is the cancellation-free
difference identity (`sin` by half-angle, Kepler in the difference
variable, …). Running the same chain over it yields `ΔD(δ)` for every
component, including ones a hand-written delta kernel would never list.
The identities are tested against a `longdouble` oracle: a naive float64
`f(x+h) − f(x)` is only good to 10⁻¹⁶, which is the cancellation they
exist to avoid.

The full residual cannot run in float32. A 500 s Roemer delay resolves to
3×10⁻⁵ s there.

---

## Measurements

**Binary Doppler.** The term is `ΔREp · n̂` (`drep * nhat` in DD). In
Vela's component order the binary runs after dispersion, so the factor
never hits the DM delay; it entered only the residual denominator
`1/(F_spin·(1+doppler))` and FD -- and since the divisor dropped the doppler
(SPEC §8), only FD. At the measured `a1·n̂` of 4×10⁻⁵–4×10⁻⁴,
a 1% error in the term is ~1 ps on a microsecond residual.

**Proper motion.** Vela (and tempo2's `calculate_bclt.C`) apply linear
motion and renormalise. PINT uses astropy `apply_space_motion`. Over the
fixture set the two lines of sight differ by ≤ 7.7×10⁻¹⁶ rad, 0.4 ps of
Roemer delay — the float64 floor. tempo2's `dt_pmtt` *is* that
second-order normalisation term; matching PINT would break the Roemer
closure, not fix anything.

| fixture | \|PM\| | span | max \|ΔL̂\| | Roemer |
|---|---:|---:|---:|---:|
| `J1802-2124.sim` | 5.1 mas/yr | 5.5 yr | 7.7e-16 rad | 0.37 ps |
| `J0453+1559.sim` | 8.1 mas/yr | 5.5 yr | 3.4e-16 rad | 0.16 ps |
| `sim_ddk` | 1.8 mas/yr | 5.5 yr | 3.2e-16 rad | 0.12 ps |

No fixture here has both a large proper motion and a long baseline. The
J0437-class estimate (~50 ns over 20 yr) is far outside this set. The
JAX graph will not call `pmsafe`: it is C, and not traceable.

**ELL1 inverse-timing.** Vela's cubic expansion and tempo2's
`ELL1model.C` truncation are a flag, and they are not a small difference:

| fixture | family | e | a1 | \|Vela − tempo2\| |
|---|---|---:|---:|---:|
| `J1802-2124.sim` | ELL1 | 2.5e-6 | 3.7 ls | 3.6 ns |
| `sim_ell1k` | ELL1k | 2.0e-5 | 0.075 ls | 0.13 ns |
| `J1227-6208.sim` | ELL1H | 1.2e-3 | 23.2 ls | **22.9 µs** |

Two scales: `a1·e²` from the Roemer delay, and `a1²·n̂·e` from the
derivative — tempo2 drops the `ε₁sin2Φ + ε₂cos2Φ` term from `drep`. On
`J1802` the second dominates. Nanosecond PINT/Vela parity is expected
only for a genuinely circular ELL1. A tempo2-fitted par has to be
evaluated with the convention it was fitted under.

Against libstempo on matched clocks, `J1227-6208` reduced to a plain ELL1
(`e = 1.15×10⁻³`) agrees to 3.3 ns under `"tempo2"` conventions and is
12 072 ns out under `"pint"`. The unmodified ELL1H par sits at 1.7 µs
under `"tempo2"`: the remaining piece is the orthometric Shapiro
harmonics, not the Roemer truncation.

**DDK `K96`.** Vela's behaviour (always apply the PM terms). `K96 N`
raises rather than silently doing something else.

**Live epochs.** `T0` and `TASC` are free and gated against pyvela,
including the interaction with the orbit-count reduction.
`POSEPOCH` / `DMEPOCH` are refused because nothing in the chain needs
them live; the parameter layout already carries a `δ` for every axis.
`PEPOCH` is the time origin the reductions are built on. Moving it means
rebuilding `phi_ref`.

---

## What needs discussion

**Per-component delays from Vela.jl.** Parity today is end to end:
`‖r − r_Vela‖∞` over a whole fixture. That catches a wrong component but
localises nothing, and a compensating pair of errors would survive. A
JSON export from `Vela.jl/test` of `inputs → (delay, phase, doppler)` per
component would drop straight into tests that are already parametrised
that way.

**A high-PM, long-baseline fixture** would settle whether linear-then-
renormalise still agrees with astropy at J0437-class `μT`.

**ELL1H Shapiro harmonics** against libstempo (1.7 µs on the unmodified
par). The Roemer truncation is constrained; this is not.

**`K96 N`** — raise, or honour Vela and ignore the flag.

**The inert list**, especially `RM`, `NHARMS`, and the read-side
settings.

---

## Limits

The tempo2 path is constrained, but narrowly: the Roemer closure
(≤ 0.4 ps), bit-identical `from_pint` on the injected table, a 100 ns
PINT–tempo2 clock floor (1.5 ns measured, `ECL` pinned to IERS2003), and
one discriminating libstempo ELL1 (3.3 ns). That last is one pulsar and
one binary family. It constrains the binary-input conventions, not the
reader in general.

Wideband TOAs, glitches, chromatic components and the GP delay bases
(`WaveX`, `PL*NoiseGP`, …) are absent, not approximated. Each raises by
name.

Without per-component Vela tables, end-to-end parity cannot say *which*
delay is wrong.
