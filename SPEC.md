# vela-jax — Vela.jl's timing-delay engine, in JAX

**Status:** normative specification v2.6 (2026-09-16). **Self-contained**: this
document supersedes the v0.2 design draft entirely. No earlier draft is
normative.

**v2.1 amends v2.0 in two places, both from implementation evidence.** §5.5
required a stable argsort on `tdbld` at the freeze; it now **forbids any
reordering**. The requirement was built on two claims about the consumers that
turned out to be false, and what it bought in practice was a permutation
protocol spanning two packages, invisible on already-ordered files — the
reasoning is in §5.5 and the disposition in Appendix A. A.2.6 said `TRACK -2`
*or* `-pn` flags; tempo2 reads the flags only inside the `TRACK -2` branch, so
it now says *and*.

**v2.2 adds the parameter-accountability rule (R5.3b) and reads `ECL`.** v2.0
and v2.1 allow-listed *components* and refused *free* parameters no stage
evaluates. Between those sat a gap: a **frozen** parameter PINT honours and no
stage here consumes was dropped in silence — `A0`/`B0` (14 µs on `sim_dd`),
`SWM 1`, wideband TOAs, and `ECL`, which is ~100 ns of Roemer delay on an
EPTA/IPTA par that says `IERS2003`. Fixtures that set `A0`/`B0`/`DMX` set
them at zero, which the zero-skip accepts; a non-zero unimplemented term is
what a parity pass cannot see. §5.3b closes the class; §7.3 resolves the
obliquity from the par instead of hard-coding Vela's.

**v2.6 adds `BINARY DDR` as the eighth binary family.** DDR is a third
Kepler convention beside DD and ELL1, not a DD subtype: it solves the
*regular* Laplace-Lagrange equation `F − k sin F + h cos F = λ` in native
`(EPS1, EPS2, TASC)` coordinates and advances periapsis through a regular
`q = ν − M`. Four things follow, and each is normative below: a fixed 16-pass
solver with an implicit-function JVP (§7.10b) because a traced array cannot
break out of Vela's 64-step loop; a **traced** validity mask (§7.10b), because
DDR's physical domain is derived from sampled parameters rather than frozen
ones, which makes `numerics.where`'s condition traced for the first time
(§11.3); the perturbative substrate gains `cbrt`, `regular_kepler` and
`nan_where` (§11.4); and R4.5/R4.5b apply to DDR's periodic longitude and
secular precession exactly as they do to DD's true anomaly (§4.5). DDR is
**PINT-host-only** — tempo2 has no such model and `Engine.from_tempo2`
refuses it by name (§1.2). DDR does **not** claim nltiming's DD
polar-to-Laplace chart: `BinaryFacts` gains `supports_domain`, and DDR reports
`kepler_convention="ddr"`, `epoch_shift_exact=False`, `supports_domain=False`
(§9). Unlike Vela's DDR, this one rotates its geometry with the par's resolved
`ECL` (§7.3's rule, applied in §7.10b), which PINT gates.

**v2.5.** A.6.1 is a freeze comparison: `Engine.residuals()` on two
timing-package reads of the same files, not `PINT.Residuals` vs libstempo.
Identical freezes are bit-identical even when the packages' own residual
functions differ. The 100 ns budget therefore requires mixed-engine-consistent
files — both packages apply clocks; a shared ecliptic frame (`ECL IERS2003`
or equatorial); `PLANET_SHAPIRO N`; no PINT-only `NE_SW1`/`SWEPOCH`;
explicit `TIMEEPH FB90` and `CLK`. Native Vela `sim_jump`/`sim_sw` are not
this gate. A.6.3's `NE_SW`/`CORRECT_TROPOSPHERE` edits are the same pins.
Measured floor ~1.5 ns (`PARITY.md`).

**v2.4.** FD and FDJUMP sit after the binary in one `STAGE_ORDER`, matching
PINT and Vela: they are a residual/phase effect, not a shift of the time
the binary sees. `binary_conventions="tempo2"` is the ELL1 Roemer
truncation only (A.4). Physics of every other stage is unchanged.

**v2.3 records the Vela.jl author's review of the archived letter.** §4.1
uses the worst-case numbers he asked for (`F0 ~ 800 Hz`, `Tspan ~ 100 yr`,
`D ~ 1000 s`), states its rounding convention, and names the platform floor.
§11 states that fp32 timing is optional, not required: the default engine
is float64 and a consumer MAY cast residuals and the design matrix for
matrix work. §16.1–16.3 are closed: separate GitHub repo, GPL-3.0-or-later,
cite Susobhanan 2025 ApJ 980 165; the "is this accurate enough?" comment on
the binary Doppler term is retired by its author. Physics, reductions and
the component chain are unchanged.

**Import name:** `vela_jax` — distribution `vela-jax`
**Authors:** Rutger van Haasteren (with Claude)
**Consumers:** `nltiming` (NumPyro NUTS through Discovery; PTMCMC through Enterprise), MetaPulsar (one thin per-PTA engine per leg), Enterprise/Discovery (through the pulsar product of Addendum B)
**Non-consumers, by design:** noise modelling, priors, likelihoods, samplers, fitters, GUIs, par writers

v2 history: v0.2 was the pre-implementation design (its Part I was a letter to
the Vela.jl author). That letter, and the hand-written perturbative kernels it
specified, are withdrawn; this document is the authority. The v1 prototype
implemented the draft, measured it, and drew a ruthless external review;
v2.0 is that review turned into rules
with gates — same physics core, an honest pulsar product, tightened tempo2
evidence — merged back into one document so an implementer never has to hold
two specs and guess the merge.

Part I is the human-language verdict and thesis. Part II is the normative
specification, down to code. Part III is the ecosystem proposal that follows
from Part II — and, by rule, never blocks it.

---

# Part I — Verdict and thesis

## I.0 For the Vela.jl reader (the short version)

This package is Vela.jl's deterministic delay chain — `correct_toa` over the
component list, `form_residuals`, the TZR TOA, no mean subtraction —
evaluated as a JAX function of the timing parameters, so that a
gradient-based sampler can move nonlinear timing inside a PTA likelihood.
PINT (or tempo2) remains the timing package exactly as pyvela uses it: clock
corrections, TT→TDB, solar-system ephemeris, pulse numbers. Where Vela
stores the TOA and phase in `Double64`, this port reduces the two large
quantities — the spin phase and the orbital phase — once, at build time, in
NumPy longdouble (§4); everything the JAX graph sees is small. Where Vela
and PINT disagree, Vela wins and the difference is gated per fixture. The
original design letter — what stays the same as Vela, what is different and
why — is this Part I. The remaining asks live in §16.

The rest of Part I is a post-mortem of the first prototype and the design
consequences it forced; the physics contract resumes at §3.

## I.1 What the v1 prototype is, measured, not narrated

The prototype has three parts of very different quality:

**The engine is finished work.** One Vela chain, frozen timing-package arrays, two longdouble
build-time reductions, a dual-number perturbative overlay through the small
`numerics` module, refusals by name. Measured: ≤ 124 ps worst-case against
Vela.jl over 19 fixtures (median 1.7 ps), `residual_delta(0)` exactly zero,
perturbative fp64 ≤ 6.6×10⁻¹³ s, end-to-end NUTS on real EPTA+NANOGrav data
with zero divergences. The review pack (`docs/REVIEW.md`,
`docs/CONFORMANCE.md`, `docs/PARITY.md`) exists and is honest about every
deviation. **v2 changes none of this**; §3–§8 below carry it forward with the
implemented improvements promoted to normative.

**The timing packages are right in design and were under-gated in test.** The
timing-package/physics split (PINT *or* tempo2 reads; Vela computes) is the one idea in
this package no other implementation has, and the build-time Roemer closure
(≤ 0.4 ps against tempo2's own `roemer`) is the correct load-bearing check.
But the PINT–tempo2 residual gate was `std(diff) < 0.05·std(r) + 10 µs` — a
pulse check, not a gate — `TRACK -2` was exercised but never *required*, and
`binary_conventions="tempo2"` was validated as a mechanism, never against
libstempo. Addendum A.6 fixes all three.

**The pulsar rebuilt the pulsar-vs-engine split inside one `__init__`.** v1's
`TimingPulsar` advertised "the same objects the engine uses" and delivered it
for `residuals`, `Mmat`, `fitpars` — then computed `toas` and `freqs` with a
*second PINT physics pass* (`get_barycentric_toas`, `barycentric_radio_freq`)
that on the tempo2 was PINT physics over injected columns: not the
chain, not tempo2, not tested. `dmx` was `{}` on files that have DMX.
`can_use_engines` and the README disagreed about what the object can do.
Addendum B replaces it. (v1 was also charged with leaving its arrays "in the timing package's
row order while every consumer downstream assumes time order". That charge was
false — see §5.5 — and v2.0's first draft acted on it. That row order was the one
thing v1 had right.)

Two review claims we checked and rejected, for the record: the "ghost docs"
exist (`docs/REVIEW.md`, `docs/CONFORMANCE.md` are in the tree and
substantive), and Venus *is* frozen (`freeze.PLANET_COLUMNS`) — the hole was
the pulsar-surface slot, not the freeze. Every other structural criticism was
accepted, and this spec is the fix.

## I.2 The thesis: pulsars are views of the timing package

Today's ecosystem is built backwards. Enterprise's `PintPulsar`/`Tempo2Pulsar`
freeze `toas`, `residuals`, `Mmat` from one code; nltiming then hangs a
*different* engine beside them as the nonlinear residual; Discovery reads a
feather file that is a serialized Enterprise pulsar. The consumer packages own
the pulsar; the timing packages own nothing but a constructor argument. That
inversion is the root of every "the engine and the pulsar disagree" bug class,
and the reason nltiming carries `validate_engine_against_pulsar` — a validator
which, we verified, **has zero call sites**: the consistency check exists as
documentation because it cannot be made to pass in general.

Three facts, established by reading the consumers rather than their docs, make
the flip cheap:

1. **The consumed surface is small.** The union of what core Enterprise and
   Discovery actually read from a pulsar object is: `name, toas, residuals,
   toaerrs, freqs, Mmat, fitpars, flags, backend_flags, telescope, pos, pos_t,
   planetssb[:, {2,4,5,6,7}, :3], sunssb[:, :3], dm, dmx, stoas` (`stoas` and
   `dmx` only by `WidebandTimingModel`). Everything else Enterprise stores —
   `setpars`, `pdist`, `theta`, `phi`, `designmatrix_units`, all `planetssb`
   velocities, planet slots 0/3/8 — is serialized dead weight core never
   reads.
2. **The attribute names are the interface.** Enterprise's `selection_func`
   (`selections.py:36`) and Discovery's `makedelay`/`make_extsignal_fourier`
   (`signals.py:1638`, `1704`) bind function arguments to pulsar attributes
   *by name via `hasattr`*. In Discovery, an attribute that stops existing
   silently becomes a *sampled parameter*. The pulsar contract is therefore a
   record of named arrays in one fixed order — nothing more.
3. **Enterprise already runs on a plain attribute record.** `FeatherPulsar`
   has no properties and no live timing object — just arrays with the right
   names — and it drives the full Enterprise likelihood.
   Discovery's `Pulsar` is a fork of it. **The existence proof is in
   production.** A timing package that emits that record, with the engine
   attached, *is* the pulsar.

So the design is one sentence: **the timing side emits `(PulsarData,
TimingEngine)` as one object built from one freeze, and Enterprise, Discovery
and nltiming consume it; nobody downstream constructs a pulsar ever again.**
vela-jax is the first package small enough to do it right; MetaPulsar already
half-does it (its duck surface + `to_feather` + `timing_engine`) and converges
on the same contract for composites; JUG and pyvela can adopt the same
emission later without any consumer noticing.

## I.3 The design in one page

```
                      par + tim
                          │
              timing package: PINT ──┴── tempo2          (Addendum A)
                          │  freeze once; NEVER REORDER              §5.5
                          ▼
        ┌───────────────────────────────────────────┐
        │ Engine        r(θ), residual_delta_jax, J │   §3–§9
        │ PerturbativeEngine  (two-dtype dual)      │   §11
        └───────────────┬───────────────────────────┘
                        │  every array below is frozen state or the
                        │  reference evaluation — THE ONE-SOURCE RULE  B.1
                        ▼
        ┌───────────────────────────────────────────┐
        │ PulsarData    the named, frozen array     │   Addendum B
        │               array record (= the feather │
        │               schema, in memory)          │
        └───────────────┬───────────────────────────┘
                        ▼
          TimingPulsar = PulsarData ∘ Engine  (composition)   B.6
                        │
        ┌───────────────┼──────────────────┬─────────────────┐
        ▼               ▼                  ▼                 ▼
   Enterprise      Discovery           nltiming          feather file
   (reads attrs;   (reads attrs or    (protocols it      (PulsarData
   FeatherPulsar-  the feather)       already defines;   serialized;
   shaped)                            its gate passes    schema v1)
                                      for free)
```

What dies relative to v1: the second PINT pass in `pulsar.py`; the empty
`dmx`; `derivative_method` accepted-and-ignored;
the 10 µs PINT–tempo2 cushion; the fp32 reference channel in working precision;
nltiming's `feature_binar_plus_models.md` plan to grow timing kernels inside
nltiming (vela-jax's perturbative engine *is* that proposal, in the right
package).

What is refused, still: multi-PTA combination (MetaPulsar's job), a
frozen-only/engine-less pulsar (nltiming `LinearTimingEngine`'s job), clocks
or ephemerides in the JAX graph, noise/GP/prior/sampler code, and growing
`TimingPulsar` toward the parts of `BasePulsar` nothing reads (`filter_data`,
`set_flags`, pickling hooks, deflate/inflate).

---

# Part II — Normative specification

Terms: **MUST/MUST NOT/SHOULD/MAY** as in RFC 2119. **timing package** =
PINT or tempo2: which package reads the par/tim (clock corrections, TT→TDB,
solar-system ephemeris, pulse numbers). `Engine.timing_package` names it
(`"pint"` or `"tempo2"`). A combined MetaPulsar feather uses
`"composite"`: not one timing package — this is independent of the
combination strategy (`per_pta` / `shared`). **build time** = code that
runs at engine construction, outside the JAX graph, in NumPy/longdouble.
**trace** = code inside `jax.jit`/`jacfwd` — the word
is kept to the JAX rules; prose says *delay chain* or *the JAX residual*.
**row** = one TOA or the TZR pseudo-TOA. `N` = number of TOAs, `R = N + 1`
rows. Citations of the form "what Enterprise actually reads" refer to the
measured inventories of §III.2.

Glossary, in the nouns Vela.jl and PINT already use — the distinctions that
matter most are the three arrival times:

- **TOA** — one pulse arrival (Vela `TOA`); the TZR TOA is one extra row,
  appended after the timing package's.
- **site arrival (SAT)** — the topocentric arrival time; Enterprise `stoas`.
- **TDB / `tdbld`** — the clock-corrected site arrival on the TDB scale
  (PINT column; Vela `TOA.value`).
- **barycentric arrival** — TDB minus the delays *before the binary* (solar
  system, solar wind, DM/DMX) — PINT `get_barycentric_toas`; Enterprise
  `toas`. **Not** the fully corrected TOA.
- **corrected TOA** — TDB minus *all* delays (Vela `corrected_toa_value`);
  the time the phase model sees, close to the emission time for a binary.
- **timing residual** — the phase residual over the pulsar-frame
  `spin_frequency`; gauge-free seconds, no mean removed. Vela divides by
  `doppler_shifted_spin_frequency` instead; this engine follows PINT and
  tempo2 (§8, G2).
- **design matrix** — the fitter-sign linearisation `r ≈ r0 − M δ`; here
  `M = −J` exactly (R9.3).
- **component** — one delay or phase correction (Vela `Component`, PINT
  component); *stage* is this spec's word for its compiled form.

## 1. Scope

### 1.1 Goals

- **G1 — JAX residual.** `r(θ): ℝ^{n_par} → ℝ^{N}` in seconds, `jit`-able and
  `jacfwd`-able, built once per pulsar.
- **G2 — Vela identity, except the residual divisor.** At the same freeze
  and the same θ, `r(θ)` agrees with `Vela.form_residuals` within §12 budgets
  **up to the phase→time divisor**, which is deliberately PINT's and not
  Vela's (§8); `residual_delta(0) == 0` exactly. Vela's divisor is the
  doppler-shifted spin frequency, PINT's and tempo2's the pulsar-frame Taylor
  series, and the two differ by `r·(v/c)`. Since `r` is the data handed to
  Enterprise/Discovery likelihoods beside PINT-formed residuals, parity with
  the timing packages outranks the Vela identity on this one point.
- **G3 — Backend contract.** The engine structurally satisfies nltiming's
  engine protocols (§10) without importing nltiming.
- **G4 — Two timing packages, one physics.** PINT (default) or tempo2 (Addendum A)
  reads the files and freezes the arrays; the delay chain is identical either
  way. The *freeze* is not: clocks, planet positions, ecliptic obliquity, and
  solar-wind / time-ephemeris defaults differ unless the par/tim is on the
  mixed-engine surface A.6.1 requires. H7 bounds that freeze disagreement;
  A.2.3 is the geometry check on a tempo2 read. Runtime deps: `numpy`, `jax`,
  `pint-pulsar`, `astropy`; `libstempo` + tempo2 only behind the `tempo2`
  extra.
- **G5 — Delay-only read.** Noise lines stripped before either timing package;
  residuals independent of white/red-noise parameters.
- **G6 — Perturbative fp32 mode** over the §11 axis subset, validated against
  G1.
- **G7 — Small.** A target of ≤ 6 000 lines of package code including
  docstrings — a smell threshold, not a hill to die on: Addendum B, the
  feather writer and `VelaJaxTimingEngine` will press it. The operative budget is
  B.3.6's: the consumer surface trending past ~700 lines is the alarm.
- **G8 — The pulsar product.** `PulsarData`/`TimingPulsar` (Addendum B):
  every public array from the frozen state or the reference evaluation, in
  one row order — the timing package's — serializable to the feather file (schema v1), structurally
  satisfying nltiming's `PulsarData`/`TimingPulsar` protocols and readable by
  Enterprise exactly as a `FeatherPulsar` is.
- **G9 — One row order, and it is the timing package's.** The freeze never reorders
  TOAs (§5.5). No argsort, no `_isort`, no permutation, anywhere in this
  package.

### 1.2 Non-goals (refused at build, not skipped)

`BINARY BT/BTX/BT_piecewise/DDGR/T2` (T2 only if PINT itself has not resolved
it), wideband TOAs and DM residuals, `Glitch`, `ChromaticCM/CMX`,
`SimpleExponentialDip`, `FDJumpDM`, `DispersionJump`, `SolarWindDispersionX`,
`WaveX/DMWaveX/CMWaveX`, `PL*NoiseGP`, `FDJUMPLOG N`, overlapping
(non-exclusive) DMX windows, a fitted JUMP selecting no TOA, DDK without
astrometry, DDK with H3/STIGMA, DDK with `K96 N`. Each refusal raises
`UnsupportedModelError` naming the offending component and the supported set.

`BINARY DDR` is supported, but **only with PINT as the timing package**:
tempo2 implements no DDR model, so `read_tempo2.load` refuses by name after
the common PINT parse and *before* a libstempo pulsar is constructed. Letting
it through would end either in an opaque parser failure or, worse, in a
silently different binary model. DDR's own build-time refusals — an unknown
`DDRPBDOT`, FBX with a kinematic or Shklovskii `Pbdot`, `DDRGEO Y` without
`DDRKINE Y` in the PB chart, geometry or kinematics without `PX > 0`,
`DDRGEO Y` without `KOM`, a free `TGEO`, a non-default galaxy constant, a
non-zero `EDOT`/`EPS1DOT`/`EPS2DOT`/`DR`/`DTH` — are all build errors too
(§7.10b).

Refused **by value**, not by component, under R5.3b: `SWM 1`/`SWM 2` (PINT's
You et al. 2007 solar wind; Vela has only the spherical model), wideband TOAs
(the tim carrying DM measurements — the freeze reads the narrowband columns,
so building one would return narrowband residuals for a wideband data set
without saying so), and any other parameter set to a non-zero value that no
stage evaluates.

Deliberate relaxation (implemented, kept): `CORRECT_TROPOSPHERE Y` is
**accepted with a warning** on both timing packages and not modelled — the delay is
~10 ns and near-constant, it cancels out of every residual difference, Vela
does not model it either, and refusing would block most EPTA/IPTA files.

Explicitly out of scope forever: priors, likelihoods, kernels, GLS, ECORR,
samplers, fitters, par writers, GUI, JAX clocks/SPK/BCLT, multi-pulsar vmap,
analytic derivative modules, multi-PTA combination, and every `BasePulsar`
facility the consumers never read (B.3.6).

## 2. Design authority

| Topic | Authority | Location |
|---|---|---|
| Component order | pyvela | `pyvela/model.py::pint_components_to_vela` |
| Correction state and helpers | Vela | `src/toa/toa.jl` |
| Residual, TZR | Vela | `src/residuals/residuals.jl` |
| SolarSystem | Vela | `src/model/solarsystem.jl` |
| Dispersion, DMX | Vela | `src/model/dispersion.jl`, `component.jl` |
| Solar wind | Vela | `src/model/solarwind.jl` |
| FD, FDJUMP | Vela | `src/model/frequency_dependent.jl` |
| Orbit, Kepler (Mikkola) | Vela | `src/model/binary/orbit.jl` |
| DD family, inverse timing formula | Vela | `binary_dd_base.jl`, `binary_dd.jl`, `binary_ddh.jl`, `binary_dds.jl` |
| ELL1 family | Vela | `binary_ell1_base.jl`, `binary_ell1.jl`, `binary_ell1h.jl`, `binary_ell1k.jl` |
| DDK | Vela | `binary_ddk.jl` |
| DDR | Vela | `binary_ddr.jl` (`feat/ddr-model`, fcf7134) |
| DDR schema, and an independent numerical oracle | PINT | `pint/models/binary_ddr.py`, `stand_alone_psr_binaries/DDR_model.py` |
| Spindown, PhaseOffset, PhaseJump | Vela | `spindown.jl`, `phase_offset.jl`, `jump.jl` |
| Parameter units, epochs, F0 split, `fix_params` | pyvela | `pyvela/parameters.py`, `pyvela/model.py` |
| Obliquity of the ecliptic per `ECL` realisation | PINT | `pint/data/runtime/ecliptic.dat`, read at build — **not** copied |
| PINT freeze columns | pyvela | `pyvela/toas.py` |
| Build-time longdouble reduction, orbit-count reduction | JUG (algorithm only, re-implemented) | `jug/delays/barycentric_jax.py` docstring, `jug/utils/orbit_reduction.py` |
| tempo2 columns, TCB rules, binary-input conventions | tempo2 (identities cited, code local) | `calculate_bclt.C`, `ELL1model.C`, `formBats.C` — Addendum A |
| Backend/engine protocols, gauge, chart facts, `nonlinear_params` vocabulary | nltiming | `nltiming/protocols.py`, `engine_config.py` |
| Pulsar attribute names, feather layout | Enterprise `FeatherPulsar` + what Enterprise/Discovery actually read | `enterprise/pulsar.py`, `discovery/pulsar.py`; §III.2 |

Forbidden as physics sources: `jug/delays/combined.py`,
`jug/fitting/derivatives_*.py`, JUG tempo2 graphs, tempo2 C (beyond the cited
identities of Addendum A). JUG's parity harness and fixtures MAY be used as
test oracles for PINT residual comparison.

## 3. Architecture

```
par + tim
   │  strip noise lines (in memory)                       §5.2
   │  tempo2 only: normalise UNITS to TDB first      A.3
   ▼
timing package reads: PINT get_model_and_toas(planets=True, ...)    §5.1
            or tempo2 via libstempo (sandboxed)           A.2
   │  freeze once (numpy, longdouble where §4 says so)
   │  never reorder: row i is the timing package's row i             §5.5
   ▼
FrozenTOAs (R rows) + ParamLayout (θ★ exact, units, live/frozen) + Chain
   │
   ▼
trace:  params = layout.build(δ)                          §6
        corr   = Correction.initial(R)
        for stage in chain: corr = stage(frozen, corr, params)   §7
        r      = form_residuals(frozen, corr)             §8
   │
   ▼
Engine (§9)  ──►  PulsarData / TimingPulsar (Addendum B)  ──►  feather (B.5)
```

Rules:

- **R3.1** The trace MUST NOT call PINT, astropy, erfa, or NumPy longdouble.
  A test monkeypatches PINT to raise during a jitted call (§12 T10).
- **R3.2** Stages are a static Python tuple fixed at build. The pipeline is an
  unrolled Python loop at trace time (not `lax.scan`).
- **R3.3** All rows are processed as arrays of shape `(R,)`/`(R,3)` (three-
  vectors as 3-tuples of `(R,)` components, mirroring Vela's `NTuple{3,GQ}`).
  No Python loop over TOAs in the trace. No `vmap` over pulsars.
- **R3.4** Per-row branching (barycentered TZR, masks) uses frozen boolean or
  index arrays and `where`, never data-dependent Python `if`. Every Vela `if`
  guarding a singular expression becomes *substitute a safe value, then
  select* — both branches of a traced `where` are evaluated, and a NaN in the
  unselected branch poisons the gradient. Stated once (`binary/orbit.py`),
  applied throughout.
- **R3.5** The TZR pseudo-TOA is row `R−1`, with `is_tzr[R−1] = True`; it is
  appended after the timing package's `N` rows and is not one of them.
- **R3.6 (the `numerics` module).** Every physics module computes with `+ - * /`
  and the `numerics` module's functions (`sin`, `cos`, `log`, `exp`, `sqrt`,
  `arccos`, `arctan2`, `kepler`, `select`/`where`, `clip`, the `Vec3`
  helpers), never with `jnp` directly. For the float64 engine the functions
  *are* `jnp`; the indirection costs nothing and is what lets the §11 dual
  reuse the same physics with no second copy.
- **R3.7 (finite-frequency guard).** A TOA — usually the TZR pseudo-TOA with
  no `TZRFRQ` — can have PINT's infinite frequency. `dm/ν²` then gives the
  right value but an `inf · 0` derivative, which NaNs every Jacobian column.
  The frozen frequency is therefore finite everywhere and dispersive terms
  are selected off through a frozen `finite_freq` mask
  (`correction.inverse_freq_sqr`).

## 4. Precision

### 4.1 The problem, quantified

Worst case, as required for this argument: `F0 ~ 800 Hz`, `Tspan ~ 100 yr`
(`Δt ~ 3.16×10⁹ s`), `D ~ 1000 s`. A typical PTA MSP span (~3 yr,
`Δt ~ 10⁸ s`) is ~30× milder on the `Δt` rows and is given in parentheses.

Convention: the "float64 ulp" column is `x · 2⁻⁵²`; the rounding error of a
single operation is at most half of it. Effects are quoted as that bound
divided by `F0`, so they are upper bounds, not typical values.

| Quantity | Magnitude | float64 ulp | Effect on residual |
|---|---|---|---|
| `Δt = t − PEPOCH` (frozen `tau`) | 3.16×10⁹ s (10⁸ s) | 7×10⁻⁷ s (1.5×10⁻⁸ s) | 700 ns (20 ns) if used absolutely in the phase; **it is not** — see the `tau` row |
| `F0·Δt` | 2.5×10¹² turns (3×10¹⁰) | 5.6×10⁻⁴ turns (7×10⁻⁶) | 700 ns (20 ns) — the quantity §4.3 removes |
| `F0·D` (`phi_ref`; `c₁·ξ`, `|ξ| ≤ D`) | ≤ 8×10⁵ turns | 1.8×10⁻¹⁰ turns | 0.22 ps — the quantity the trace forms |
| `tau` cast to float64 | 3.16×10⁹ s | 7×10⁻⁷ s | enters only secular rates through `corrected_time`: `F1·t·δτ ~ 10⁻¹⁵ × 3×10⁹ × 7×10⁻⁷ ≈ 2×10⁻¹²` turns, `δF0·δτ`, `A1DOT·δτ`, `PBDOT`, PM — all ≪ 1 fs |
| `δF0·Δt`, δF0 ~ 10⁻¹² Hz | 3×10⁻³ turns (10⁻⁴) | — | negligible (`Δt` error × δF0) |
| `F1·Δt²/2`, F1 ~ 10⁻¹⁵ | 5×10³ turns (5) | — | `Δt` error × `F1·Δt` ≪ 1 fs |
| `2π(t−T0)/PB` unreduced | 6×10⁴ rad at 10⁴ orbits | 10⁻¹¹ rad | ~0.4 ps sawtooth × A1 (measured) |
| `PM·(t−POSEPOCH)` | 10⁻⁵ rad | — | negligible |

`ulp(F0·Δt)/F0 = Δt · 2⁻⁵²` is independent of `F0`: raising `F0` does not
save the unreduced product. `F0·D` barely moves between the typical and the
worst case (0.2 → 0.22 ps) because `ulp(F0·D)/F0 ≈ D · 2⁻⁵²`.

The two `Δt` rows are why §4.3–§4.4 exist: `F0·Δt` is formed once, in numpy
longdouble, and never enters the trace. The quantity the JAX graph *does*
form is `F0·D` (as `phi_ref`, and as `c₁ξ` with `|ξ| ≤ D`), which float64
carries to ~0.2 ps at the worst case. The `tau` the trace holds in float64
reaches only secular terms, whose rates are small enough that a 0.7 µs error
in `Δt` is invisible.

**The longdouble floor is platform-dependent.** The build-time `F0·Δt − N`
reduction is done in `numpy.longdouble`: 113-bit on aarch64 (ulp of
2.5×10¹² turns ≈ 10⁻²¹ turns, nothing) and 80-bit x87 on x86-64 (ulp
≈ 2.5×10¹²·2⁻⁶³ ≈ 2.7×10⁻⁷ turns ≈ **0.34 ns** at 800 Hz × 100 yr; ~10 ps at
a 3 yr span). So at the worst case the engine's floor is ~0.3 ns on x86-64,
set by the longdouble reduction, and ~0.2 ps on aarch64, set by the trace.
`require_longdouble()` refuses a platform where `longdouble is float64`,
where the floor would be the 700 ns of the first row. `docs/PARITY.md`
records the measured platform dependence on the fixtures.

Conclusion: only the spin phase needs a build-time reduction; the binary
phase benefits from a ps-level refinement; everything the trace sees is
float64-safe at the worst-case numbers, not merely the typical ones.

### 4.2 Build-time epoch reduction (R4.2)

For every epoch parameter `E ∈ {PEPOCH, POSEPOCH, DMEPOCH, SWEPOCH, T0,
TASC}` present in the model, and for the TZR row:

```python
# build time, numpy longdouble
tdb_ld  = toas.table["tdbld"].value.astype(np.longdouble)          # MJD
tau_ld  = (tdb_ld - np.longdouble(PEPOCH_mjd)) * np.longdouble(86400)
tau     = tau_ld.astype(np.float64)                                 # s since PEPOCH
```

Other epochs are stored as float64 seconds relative to PEPOCH exactly as
pyvela does: `e_rel = float64(longdouble(E_mjd − PEPOCH_mjd)·86400)`. Inside
the trace, `t − E = t_c − e_rel` where `t_c = tau − delay`. Live epochs (`T0`,
`TASC` MAY be free; `PEPOCH`, `POSEPOCH`, `DMEPOCH` MUST be frozen — refuse
otherwise) enter as `e_rel_live = e_rel + δE·86400`.

### 4.3 Reference phase constant (R4.3)

```python
# build time, numpy longdouble, once; φ = the reference spin Taylor series
N        = (pulse_number - delta_pulse_number).astype(np.longdouble)
tau_tzr  = longdouble seconds of get_TZR_toa(toas) minus PEPOCH
phi_ref  = (φ(tau_ld) - N - φ(tau_tzr)).astype(np.float64)          # (N,)
phi_ref  = np.append(phi_ref, 0.0)                                  # TZR row: 0
```

`phi_ref` MUST be built from `tdbld`, never from `toas.get_mjds()` (float64).
`F0` (and every `F_k`) MUST come from the parsed par token via
`Decimal`/`str`, never `float(model.F0.value)`. `phi_ref` contains no delay
physics — only reference spin coefficients, raw TDB times and pulse numbers —
so `r(θ★)` and `r(θ★+δ)` see identical physics in the trace. Its magnitude is
`~F0·D ≤ 8×10⁵` turns at the §4.1 worst case, which float64 carries to
~0.22 ps.

### 4.4 Spin phase in the trace (R4.4) — the full reference series

The v0.2 draft kept only `F0★` at build time; the implementation went further
and that is now the norm. The build-time reduction evaluates the *whole* reference spin
series at each undelayed TDB time and hands the trace its Taylor re-expansion
in the delay:

```python
# build time (precision.spin_coefficients): c_m = φ^(m)(tau_ld) in longdouble → float64
# trace, with ξ = −corr.delay, t_c = tau − corr.delay:
psi = (frozen.phi_ref
       + Σ_{m≥1} c_m ξ^m / m!                 # reference series, re-expanded
       + Σ_k   δF_k t_c^(k+1) / (k+1)!        # live spin deltas
       + jumps − PHOFF)
spin_frequency = c_1 + Σ_k δF_k t_c^k / k!    # instantaneous, per row
```

Nothing large is ever multiplied inside the trace: `ξ` is at most a few
thousand seconds and every `c_m` fits float64 comfortably. This makes the
spin phase exact to well under a picosecond for arbitrarily large `F1` (the
`F0★`-only form left 0.87 ns on `sim_dd`, whose `F1 = −4.2×10⁻⁸ Hz/s`; the
full series leaves 0.12 ns, which is Vela's own float64 floor).
`residual_delta(0)` is exactly zero because `r(θ★)` and `r(θ★+δ)` are the
same graph.

### 4.5 Binary time argument and orbit-count reduction (R4.5)

At build, for the binary epoch `E ∈ {T0, TASC}` and reference period
`P★ = PB★·86400` (or `1/FB0★`):

```python
dt_ld   = tau_ld - np.longdouble(e_rel)          # (t_raw − E)★, longdouble
n_orb   = np.round(dt_ld / P_ld)                 # exact integers, frozen
dt_red  = (dt_ld - n_orb * P_ld).astype(np.float64)   # |dt_red| ≤ P/2
```

In the trace, with `Δt = t_c − e_rel_live` (full, float64; used for every
secular term) and `Δt_red = dt_red − delay − δE·86400`:

```
PB mode :  Φ = 2π [ Δt_red/PB_live − n_orb·δPB·86400/PB_live − ½ PBDOT (Δt/PB)² ]
FB mode :  Φ = 2π [ FB0_live·Δt_red + n_orb·(FB0_live·P★ − 1) + ∫ FB1… on Δt ]
```

The integer `n` drops out of all trig; the `n·δPB` term is *physics* (how δPB
moves late-time orbital phase) and float64-safe because `n` is exact and
`δPB` small. If T0/TASC is live, `Δt_red` shifts by `−δE·86400` and nothing
else changes; no re-reduction inside the trace is ever needed.

**R4.5 for DDR.** DDR's mean longitude is reduced by the same two identities,
with `E = TASC` and `Δt_red = dt_red − delay − δTASC`. A frozen `TASC` makes
`δTASC` zero, so omitting that term is silent until `TASC` is live — which
§12's T13 deliberately perturbs. Vela forms the absolute orbit count in
float64 and reduces it inside its solver instead, which loses phase before the
reduction; this is a precision improvement over Vela and is gated at
`n_orb = 10⁴`.

**R4.5b (the unwrapped true anomaly).** The reduction removes the orbit count
from the eccentric/true anomaly, but `ω = OM + (OMDOT/n̂)·v` is *secular* in
`v`, not periodic: the count MUST be restored (`v + 2π·n_orb`) before the
OMDOT advance. Missing this was a **2.7 ms** error on `J0955-6150`; with it,
2.4 ps.

**R4.5b for DDR.** The same rule, in DDR's coordinates: `λ` as returned by the
reduced helpers is periodic, and only `precession_delta` reads the restored
`λ_secular = λ + 2π·n_orb`. Omitting the restoration at `n_orb = 10⁴` moves
the precession angle by `κ·2π·n_orb ≈ 0.129 rad` and the Roemer delay by
≈ 0.64 s on the §12 DDR anchor model.

### 4.6 TZR

The TZR row goes through the same stages with `phi_ref = 0`, pulse number
irrelevant, `is_tzr = True`, and `ssb_obs_pos = 0` if `TZRSITE == "@"`.
Stages Vela zeroes for TZR (`PhaseOffset`, `PhaseJump`,
`FrequencyDependentJump`, `DispersionPiecewise`) apply
`where(~frozen.is_tzr, value, 0)`.

### 4.7 Precision-critical parameters (R4.7)

`precision_critical_params()` returns `{F0, PEPOCH, POSEPOCH, DMEPOCH, T0,
TASC, TGEO, PB, FB0} ∩ param_names`. `TGEO` is listed although a valid DDR
model cannot free it (the freeze refuses that): this set is what the exact
metadata publishes as "unsafe as an absolute float64 coordinate", and every
epoch belongs in it. Callers MUST keep `reference_theta_exact()`
strings and apply deltas additively; inside the engine every parameter enters
as `θ★ + δ` with `θ★` folded into frozen constants, so there is no
`(θ+δ) − θ` cancellation anywhere.

### 4.8 x64

`vela_jax.config.require_x64()` is called by `Engine` construction and raises
if `jax_enable_x64` is off; it does not silently enable it. The perturbative
engine (§11) does not require x64.

## 5. Freeze: timing package → frozen arrays

### 5.1 Ingest (PINT)

```python
def load_pint(par, tim, **kw):
    par_text = strip_noise_lines(Path(par).read_text())     # §5.2
    model, toas = pint.models.get_model_and_toas(
        par_tmp, tim, planets=True, allow_T2=True, allow_tcb=True,
        add_tzr_to_model=True, **kw)
    prepare_model(model, toas)                              # §5.3
    toas.compute_pulse_numbers(model)
    return model, toas
```

`Engine.from_pint(model, toas)` accepts a caller-built pair, runs
`prepare_model` (idempotent) and checks the same invariants (`toas.planets`,
pulse numbers, `tdbld`, ephemeris columns), raising `FreezeError` otherwise.
The tempo2 is Addendum A.

### 5.2 Noise strip (R5.2)

Lines whose first token (case-insensitive) is in
`{EFAC, EQUAD, ECORR, T2EFAC, T2EQUAD, TNEF, TNEQ, TNECORR, TNGLOBALEF,
TNGLOBALEQ, TNRED*, TNDM*, TNCHROM*, TNSW*, RNAMP, RNIDX, DMEFAC, DMEQUAD,
DMJUMP, PLRED*, PLDM*, PLCHROM*, CHI2*, TRES, DMRES}` are removed. The
classifier lives in `vela_jax.freeze.NOISE_KEYS` and is a superset of
MetaPulsar's `parfile_lines.is_noise_line`; MetaPulsar's wrapper MUST produce
a byte-identical strip for the same input (tested in MetaPulsar). Strip never
rewrites the caller's file.

### 5.3 `prepare_model` (pyvela `fix_params`, delay part)

1. `PEPOCH` required.
2. Missing `*EPOCH` ← `PEPOCH`.
3. `PhaseOffset` added if absent; `PHOFF` free (nltiming requires the gauge
   column; §10).
4. `H4` → `STIGMA = H4/H3` if `STIGMA` unset; `H4` frozen.
5. Zeroable if unset: `M2, SINI, PBDOT, XPBDOT, A1DOT, EPS1DOT, EPS2DOT, H3,
   STIGMA, LNEDOT, EDOT, OMDOT, GAMMA, DR, DTH`.
6. Frozen JUMP selecting no TOA: dropped with a warning. Fitted JUMP
   selecting no TOA: `UnsupportedModelError`.
7. `BINARY T2` unresolved by PINT: `UnsupportedModelError`.

### 5.3b Parameter accountability (R5.3b)

**Every parameter the par sets must be consumed, known-inert, or refused by
name.** After the chain is built, each parameter of the prepared model MUST
fall into exactly one of:

- **consumed** — some stage reads it (`Chain.consumed`), or the freeze does
  (`PEPOCH`, `TZR*`, `DMXR1_*`/`DMXR2_*`, `BINARY`, `PLANET_SHAPIRO`, …);
- **inert** — on an explicit list, each entry with a stated reason: identity
  and bookkeeping (`PSR`, `NTOA`, `START`, `FINISH`, `DMDATA`), read-side
  settings the timing package already applied before the freeze (`EPHEM`,
  `CLOCK`, `UNITS`, `TIMEEPH`, `T2CMETHOD`, `DILATEFREQ`), quantities that
  are not a delay (`RM`, the DispersionDMX info line `DMX` — the delay is
  `DMX_NNNN` inside `DMXR1_`/`DMXR2_` windows), and parameters only
  meaningful alongside a refused component;
- **inert at its value** — a numeric parameter sitting at zero. Every one of
  these enters its delay additively, so zero is not a lie. A bool or string
  is not an amplitude (`float(False) == 0.0` must not skip the offender check);
- **pinned** — inert only at one value, refused at any other (`SWM 0`).
  Ingest conventions (`T2CMETHOD`, `TIMEEPH`) are *not* pinned: the host
  already applied them, so they sit on the inert list;
- otherwise **refused**, by name, with the supported set, exactly as an
  unsupported component is.

The rule exists because a **frozen** parameter that PINT applies and no stage
here consumes is otherwise dropped in silence, and a fixture that does not
set the parameter looks identical whether the engine reads it or not.
`sim_dd` sets `A0`/`B0` at zero (the zero-skip accepts them); `sim_dmx` sets
`DMX 0.0`; no fixture sets a non-zero unimplemented term, and every one that
sets `ECL` sets `IERS2010` — the value Vela hard-codes. Only an
accountability rule can close the class, which is why it is a rule and not a
parity test. Current behaviour:

| par edit | this engine |
|---|---|
| `A0` / `B0` non-zero | `UnsupportedModelError` by name |
| `SWM 1` | `UnsupportedModelError` (`SWM 0` is pinned) |
| frozen non-zero bare `DMX` (`0.5`, `14`, …) | accepted; PINT and this engine residuals bitwise identical |
| fitted bare `DMX` | `UnsupportedModelError` (unconsumed free parameter) |
| `ECL IERS1992` / `IAU1976` / `IERS2003` on `sim_sw` | residual shift matches PINT to < 0.1% (1326.7 ns / 9286.7 ns / 22.1 ns) |

### 5.4 `FrozenTOAs` (R5.4)

As implemented (the archive's draft shape is superseded):

```python
class FrozenTOAs(NamedTuple):     # R = N + 1 rows; row R-1 is the TZR
    tau: Array            # float64 s since PEPOCH (float64 cast of longdouble)
    phi_ref: Array        # §4.3
    spin_coeffs: tuple    # §4.4 reference-series derivatives c_m, per row
    freq_hz: Array        # topocentric observing frequency, Hz, finite
    finite_freq: Array    # bool mask — R3.7
    is_tzr: Array         # bool
    is_bary: Array        # bool: Vela is_barycentered — ssb_obs_pos == 0
    ssb_obs_pos: Vec3     # light-seconds
    ssb_obs_vel: Vec3     # light-seconds / s
    obs_sun_pos: Vec3     # light-seconds
    planet_pos: dict      # jupiter, saturn, venus, uranus, neptune (if PLANET_SHAPIRO)
    dmx_index: Array | None      # int32 exclusive; 0 = none
    jump_index: Array | None     # int32 exclusive, or
    jump_masks: tuple            # bool masks when non-exclusive (pyvela PhaseJump)
    fdjump_masks: tuple
    fdjump_exp: tuple            # static
    binary: FrozenBinary | None  # dt_red, n_orb (exact ints), period_ref_s — §4.5
```

Units and columns follow `pyvela/toas.py`: positions in light-seconds,
velocities in ls/s, MHz → Hz, pulse number `pulse_number −
delta_pulse_number`. Masks come from PINT's `select_toa_mask` (pyvela
`read_mask`), converted to exclusive index arrays when `is_exclusive_mask`
holds (DMX MUST be exclusive).

### 5.5 Row order: the timing package's, unchanged (R5.5)

**This package never reorders TOAs.** Row `i` of every frozen array, every
residual, every Jacobian row, the pulsar record and the feather file is row `i`
of the timing package's TOA table — PINT's or tempo2's, whichever read the files.

- **R5.5.1** `freeze()` MUST NOT sort, argsort, filter or otherwise permute
  the timing package's rows. The TZR pseudo-TOA is appended as row `R−1` (R3.5) and is
  not one of them.
- **R5.5.2** No permutation is published. There is no `data_order`, no
  `toa_index`, no `_isort`, and nothing for a consumer to apply.
- **R5.5.3** A consumer that wants a different order produces it *on read*.
  Enterprise already does (`_isort`, at its property layer, on the barycentric
  arrival); Discovery can. That is their business and it is reversible. A
  timing package that permutes the rows its residual, its Jacobian and its
  design matrix are all built from is not doing them a favour.

**Why this is a rule and not a preference.** Version 2.0 of this document
briefly required the opposite — a stable argsort on `tdbld` at the freeze — on
two premises, both of which are false:

1. *"Discovery assumes its input is already sorted."* It does not. Discovery's
   `quantize` (`signals.py:51`) takes its own `argsort` internally and writes
   bin membership back into the caller's row order. Enterprise's
   `create_quantization_matrix` (`signals/utils.py:1149`) does the same. ECORR
   groups TOAs by *value*, not by adjacency; neither consumer needs sorted
   input, and MetaPulsar has a test that builds ECORR on a deliberately
   shuffled pulsar to keep it that way.
2. *"Enterprise sorts lazily through `_isort`, so the orders must be
   reconciled."* Enterprise sorting at its property layer is exactly the
   design that needs no help: it is applied on read, it is Enterprise's own,
   and it does not reach back into anyone's engine.

The one order-sensitive path in either consumer — Enterprise's
`quant2ind(as_slice=True)` — checks contiguity itself and falls back to index
arrays, so it is correct on unsorted rows too.

What the freeze sort actually bought was a permutation protocol across a
package boundary. The engine published one row order and a composite's leg
published another, both from the same freeze; the leg adapter carried the
inverse permutation on the hot path; and a consumer who compared
`engine.residuals()` against a composite's rows *without* going through that
adapter got a plausible, wrong likelihood — which is precisely the failure the
sort was introduced to prevent. Worse, on a release `.tim` that is already in
time order the permutation is the identity, so a missing or wrong inverse
passes every end-to-end test and fails only on the files where order matters.

A sort at the freeze is not a normalisation. It is a silent, order-dependent
transformation of the thing every downstream number is indexed by, and its
absence is the contract.

## 6. Parameters

### 6.1 Names, units, order

- `param_names`: PINT names of the engine's **free** parameters, in PINT
  `model.free_params` order, after `prepare_model`. Frozen PINT parameters
  are constants.
- `param_units[name] = str(model[name].units)` (`""` → `"1"`).
- A free PINT parameter no stage consumes → `UnsupportedModelError` (typical
  cause: a noise/GP parameter that survived a bad strip).

### 6.2 Internal units (Vela's `[T^n]` system)

| PINT unit | internal | factor |
|---|---|---|
| hourangle (RAJ) | rad | π/12 |
| deg (DECJ, ELONG, ELAT, OM, KIN, KOM) | rad | π/180 |
| mas/yr (PM*) | rad/s | via astropy |
| mas (PX) | 1/light-second | pyvela's GQ{-1} factor |
| pc/cm³ (DM, DMX_) | s·Hz² | `DMconst` from PINT |
| pc/cm³/yr^k (DMk) | s·Hz²/s^k | `DMconst`/yr^k |
| cm⁻³ (NE_SW) | as pyvela | pyvela `get_scale_factor` |
| d (PB, T0, TASC) | s | 86400 |
| ls (A1), s (JUMP, FD, GAMMA), Msun (M2) | s | 1 / `T_sun = 4.925490947e-6 s` |
| deg/yr (OMDOT) | rad/s | via astropy |
| Hz, Hz/s^k (Fk), Hz^k (FBk) | same | 1 |
| dimensionless (ECC, EPS*, SINI, STIGMA, H3 in s) | same | 1 |

The factor table is computed with astropy at build time using pyvela's
`get_scale_factor`/`get_unit_conversion_factor` logic (ported, credited), so
no factor is hand-typed except `T_sun` and `86400`.

### 6.3 `ParamLayout` and `Params` (R6.3)

```python
@dataclass(frozen=True)
class ParamLayout:
    names: tuple[str, ...]            # free, PINT order
    units: dict[str, str]
    scale: np.ndarray                 # PINT unit → internal, per name
    theta_exact: dict[str, str]       # decimal strings from the par tokens
    frozen_internal: dict[str, float]
    ref_internal: dict[str, float]
    families: dict[str, tuple[str, ...]]   # "F", "DM", "FB", "DMX_", "JUMP", "FD", "FDJUMP", "NE_SW"

    def build(self, delta, *, wrap=None) -> Params:
        """δ in PINT units, names order → internal-unit values.

        Live scalar: ref_internal + delta·scale.  Frozen: frozen_internal.
        Special: F0 → spin-delta family; T0/TASC → e_rel + dE; PB/FB0 → live
        value AND its delta (§4.5). Families packed in prefix order.
        ``wrap`` is the §11 dual-injection hook: (name, ref, step) → value.
        """
```

`Params` is a `NamedTuple` (pytree); unused fields are `None` so the trace
closes over nothing it does not read.

### 6.4 Reference θ

`reference_theta_exact()` returns `{name: decimal string}` for `param_names`,
taken from the par tokens as parsed (`Decimal(str(param.quantity.value))` is
*not* acceptable for MJDs; use the retained token text, or `np.longdouble`
printing for `MJDParameter`). `reference_theta()` is the float64 array of
those.

## 7. Components

Each component — Vela's `Component`; *stage* is this spec's word for its
compiled form — is a pure function `stage(frozen, corr, p) -> Correction`,
written against the `numerics` module (R3.6). Module docstrings cite the Vela file.
Snippets are normative for structure and formulae; names may differ.

### 7.1 `TOACorrection` state (`toa.jl`; here `Correction`)

```python
class Correction(NamedTuple):
    delay: Array           # (R,) s
    phase: Array           # (R,) turns — small by construction (§4.3)
    spin_frequency: Array  # (R,) Hz, 0 until Spindown
    doppler: Array         # (R,) dimensionless
    ssb_psr_pos: Vec3      # unit vector, 0 until SolarSystem

def corrected_time(frozen, corr):   # Vela corrected_toa_value — the corrected TOA
    return frozen.tau - corr.delay
def bary_freq(frozen, corr):        # Vela doppler_corrected_observing_frequency
    return frozen.freq_hz * (1.0 - corr.doppler)
def topo_spin_frequency(corr):      # Vela doppler_shifted_spin_frequency
    return corr.spin_frequency * (1.0 + corr.doppler)   # NOT the residual divisor (§8)
```

No `efac`/`equad2` fields — noise belongs to Discovery/Enterprise.

### 7.2 Component order (R7.2) — copy of `pint_components_to_vela`

| # | PINT component | stage | v1 |
|---|---|---|---|
| 1 | `AstrometryEcliptic` / `AstrometryEquatorial` (+`SolarSystemShapiro`) | `solar_system` | **optional** — a par with no astrometry (`pure_rotator`) builds, as in pyvela; every TOA is then barycentred |
| 2 | `SolarWindDispersion` (if `NE_SW` present, not frozen-zero) | `solar_wind` | required |
| 3 | `DispersionDM` | `dispersion_taylor` | required |
| 4 | `DispersionDMX` | `dispersion_piecewise` | required |
| 5 | `Binary*` (ELL1, ELL1H, ELL1k, DD, DDH, DDS, DDK, DDR) | `binary.<family>` | required if `BINARY` |
| 6 | `FD` | `frequency_dependent` | required |
| 7 | `FDJump` (`FDJUMPLOG Y` only) | `frequency_dependent_jump` | required |
| 8 | `Spindown` | `spindown` | required |
| 9 | `PhaseOffset` | `phase_offset` | required (always present after §5.3) |
| 10 | `PhaseJump` | `phase_jump` / `phase_jump_exclusive` | required |

`binary_conventions="tempo2"` does not reorder this table; it only selects
the ELL1 truncation of A.4. Components in §1.2's refusal list raise if present.

### 7.3 `solar_system` (`solarsystem.jl`)

Constants: `AU_LS = 499.00478383615643` and the masses in seconds as in Vela —
copied verbatim with the Vela line cited.

**The obliquity is not one of them (R7.3).** Vela hard-codes
`OBL = 0.4090926006005829` (84381.406″, IERS2010). A par's `ECL` keyword
selects the realisation its ELONG/ELAT were defined in, PINT honours it, and
`ECL IERS2003` (84381.4059″) is 0.1 mas away — **~100 ns RMS of Roemer delay**
on a real MSP, 65× the PINT–tempo2 clock floor, absorbed by the sampled sky
position as a bias of order a real `ELAT` uncertainty. EPTA and IPTA release
pars typically say `IERS2003`. So the obliquity is resolved at build time
from `ECL` against PINT's own table (read, not copied) and recorded as
`Engine.obliquity`; `OBL` remains the default for a par that says nothing.

On the tempo2 timing package it is tempo2's `ECLIPTIC_OBLIQUITY_VAL` instead,
whatever the par says (A.2.9): tempo2 has no `ECL` keyword and has already
rotated every ephemeris vector with its own constant, so the line-of-sight
rotation must match it or the composition is not the identity. Using two
different constants across `read_tempo2.ecliptic_to_icrs` and
`astrometry.ecliptic_to_equatorial` is a 0.1 mas frame twist from nothing
but two files disagreeing.

```python
def evaluate_proper_motion(long0, lat0, pm_long, pm_lat, dt):
    sa, ca = sin(long0), cos(long0);  sd, cd = sin(lat0), cos(lat0)
    x0   = (ca*cd, sa*cd, sd)
    xdot = (-sa*pm_long - ca*sd*pm_lat, ca*pm_long - sa*sd*pm_lat, cd*pm_lat)
    x1 = add3(x0, scale3(dt, xdot))
    return scale3(1/norm3(x1), x1)          # linear motion, renormalised (Vela)

def solar_system(frozen, corr, p, *, ecliptic, planet_shapiro):   # static flags
    dt   = corrected_time(frozen, corr) - p.POSEPOCH_rel
    Lhat = evaluate_proper_motion(p.long0, p.lat0, p.pm_long, p.pm_lat, dt)
    if ecliptic:                            # rotate the LINE OF SIGHT with OBL
        s, c = sin(OBL), cos(OBL)
        Lhat = (Lhat[0], c*Lhat[1] - s*Lhat[2], s*Lhat[1] + c*Lhat[2])
    R    = frozen.ssb_obs_pos
    LdR  = dot3(Lhat, R)
    delay = -LdR                                              # Roemer
    delay += 0.5 * p.PX * (dot3(R, R) - LdR*LdR)              # parallax (PX in 1/ls)
    delay += shapiro(M_SUN, frozen.obs_sun_pos, Lhat)
    if planet_shapiro:
        for name, M in PLANET_MASSES.items():
            delay += shapiro(M, frozen.planet_pos[name], Lhat)
    doppler = dot3(Lhat, frozen.ssb_obs_vel)
    skip = frozen.is_bary
    return corr.add_delay(where(skip, 0.0, delay),
                          where(skip, 0.0, doppler),
                          where3(skip, corr.ssb_psr_pos, Lhat))

def shapiro(M, rvec, Lhat):
    r = norm3(rvec)
    return -2.0 * M * log((r - dot3(Lhat, rvec)) / AU_LS)
```

Vela's `iszero(pm)` short-circuit is dropped: the formula reduces to `x0`
exactly.

### 7.4 `solar_wind` (`solarwind.jl`)

```python
def solar_wind(frozen, corr, p):
    rvec = frozen.obs_sun_pos;  r = norm3(rvec)
    cos_rho = -dot3(corr.ssb_psr_pos, rvec) / r
    rho = arccos(clip(cos_rho, -1.0, 1.0))
    t = corrected_time(frozen, corr) - p.SWEPOCH_rel
    ne_sw = taylor_horner(t, p.NE_SW)
    slope = ne_sw * AU_LS * AU_LS * rho / (r * sin(rho))
    nu2inv = inverse_freq_sqr(frozen, corr)          # R3.7 mask inside
    return corr.add_delay(where(frozen.is_bary, 0.0, slope * nu2inv))
```

Requires `ssb_psr_pos` set → the SolarSystem-before-SolarWind order rule is
enforced at build.

### 7.5 `dispersion_taylor`, `dispersion_piecewise` (`dispersion.jl`, `component.jl`)

```python
def dispersion_taylor(frozen, corr, p):
    t  = corrected_time(frozen, corr) - p.DMEPOCH_rel
    dm = taylor_horner(t, p.DM)                       # internal: s·Hz²
    return corr.add_delay(dm * inverse_freq_sqr(frozen, corr))

def dispersion_piecewise(frozen, corr, p):
    dmx = concat([zeros(1), p.DMX_])[frozen.dmx_index]     # index 0 → 0
    dmx = where(frozen.is_tzr, 0.0, dmx)
    return corr.add_delay(dmx * inverse_freq_sqr(frozen, corr))
```

`taylor_horner`/`taylor_horner_integral` are the `Vela.taylor_horner` twins
(Horner form, factorials folded).

### 7.6 `frequency_dependent`, `frequency_dependent_jump` (`frequency_dependent.jl`)

```python
NU_REF = 1e9  # Hz

def frequency_dependent(frozen, corr, p):
    lam = log(bary_freq(frozen, corr) / NU_REF)
    return corr.add_delay(sum(fd * lam**(k+1) for k, fd in enumerate(p.FD)))

def frequency_dependent_jump(frozen, corr, p):
    lam = log(bary_freq(frozen, corr) / NU_REF)
    delay = zeros_like(lam)
    for j, (fdj, expo) in enumerate(zip(p.FDJUMP, frozen.fdjump_exp)):  # static
        delay = delay + where(frozen.fdjump_masks[j], fdj * lam**expo, 0.0)
    return corr.add_delay(where(frozen.is_tzr, 0.0, delay))
```

Uses the *barycentric* frequency, as Vela. FD sits after the binary in
`STAGE_ORDER`: it is a residual/phase effect, not a shift of the time the
binary sees.

### 7.7 Binary — shared orbit code (`orbit.jl`)

```python
def mean_anomaly(fb, dt_full, dt_red, p, use_fbx):
    if use_fbx:
        lin = p.FB[0] * dt_red + fb.n_orb * (p.FB[0] * fb.period_ref_s - 1.0)
        return 2*pi * (lin + taylor_horner_integral(dt_full, p.FB[1:], order_offset=1))
    return 2*pi * (dt_red / p.PB - fb.n_orb * p.dPB / p.PB
                   - 0.5 * p.PBDOT * (dt_full / p.PB)**2)

def mean_motion(dt_full, p, use_fbx):
    return 2*pi * (taylor_horner(dt_full, p.FB) if use_fbx
                   else 1.0 / (p.PB + p.PBDOT * dt_full))
```

Mikkola, translated with the R3.4 substitute-then-select rule:

```python
def mikkola(l0, e0):
    trivial = (e0 == 0.0) | (l0 == 0.0)
    e = where(trivial, 0.5, e0);  l = where(trivial, 1.0, l0)
    sgn = sign(l); l = abs(l)
    ncyc = floor(l / (2*pi)); l = l - 2*pi*ncyc
    flag = l > pi;  l = where(flag, 2*pi - l, l)
    alpha = (1 - e) / (4*e + 0.5); alpha3 = alpha**3
    beta = (l/2) / (4*e + 0.5); beta2 = beta*beta
    root = sqrt(alpha3 + beta2)
    z = where(beta > 0, cbrt(beta + root), cbrt(beta - root))
    s = z - alpha / z; w = s - 0.078 * s**5 / (1 + e)
    E0 = l + e * (3*w - 4*w**3)
    su, cu = sin(E0), cos(E0); esu, ecu = e*su, e*cu
    fu = E0 - esu - l; f1 = 1 - ecu; f2 = esu; f3 = ecu; f4 = -esu
    u1 = -fu / f1
    u2 = -fu / (f1 + f2*u1/2)
    u3 = -fu / (f1 + f2*u2/2 + f3*u2*u2/6)
    u4 = -fu / (f1 + f2*u3/2 + f3*u3*u3/6 + f4*u3**3/24)
    xi = E0 + u4
    sol = where(flag, 2*pi - xi, xi)
    return where(trivial, l0, sgn * (sol + ncyc*2*pi))
```

`e` outside `[0,1)` is not clamped (Vela asserts; we let NaN propagate —
nltiming's chart layer keeps the domain, and `BinaryChartCapability.
supports_domain` reports it).

### 7.8 DD family (`binary_dd_base.jl` + `binary_dd.jl`, `binary_ddh.jl`, `binary_dds.jl`)

```python
def shapiro_params(family, p):                      # static dispatch at build
    if family == "DD":  return p.M2, p.SINI
    if family == "DDH": return p.H3 / p.STIGMA**3, 2*p.STIGMA / (1 + p.STIGMA**2)
    if family == "DDS": return p.M2, 1 - exp(-p.SHAPMAX)

def dd_state(frozen, corr, p, *, family, use_fbx, kopeikin=None):
    fb = frozen.binary
    dt_full = corrected_time(frozen, corr) - p.T0_rel_live
    dt_red  = fb.dt_red - corr.delay - p.dT0
    n = mean_motion(dt_full, p, use_fbx)
    l = mean_anomaly(fb, dt_full, dt_red, p, use_fbx)
    et = p.ECC + dt_full * p.EDOT; er = et*(1 + p.DR); ephi = et*(1 + p.DTH)
    u = kepler(l, et, mikkola);  sinu, cosu = sincos(u)
    eta = sqrt(1 - ephi**2); bphi = (1 - eta) / ephi
    v = 2*arctan2(bphi*sinu, 1 - bphi*cosu) + u + 2*pi*fb.n_orb   # R4.5b unwrap
    a1 = p.A1 + dt_full * p.A1DOT
    omega = p.OM + (p.OMDOT / n) * v
    m2, sini = shapiro_params(family, p)
    if kopeikin is not None:                          # DDK, §7.10
        dx, domega, dinc = kopeikin(frozen, corr, p, dt_full, a1)
        a1 += dx; omega += domega; sini = sin(p.KIN + dinc); m2 = p.M2
    so, co = sincos(omega)
    return DDState(a1*so, a1*eta*co, p.GAMMA, sinu, cosu, et, er, a1, n, m2, sini)

def dd_delay(s):
    RE   = s.alpha*(s.cosu - s.er) + (s.beta + s.gamma)*s.sinu
    REp  = -s.alpha*s.sinu + (s.beta + s.gamma)*s.cosu
    REp2 = -s.alpha*s.cosu - (s.beta + s.gamma)*s.sinu
    nhat = s.n / (1 - s.et*s.cosu)
    RE_inv = RE * (1 - nhat*REp + nhat*nhat*REp*REp + 0.5*nhat*nhat*RE*REp2
                   - 0.5*s.et*s.sinu/(1 - s.et*s.cosu)*nhat*nhat*RE*REp)
    S = -2*s.m2*log(1 - s.et*s.cosu
                    - (s.sini/s.a1)*(s.alpha*(s.cosu - s.er) + s.beta*s.sinu))
    return RE_inv + S, REp*nhat                    # delay, doppler
```

`ephi = 0` (circular DD) divides by zero in `bphi`; Vela has the same —
document, do not guard (use ELL1 for circular orbits).

### 7.9 ELL1 family (`binary_ell1_base.jl`, `binary_ell1.jl`, `binary_ell1h.jl`, `binary_ell1k.jl`)

```python
def ell1_state(frozen, corr, p, *, family, use_fbx, ell1_t2):
    fb = frozen.binary
    dt_full = corrected_time(frozen, corr) - p.TASC_rel_live
    dt_red  = fb.dt_red - corr.delay - p.dTASC
    a1 = p.A1 + dt_full * p.A1DOT
    if family == "ELL1k":
        s, c = sincos(p.OMDOT*dt_full); g = 1 + p.LNEDOT*dt_full
        eps1 = g*(p.EPS1*c + p.EPS2*s); eps2 = g*(p.EPS2*c - p.EPS1*s)
    else:
        eps1 = p.EPS1 + dt_full*p.EPS1DOT; eps2 = p.EPS2 + dt_full*p.EPS2DOT
    Phi = mean_anomaly(fb, dt_full, dt_red, p, use_fbx)
    trig = tuple(sincos(k*Phi) for k in (1, 2, 3, 4))
    n = mean_motion(dt_full, p, use_fbx)
    m2, sini = shapiro_params_ell1(family, p)   # ELL1/ELL1k: M2,SINI; ELL1H: H3,STIGMA map
    return ELL1State(trig, n, a1, eps1, eps2, m2, sini)

def ell1_delay(s):
    R = roemer(s); Rp = d_roemer_dPhi(s); Rp2 = d2_roemer_dPhi2(s); nhat = s.n
    R_inv = R * (1 - nhat*Rp + nhat*nhat*Rp*Rp + 0.5*nhat*nhat*R*Rp2)
    return R_inv + shapiro_ell1(s), -Rp*nhat
```

`roemer`, `d_roemer_dPhi`, `d2_roemer_dPhi2` are the three polynomials of
`binary_ell1_base.jl` transcribed term by term (coefficients copied, never
re-derived). `ELL1k` subtracts `1.5·a1·eps1` from the Roemer delay. `ELL1H`
subtracts the `a0 + b1 sinΦ + a2 cos2Φ` harmonics from the Shapiro delay as
`binary_ell1h.jl`. `ell1_t2=True` (A.4) swaps the three polynomials for
tempo2's `ELL1model.C` truncation — first order in the Laplace–Lagrange
parameters, no harmonics in either derivative; the inverse-timing
combination, ELL1k's term and ELL1H's harmonic subtraction unchanged.

### 7.10 DDK (`binary_ddk.jl`)

```python
def kopeikin(frozen, corr, p, dt, x, *, ecliptic, obliquity):
    mu_a, mu_d = (p.PMELONG, p.PMELAT) if ecliptic else (p.PMRA, p.PMDEC)
    si, ci = sincos(p.KIN); sO, cO = sincos(p.KOM)
    cot, csc = ci/si, 1/si
    dinc_pm = (-mu_a*sO + mu_d*cO) * dt
    dx_pm   = x * cot * dinc_pm
    dom_pm  = csc * (mu_a*cO + mu_d*sO) * dt
    L = corr.ssb_psr_pos                              # ICRS; set by solar_system
    R = frozen.ssb_obs_pos                            # ICRS
    if ecliptic:                                      # same sky frame as KOM
        L = equatorial_to_ecliptic(L, obliquity)      # Vela icrs_to_ecliptic
        R = equatorial_to_ecliptic(R, obliquity)
    sd = L[2]; cd = sqrt(1 - sd*sd); ca = L[0]/cd; sa = L[1]/cd
    I0 = (-sa, ca, 0*sa); J0 = (-ca*sd, -sa*sd, cd)
    dI = dot3(R, I0); dJ = dot3(R, J0)
    dx_px  = x * cot * p.PX * (dI*sO - dJ*cO)
    dom_px = -csc * p.PX * (dI*cO + dJ*sO)
    return dx_pm + dx_px, dom_pm + dom_px, dinc_pm
```

`solar_system` stores `ssb_psr_pos` in ICRS. `KOM` is measured from east in
the model's sky frame, so the annual-parallax `I0`/`J0` are built in that
frame (PINT `update_binary_object`). The rotation is the inverse of
`ecliptic_to_equatorial` and uses the same build-time `obliquity` as
`solar_system`. Equatorial DDK does not rotate. Proper-motion Kopeikin
terms already read ecliptic PM when `ecliptic`.

Build-time checks: astrometry stage present; `ssb_psr_pos` populated before
the binary (order); H3/STIGMA absent; `K96 N` refused (Vela always applies
the PM terms). `ecliptic` follows the astrometry component.

### 7.10b DDR (`binary_ddr.jl`)

The eighth family, and the third Kepler convention. `binary/ddr.py` is a
line-by-line translation of Vela's `binary_ddr.jl`; `docs/CONFORMANCE.md`
carries the source map and the complete deviation list. Normatively:

**Coordinates and solver.** Native `(EPS1, EPS2, TASC)` with `h = EPS1`,
`k = EPS2`, and the regular Laplace-Lagrange equation

```
F − k sin F + h cos F = λ
```

solved by Vela's Newton-with-bisection-bracket update. The bracket moves only
when bisection is used — it is Vela's and PINT's update, not a generic Newton
loop. Because a traced array cannot break on convergence, the loop count is a
**constant 16**, justified by measurement rather than assertion: a
two-million-point fixed-seed disk probe over `√(h²+k²) ≤ 0.99` has a maximum
first-converged **0-based loop index of 8** (the 9th pass) at 22 points and
never fails within 64, so 16 retains seven unused passes while avoiding four
times the transcendental work of a literal 64-step unroll. The probe recipe,
the retained worst cases and the measured cost live in `docs/PARITY.md` and
`tests/test_ddr_performance.py`; the loop count MUST NOT be changed without
re-deriving them.

Differentiation is an **implicit-function custom JVP**,

```
dF = (dλ + sin F·dk − cos F·dh) / (1 − k cos F − h sin F),
```

so no branch predicate and no data-dependent convergence history is ever
differentiated. Convergence is reported from the plain float64 *reference*
channel (R11.3-0).

**Modes.** Six static booleans, resolved once after PINT's `setup()` into a
frozen `DDRConfig` and bound into the stage closure: `use_fbx`,
`ecliptic_coordinates`, `use_pk`, `pbdot_kinematic`, `use_geo`, `use_kine`,
plus the resolved obliquity. No mode is ever a traced predicate, and the stage
MUST NOT inspect a PINT model.

**Consumed parameters are mode-dependent** (R5.3b). A static union over every
mode would claim that an inactive-mode field reaches the trace. `DDRPK`,
`DDRPBDOT`, `DDRGEO` and `DDRKINE` are `INERT_PARAMS`, like `PLANET_SHAPIRO`;
the five galaxy constants are `PINNED_PARAMS` at Vela's/PINT's defaults, since
the physics uses Vela's already-converted literals and a par that moved one
would otherwise be ignored in silence. `TGEO` is **both** consumed (when
geometry or kinematics is on) and inert — PINT's `BinaryDDR.setup()`
materialises `TGEO = TASC` unconditionally, so without the inert
classification an ordinary geometry-off DDR par would fail accountability on
an epoch it never reads. This is the existing `POSEPOCH` pattern.

**Geometry uses the engine's resolved obliquity.** Vela's DDR hard-codes
IERS2010; §7.3's rule applies here too, with the same value `solar_system`
rotates the line of sight with and the same rotations DDK already uses. The
deviation has a measurable physics consequence, so it is gated against PINT —
which honours `ECL` — as a *movement* between two obliquities on one frozen
pre-binary correction.

**Domain: build-time where it can be, traced where it cannot.** Missing
parameters, impossible flag combinations, non-default galaxy constants,
non-zero unsupported placeholders and a free `TGEO` are **build** errors.
Quantities that can leave the physical domain while *sampling* — `COSI`,
the inferred pulsar mass, an evolved `A1`, the phase slope `λ̇`, and the
Shapiro argument `B_S` — are an elementwise `valid` mask on `DDRState`, and
the stage returns NaN delay *and* NaN doppler at those rows. The rule is
"substitute, then select": a safe value MUST be substituted before every
singular square root, division, cube root and logarithm, and `valid` MUST be
updated before each substitution and never derived from a value that was
already replaced. In particular the fallback MUST NOT be spelled
`0.0 * value + fallback`, because exactly where `value` is NaN, `0.0 · NaN`
is still NaN.

`valid` is an `R = N + 1` mask, so an invalid **TZR** row NaNs every residual
through the phase offset even when every science TOA is inside the domain;
domain tests MUST treat the TZR row as its own crossing.

Converting a non-finite residual to a `−inf` log density belongs to
Discovery/nltiming. This package returns NaN and adds no likelihood layer.

### 7.11 `spindown` — §4.4. `phase_offset`, `phase_jump` (`phase_offset.jl`, `jump.jl`)

```python
def phase_offset(frozen, corr, p):
    return corr.add_phase(where(frozen.is_tzr, 0.0, -p.PHOFF))

def phase_jump_exclusive(frozen, corr, p):
    F0 = p.F0_const                                   # Vela: constant F0, not F_spin(t)
    jump = concat([zeros(1), p.JUMP])[frozen.jump_index]
    return corr.add_phase(where(frozen.is_tzr, 0.0, jump * F0))

def phase_jump(frozen, corr, p):                      # non-exclusive bit masks
    F0 = p.F0_const
    jump = sum(where(mask, J, 0.0) for mask, J in zip(frozen.jump_masks, p.JUMP))
    return corr.add_phase(where(frozen.is_tzr, 0.0, jump * F0))
```

## 8. Residuals and TZR (`residuals.jl`)

```python
def form_residuals(frozen, corr):
    psi = corr.phase                                  # includes phi_ref (§4.4)
    return (psi[:-1] - psi[-1]) / corr.spin_frequency[:-1]         # (N,), s
```

The divisor is the **pulsar-frame** `spin_frequency`, not Vela's
`doppler_shifted_spin_frequency` (G2). PINT's `Residuals` default
(`calctype="taylor"`) and tempo2 both divide by the spin Taylor series; Vela
divides by the doppler-shifted instantaneous frequency, and `r·(v/c)` between
them is ~1e-4 of the residual — 87 ns rms on AEI-DR2 EPTA J0613-0200 (86 µs
residuals), 0.2 ns on PPTA DR2 (2.6 µs). With the pulsar-frame divisor this
engine sits at 0.16/0.73/0.17 ns against PINT/libstempo on EPTA/NANOGrav-9y/
PPTA-DR2 J0613-0200, inside the ~1–1.5 ns PINT-vs-tempo2 spread itself.
`design_matrix` is `-jacfwd` of this and follows; `Engine.gauge_direction` is
hand-written and must track it.

`residual_delta(δ) = residuals(δ) − residuals(0)` — the residual minus the
reference residual; the name is nltiming's. `residuals(0)` is cached at
build. Gauge: no mean removed, no gauge applied (`GaugeFacts(export="none",
reference_mode="none", reporting_mode="mean", reporting_weighted=True)` —
PINT-family reporting metadata only).

## 9. Public API (`vela_jax.engine`)

```python
class Engine:
    """Frozen timing package + JAX residual pipeline for one pulsar. Immutable after build."""

    param_names: tuple[str, ...]              # free PINT params, PINT order
    param_units: Mapping[str, str]
    toa_count: int
    stages: tuple[str, ...]
    frozen: FrozenTOAs
    layout: ParamLayout
    pint_model: Any                           # frozen model (read-only)
    timing_package: str                       # "pint" | "tempo2"
    source_units: str                         # "TDB" | "TCB" (as the user wrote it)
    binary_conventions: str                   # "pint" | "tempo2"
    pulse_number_source: str                # "model" | "tempo2"

    @classmethod
    def from_files(cls, par, tim, *, timing_package="pint", binary_conventions="pint", **kwargs): ...
    @classmethod
    def from_pint(cls, model, toas, *, binary_conventions="pint"): ...  # already-built PINT objects
    @classmethod
    def from_tempo2(cls, par, tim, *, binary_conventions="pint", ...): ...   # alias of from_files(..., timing_package="tempo2")

    def reference_theta_exact(self) -> Mapping[str, str]: ...
    def reference_theta(self) -> np.ndarray: ...
    def residuals(self, theta=None) -> np.ndarray: ...
    def residual_delta(self, delta) -> np.ndarray: ...
    def residual_delta_jax(self, delta): ...
    def residual_jacobian(self) -> np.ndarray: ...        # J: r ≈ r0 + J δ
    def design_matrix(self, *, source="jacobian") -> np.ndarray: ...   # R9.3/R9.4
    def precision_critical_params(self) -> frozenset[str]: ...
    def identically_linear_params(self) -> frozenset[str]: ...
    def binary_chart_facts(self) -> BinaryFacts | None: ...
    def gauge_provenance(self) -> GaugeFacts: ...
    def perturbative(self, live_nonlinear, *, dtype=None): ...          # §11
    def pulsar_data(self) -> "PulsarData": ...                          # Addendum B
```

Behavioural rules:

- **R9.1** `residual_delta_jax` is `jax.jit`-compiled at build with the frozen
  arrays baked into the compiled residual as constants; the only traced
  argument is `delta`
  of shape `(n_par,)`.
- **R9.2** `residuals(θ)` computes `δ = θ − θ★` in `Decimal` against
  `reference_theta_exact()` and calls the delta path; it never forms `θ` in
  float64 inside the trace.
- **R9.3 (the canonical design matrix).** `design_matrix()` (default
  `source="jacobian"`) returns

  ```
  M := −residual_jacobian() = −jacfwd(residual_delta_jax)(0)
  ```

  in fitter sign (`r(θ★+δ) ≈ r(θ★) − M δ`, so `J = −M` **exactly, by
  construction** rather than by gate), seconds per PINT unit, `param_names`
  order. This is the matrix `PulsarData.Mmat` carries, the matrix Enterprise
  and Discovery marginalize, and the matrix nltiming's `"analytic"` route
  reads as its source of record (`pulsar.Mmat`) — so the analytic and
  autodiff routes *coincide* for this pulsar. It is TZR-aware and
  feedback-aware because the residual it differentiates is.
- **R9.4 (PINT as oracle).** `design_matrix(source="pint")` returns PINT's
  analytic matrix on the frozen pair (Offset column dropped — PHOFF is a
  parameter — converted to `param_names` units and fitter sign). It is an
  oracle, not a product: identically-linear columns MUST agree with R9.3 to
  10⁻⁶ relative (§12 M1/T9); the remaining columns' phase-frame discrepancy
  (PINT ignores the TZR row's parameter dependence, the delay-feedback term,
  and divides by constant `F0` rather than the doppler-shifted spin
  frequency) is *reported* as a diagnostic bounded by the total `|dD/dt|`
  (~3×10⁻⁴ on a compact binary), never gated as equality.
- **R9.5** Zero-delta short-circuit is not implemented in the traced path (it
  must be a pure function); the NumPy `residual_delta` may short-circuit.
- **R9.6 (the Jacobian's construction cost, named).** `residual_jacobian()`
  is an `N × n_par` forward-mode evaluation with its own compile — a real
  cost on J1713-class data. It MUST be computed **once** and cached on the
  engine; `pulsar_data()` triggers it (a `PulsarData` without `Mmat` does not
  exist). A consumer that wants a frozen T0 analysis without paying it uses a
  previously written feather plus nltiming's `LinearTimingEngine` — **not** a
  live engine rebuild, and **never** by substituting the `source="pint"`
  matrix into `Mmat`: that substitution reintroduces the pulsar/engine
  disagreement and is a spec violation, not an optimization. The README MUST
  say that a residuals-only look uses `Engine`, not `TimingPulsar` (V2-P6).

```python
@dataclass(frozen=True)
class BinaryFacts:
    family: str                  # "ELL1" | "ELL1H" | "ELL1k" | "DD" | "DDH" | "DDS" | "DDK" | "DDR"
    kepler_convention: str       # "dd" | "ell1" | "ddr"
    use_fbx: bool
    shapiro: str                 # "m2_sini" | "h3_stig" | "shapmax" | "kin" | "m2_cosi"
    epoch_shift_exact: bool      # False for DDR: its epoch convention is its own
    secular_terms: tuple[str, ...]
    ell1_t2: bool = False        # A.4 truncation active
    supports_domain: bool = True # False for DDR (§7.10b)
```

`supports_domain` answers "does a valid box prior on this family's independent
inputs guarantee a physical state?". Seven families say yes and keep the
default; DDR says no, because a sampled `(A1, PB, M2, COSI)` can imply a
negative pulsar mass or a non-positive Shapiro `B_S`. `binary_chart_capability`
forwards `kepler_convention`, `epoch_shift_exact`, `secular_terms`,
`origin_certified` and `supports_domain` — **not** `shapiro`, which is
inventory only. nltiming MUST NOT apply its DD polar-to-Laplace chart to a
family reporting `kepler_convention="ddr"`.

## 10. nltiming interface

This package implements the **backend** side of nltiming's protocols
structurally — `TimingEngine`, `JacobianTimingEngine`, `JaxTimingEngine` for
the engine; `PulsarData`, `TimingPulsar` for the product — without importing
nltiming. The engine protocol surface, as nltiming defines it today:

```python
class TimingEngine(Protocol):
    fitpars: tuple[str, ...]
    native_units: Mapping[str, str]
    def reference_theta(self) -> np.ndarray: ...
    def reference_theta_exact(self) -> Mapping[str, str]: ...
    def residual_delta(self, delta_theta) -> np.ndarray: ...
    def design_matrix(self, params=None) -> np.ndarray: ...
    def gauge_provenance(self) -> GaugeProvenance: ...
    @property
    def gauge_applied(self) -> bool: ...

class JacobianTimingEngine(TimingEngine): residual_jacobian() -> np.ndarray
class JaxTimingEngine(TimingEngine):     residual_delta_jax(delta); precision_critical_fitpars()
```

Rules:

- **R10.1** The `VelaJaxTimingEngine` adapter (B.7) renames `param_names → fitpars`
  and `param_units → native_units`, adds the `params=None` signature, and
  forwards everything else. It MUST NOT recompute anything.
- **R10.2** `derivative_method` is honoured, not ignored: under R9.3 the
  `"analytic"` and `"autodiff"` routes are the same matrix, so both values
  are accepted and **recorded** on the backend for nltiming's run manifest;
  any other value raises. The v1 behaviour — nltiming asks for `"autodiff"`
  and silently gets PINT's M — is impossible by construction.
- **R10.3** `nonlinear_params` is the **executed** hybrid mode: `None` keeps
  every axis on the full nonlinear path; `"binary"`/`"binary+"` (and
  `"binary+astrometry"`) are executed by `Engine.perturbative` (§11), and the
  backend's `nonlinear_params` attribute reports the mode actually executed —
  satisfying nltiming's `_check_engine_nonlinear_params` refusal contract and
  its manifest recording. A mode nltiming grows that this package cannot
  execute is refused by name, never silently narrowed.
- **R10.4** Engine vocabulary: `vela_jax` is registered under **both** native
  packages (`"tempo2": (..., "vela_jax")`, `"pint": (..., "vela_jax")`), with
  `_IMPL_FAMILY["vela_jax"] = "pint"` — the only engine name valid for both,
  because the timing package is separated from the physics. MetaPulsar
  dispatches on the engine name and picks the timing package from each leg's own `timing_package`.
- **R10.5** Gauge: the engine is gauge-free (`export="none"`); `PHOFF` is
  always free (§5.3) and its R9.3 column is the exact constant-phase
  direction, so nltiming's `assert_gauge_column_present` numeric check (SVD
  span of the constant on the gauge column) holds by construction. `Offset`
  is never implemented as a TZRMJD perturbation — trivially honoured here:
  TZR geometry is frozen, `PHOFF` is a parameter.

## 11. Single-precision perturbative engine (`vela_jax.perturbative`)

### 11.1 Contract

The default engine is float64 and MUST remain so (G1, `require_x64`). A PTA
likelihood is dominated by the `O(n_par² n_TOA)` timing-marginalisation and
the red-noise / GW matrix work, not by the `O(n_TOA)` residual; a consumer MAY
therefore keep this engine in float64 and cast `r` and `M` to float32 at the
likelihood boundary. JAX differentiates that cast (`lax.convert_element_type`
between floating dtypes has the identity JVP with a dtype change), so a
float32 likelihood over a float64 residual is a valid gradient path. That is
the recommended integration.

This section exists for the other case: a Discovery kernel whose *working
dtype is already float32* and which wants the live nonlinear axes evaluated
in the same kernel without a dtype seam. It is optional. It is not a claim
that residual evaluation is the bottleneck. The linear block of
`PerturbativeEngine` already is the cast-at-the-boundary recipe: `M` baked
in fp64, cast once, `M @ δ` under `jax.default_matmul_precision("highest")`.

```python
class PerturbativeEngine:
    """fp32-capable hybrid residual delta over a restricted live set.

    Δr(δ) = −M @ δ_lin − (ΔD − (F_tzr/F_i)·ΔD_tzr)
    """
    dtype: jnp.dtype                     # float32 or float64
    live_nonlinear: tuple[str, ...]      # §11.2
    parent: Engine
    param_names, param_units, reference_theta_exact, ...   # delegated
    def residual_delta_jax(self, delta): ...
    def residual_delta(self, delta): ...
    def residual_jacobian(self): ...
    def certify(self, deltas=None, *, rtol=1e-5, atol=1e-12) -> CertifyReport
```

Built from a parent `Engine` (`engine.perturbative(mode_or_axes, dtype=...)`).
Every axis not live is served by the R9.3 matrix cast to the working dtype.
It never evaluates the full pipeline absolutely in the trace.

### 11.2 Supported live nonlinear set (R11.2)

```
astrometry:  RAJ DECJ | ELONG ELAT, PMRA PMDEC | PMELONG PMELAT, PX
binary:      A1 PB|FB0 ECC OM T0 | EPS1 EPS2 TASC, SINI M2 | H3 STIGMA | SHAPMAX,
             KIN KOM, GAMMA, OMDOT PBDOT EDOT A1DOT EPS1DOT EPS2DOT LNEDOT,
             COSI GGAMMA XPBDOT                                    (DDR)
```

`TGEO` is frozen metadata, not an axis, and DDR's static mode flags and galaxy
constants are not axes either. The companion registry in
`nltiming.hybrid.BINARY_AXES` MUST gain `COSI` and `GGAMMA` (`XPBDOT` is
already there), and `COSI` MUST get the signed physical domain `(-1, 1)` so a
sampled prior matches Vela/pyvela's isotropic `Uniform(-1, 1)`. The engine
still enforces the strict `|COSI| < 1`: a sampled endpoint has zero measure
and returns the documented invalid-state NaN rather than being clamped.

Named modes (aligned with nltiming's `nonlinear_params` vocabulary, R10.3):
`"binary"` (binary only), `"binary+"` (adds PX), `"binary+astrometry"` (adds
sky, PM, PX everywhere), `"astrometry"`. Spin, DM, FD, JUMP, PHOFF, DMX are
never perturbative-live (identically linear, or linear far below fp32
resolution for a PTA MSP); requesting them raises.

### 11.3 Delta-formulation rules (R11.3)

0. **`numerics.where`'s condition MAY be traced** (new in v2.6). Before DDR it
   was always a frozen boolean array. DDR's physical domain is derived from
   *sampled* parameters, so its validity predicate is evaluated from
   `ref + delta` under `Pert`. A `Numeric` implementation MUST therefore apply
   the chosen branch consistently to every channel it carries, and a caller
   MUST still substitute a safe value *before* the singular expression rather
   than selecting a NaN away afterwards — an unselected NaN poisons `jacfwd`
   whether or not its branch is taken. `Pert.value` (`ref + delta`) MAY be
   formed for such a predicate or for reporting, never for arithmetic in the
   perturbation channel; solver convergence specifically reads
   `Pert.reference`, because the difference solve is deliberately
   float32-capable and cannot meet a float64-epsilon residual bound.

1. **The reference channel is float64, always.** `Pert.ref` (§11.5) is
   evaluated in float64 regardless of the working dtype. The reference
   channel depends only on frozen constants — never on δ — so under `jit`
   the fp64 reference is computed once at compile time and costs nothing per
   evaluation. Reference factors are cast to the working dtype **at
   the point where they enter the perturbation channel**, after being
   computed in fp64. (`cast_frozen` down-casts only delta-channel inputs.)
   This is the rule the v1 dual violated — it recomputed its reference in the
   working dtype, and a strongly-cancelling reference (DDS `SHAPMAX≈9`,
   `sin i = 0.99989`: the Shapiro log argument at 10⁻⁴) poisoned the fp32
   delta on `J2302+4442`. With this rule that fixture moves from "known gap"
   into the passing set (§12 T15).
2. A traced term MAY be evaluated absolutely in the perturbation channel only
   if `|value★|·2⁻²⁴ ≤ 10⁻¹³ s` (i.e. `|value★| ≲ 1 µs`); otherwise it MUST
   be formed by a §11.4 identity.
3. No working-dtype quantity of magnitude > 2²⁴ may carry sub-unit
   information: `Δt★` (10⁸ s) may only multiply *deltas*.
4. `jax.default_matmul_precision("highest")` is required for the `M @ δ`
   product on GPU (TF32 is not acceptable).

### 11.4 Identities (normative)

```
Δsin(Φ) = sinΦ★·(cos x − 1) + cosΦ★·sin x,   cos x − 1 = −2 sin²(x/2)     x = ΔΦ
Δcos(Φ) = cosΦ★·(cos x − 1) − sinΦ★·sin x
Δ(a·b)  = a′·Δb + b★·Δa
Δlog A  = log1p(ΔA / A★)
Δexp A  = exp(A★)·expm1(ΔA)
Δ√Q     = √Q★ · ε/(1 + √(1+ε)),   ε = ΔQ/Q★
Δatan2(y,x) = atan2( y′x★ − x′y★ , x′x★ + y′y★ )
Δarccos     = third-order expansion (sole use: solar-wind sun angle, dc ~ 1e-9)
Δkepler     : u′ − e′sin u′ = l′ solved IN THE DIFFERENCE VARIABLE x = u′ − u★:
              x − e′[sin u★(cos x − 1) + cos u★ sin x] = Δl + δe·sin u★,
              Newton from the linear guess, 4 fixed iterations
Δcbrt A     = ΔA / (b² + ab + a²),  a = ∛A★, b = ∛A′   — never ∛A′ − ∛A★
Δregular_kepler : F′ − k′sin F′ + h′cos F′ = λ′, again in x = F′ − F★,
              4 fixed Newton iterations; the reference solve is the fp64
              16-pass kernel of §7.10b
nan_where(valid, ·) : NaN in EVERY channel where ¬valid. `select(valid, ·, nan)`
              is not equivalent — lifting a scalar NaN gives it a *zero*
              perturbation, so `delay_delta` would report an invalid sampled
              point as no change at all
```

### 11.5 The dual (R11.5) — one physics, two channels

The archive specified a hand-written delta kernel per component; the
implementation replaced that with a structurally better mechanism, which is
now the norm. Because the chain is written against `numerics` (R3.6), a value
type `Pert(ref, delta)` implements the algebra with each operation being the
corresponding §11.4 identity, and **running the unmodified component chain
over `Pert` values yields `ΔD(δ)` directly** — no second copy of the physics,
and delta coverage of every component, including ones no kernel list would
have named. Injection happens in `ParamLayout.build(delta, wrap=...)`: live
axes become `Pert(θ★, δ·scale)`, everything else stays a plain reference
value. The identities are tested against a `longdouble` oracle (a naive
float64 `f(x+h) − f(x)` is only good to ~10⁻¹⁶ — exactly the cancellation
they exist to avoid).

### 11.6 Assembly (R11.6)

```
Δr = −M @ δ_lin − (ΔD − (F_tzr/F_i)·ΔD_tzr)
```

The delta chain runs the delay stages only (phase stages — spindown,
phase_offset, phase_jump — are served by the design matrix). Justification:
`Δψ = −F_spin·ΔD` to first order in ΔD, and `r = ψ/(F_spin(1+doppler))`, so
`F_spin` cancels for the TOA's own term; the TZR row moves too and carries
its own spin frequency into the ratio (1 to a part in 10⁹ for an MSP, several
percent for a fast spinner — carried, not assumed). The dropped
`Δdoppler·ΔD` cross term is **second order in the residual change** with
measured coefficient ~4×10⁻⁴ s⁻¹, reported by
`CertifyReport.quadratic_coefficient`: a picosecond at Δr ≈ 50 µs, a
nanosecond at Δr ≈ 1.6 ms — outside anything a posterior visits. A
`carry_doppler=True` build option MAY add the term (the chain already
computes `Δdoppler`); it is not required.

### 11.7 Gates (R11.7)

`certify(deltas)` compares against the fp64 parent on ±1σ/±3σ per live axis
plus random joint draws, restricted to the live nonlinear axes (the linear
block is the parent's own matrix), and requires
`‖Δr_pert − Δr_exact‖∞ ≤ rtol·‖Δr_exact‖∞ + atol` with `rtol = 1e-5,
atol = 1e-12 s`, plus Jacobian columns ≤ 1e-4 relative. An fp64
`PerturbativeEngine` (same code, `dtype=float64`) MUST agree with the full
engine to 10⁻¹² s absolute on the same deltas — the test that the delta
*formulation* is right, independent of fp32.

## 12. Tests and parity budgets

Carried gates (v1, measured status in `docs/PARITY.md` / `CONFORMANCE.md`):

| # | Gate | Oracle | Budget |
|---|---|---|---|
| T1 | freeze refuses missing planets/pulse numbers/tdbld; every §1.2 refusal | — | raises by name |
| T2 | `residual_delta(0)` | zeros | exactly 0 |
| T3 | `residuals(θ★)` vs PINT `Residuals` | PINT | 10⁻⁷ s per fixture (10⁻⁶ ELL1H/DDH — PINT's NHARMS vs Vela's analytic harmonics; Vela is the authority, T4 gates it) |
| T4 | `residuals(θ★)` vs `SPNTA` | pyvela (oracle extra) | RMS ≤ 1 ns, max ≤ 10 ns (measured: ≤ 124 ps) |
| T5 | per-component delay/phase vs Vela JSON tables | Vela fixtures | 10⁻¹² s relative — **open; only Vela.jl can close (§16.7)** |
| T6 | Mikkola vs Vela table + scipy Kepler on grid | both | 10⁻¹³ rad |
| T7 | `residual_delta(δ)` vs `SPNTA` deltas, ±1σ/±3σ | pyvela | RMS ≤ 1 ns (measured: ≤ 105 ps) |
| T8 | `residual_jacobian` vs central differences | self | 10⁻⁶ per column, no NaN |
| T9 | R9.4 diagnostic: PINT M vs −J | PINT | identically-linear ≤ 10⁻⁶; rest reported in phase, alert beyond the `|dD/dt|` bound |
| T10 | trace never calls PINT (monkeypatch during jit) | — | passes |
| T11 | par with noise lines builds; residuals bitwise identical to pre-stripped par | self | bitwise |
| T12 | live T0/TASC/PB vs pyvela | pyvela | ≤ 1 ns |
| T13 | orbit-count reduction vs unreduced at n_orb = 10⁴ | self | reduced floor ≤ 0.05 ps. **DDR PB chart: met. DDR FBX chart: ≈ 16 ps**, and the excess is one named rounding — `FB0·P★` rounds to exactly 1.0, so the retained-integer bracket `n_orb·(FB0·P★ − 1)` evaluates to zero against a true ~5·10⁻¹⁷. This is the reduction identity's own floor, shared verbatim with `binary.orbit.mean_anomaly` and every FBX family; the gate asserts the error **is** that term rather than loosening a bound |
| T14 | perturbative fp64 vs full engine | self | ≤ 10⁻¹² s |
| T15 | perturbative fp32 vs full engine, **full fixture set incl. J2302+4442** (R11.3 rule 1) **and `sim_ddr`** | fp64 parent | rtol 10⁻⁵, atol 10⁻¹² s; Jacobian 10⁻⁴ |
| T16 | fp32 engine inside a Discovery fp32 likelihood (smoke) | Discovery | NUTS runs; SINI/M2 posterior matches fp64 within MC error |

New gates (v2):

| # | Gate | Oracle | Budget |
|---|---|---|---|
| P1 | R-B1.4 identities: `toas == tdbld·86400 − delay_bary★`, `freqs == freq_hz·(1−doppler_bary★)/1e6` (pre-binary snapshot), **and** `toas` vs PINT `get_barycentric_toas` on a PINT binary fixture | self + PINT | exact array equality; vs PINT ≤ 10⁻⁷ s (T3's budget — a wrong cutoff is *seconds*) |
| P2 | one-source guard: PINT physics methods raise during `PulsarData` build | — | passes |
| P3 | R5.5: the freeze reproduces the timing package's rows unchanged; no permutation is published anywhere | self | exact |
| P4 | feather round-trip lossless; file then read by stock `discovery.Pulsar.read_feather` **and** Enterprise `FeatherPulsar.read_feather` | consumers' own readers | lossless / consumable |
| P5 | `dmx` table vs par DMX lines; windows partition the `stoas` they claim | par | exact |
| P6 | `planetssb`, the slots Enterprise actually reads: slots {2,4,5,6,7} positions finite and matching Enterprise-PINT on a PINT fixture; velocities NaN. **Slot 1 (Venus) is excluded from the Enterprise-PINT match** — Enterprise's PINT path leaves it NaN, this package fills it when frozen (B.3.3); nobody NaNs Venus "for parity" | enterprise | ≤ 10⁻¹⁰ ls |
| M1 | R9.3: `−Mmat` vs central finite differences of `residual_delta`, **all** columns, posterior scale | self | 10⁻⁶ relative, no NaN |
| H7 | A.6.1 freeze comparison (same JAX chain, two packages) | both timing packages | RMS ≤ 100 ns on mixed-engine-consistent observatory files; `ECL IERS2003` so PINT uses tempo2's obliquity |
| H8 | A.6.2 required `TRACK −2` | tempo2 | `pulse_number_source == "tempo2"` + half-turn tripwire |
| S4 | A.6.3 libstempo, ELL1+FD, discriminating eccentricity | libstempo | RMS ≤ 50 ns under `"tempo2"` conventions; MUST fail under `"pint"` |
| N1 | nltiming installed: protocols by `isinstance`, `validate_engine_against_pulsar`, gauge assert, `TimingSpec.for_pulsar` end-to-end, hybrid manifest check | nltiming | passes; **a CI job with nltiming installed is mandatory** — a structural twin guarded only by an optional test is unguarded |
| P7 | R5.3b: a non-zero unconsumed parameter is refused by name (`A0`, `B0`, `SWM 1`, wideband); frozen bare `DMX` is inert; `ECL` is read and moves the residual as PINT's does | PINT | refuses by name; `ECL` agreement ≤ 0.1% of the shift it causes |

DDR gates (v2.6):

| # | Gate | Oracle | Budget |
|---|---|---|---|
| D1 | Vela's own scalar delay anchors, PK / phenomenological / FBX charts, plus `mp`, `g_gamma` and the derived `κ` | Vela `test_ddr.jl` | 1 ps absolute (measured ≤ 1·10⁻¹⁵ s) |
| D2 | Vela's injected-`(I, J)` geometry anchors and its piecewise kinematic `Pbdot` (`p`, `p_shk`, `p_gal`, `p_gw`) | Vela `test_ddr.jl` | 1 ps; relative 10⁻¹⁰ for the `Pbdot` pieces |
| D3 | the derived TGEO triad — real equatorial geometry — against PINT's own DDR kernel on one frozen pre-binary correction | PINT | 1 ps |
| D4 | `ECL` **movement**: two obliquities, one frozen pre-binary correction passed identically to both engines and both PINT components | PINT | fixture must move materially; agreement 1 ps |
| D5 | invalid sampled domain returns NaN and does not raise, under `jit`; `jacfwd` at a valid reference stays finite; an invalid **TZR** row NaNs every residual on its own | self | NaN, never an exception |
| D6 | the 16-pass solver converges on the committed stress grid and on the 2·10⁶-point probe; the retained 22 worst cases still first converge at 0-based index 8 | self | every point by index 15 |
| D7 | cost: 16-step vs a 64-step twin (primal), the implicit JVP vs one primal solve, the solver vs compiled `mikkola`, and `binary.DDR` vs `binary.DD` — all on the same 20 000 rows, with the orbital **epoch traced** so the solve is not constant-folded | self | ≤ 35 %, ≤ 2×, ≤ 10×, ≤ 15×; ratios are the gate, wall clock is informational. A 16-vs-64 *gradient* ratio is not measurable: with the implicit rule the derivative cost is independent of the loop count, and a 64-step twin without a custom rule does not compile in six minutes — which is itself the measurement justifying the rule (`docs/PARITY.md`) |
| D8 | `M2 / M_SUN` mismatch pinned: the layout converts `M2` with PINT's `GMsun/c³` and DDR divides by Vela's literal, which differ by −1.36·10⁻¹⁰ relative | measured | pinned, not silently changed — see `docs/PARITY.md` |

Fixtures: Vela.jl's `pyvela/examples` (`VELA_JAX_FIXTURES`), plus the named
A.6 fixtures. CI without Julia runs everything but T4/T7/T12 (the `oracle`
extra); the `tempo2` extra runs the H/S gates.

## 13. Package layout and dependencies

```
vela-jax/
  pyproject.toml            # deps: numpy, jax, pint-pulsar, astropy
                            # extras: oracle = [pyvela, juliacall]; tempo2 = [libstempo]
  SPEC.md                   # this document
  docs/REVIEW.md CONFORMANCE.md PARITY.md
  src/vela_jax/
    config.py errors.py constants.py units.py taylor.py
    freeze.py               # strip, load_pint, prepare_model, FrozenTOAs, TOA columns
    precision.py            # τ, phi_ref + spin_coeffs, orbit reduction (§4)
    params.py correction.py numerics.py
    pipeline.py             # stage tables (both conventions), run_chain, form_residuals
    astrometry.py solarwind.py dispersion.py frequency_dependent.py
    spindown.py phase.py
    binary/                 # orbit.py dd.py ell1.py facts.py
    engine.py               # Engine (§9)
    read_tempo2.py tcb.py   # Addendum A
    perturbative/           # dual.py (two-dtype Pert, R11.3-1) certify.py
    pulsar_data.py          # PulsarData + feather schema v1 (Addendum B)
    pulsar.py               # TimingPulsar (composition, B.6)
    backend.py              # VelaJaxTimingEngine + capability mirror (B.7)
  tests/  examples/
```

No `combined.py`, no `derivatives_*.py`, no `session.py`, no GUI, no
`noise/`, no `prior/`.

## 14. Implementation phases (v2 migration order)

**Shipping rule (R14.0).** Part II ships on its own gates (P1–P6, H7/H8, S4,
M1, N1, T15). The Part III consumer changes — Enterprise's `Pulsar(...)` duck
check, nltiming calling its consistency gate — are **follow-up PRs in those repos**;
if they slip, the timing package is still honest. Part III never blocks
Part II.

- **V2-P0** The row-order rule (R5.5) + P3: the freeze passes the timing package's rows
  through untouched, and nothing publishes a permutation. Everything else
  builds on one order. Start A.6 fixture procurement now — it is the long
  pole.
- **V2-P1** `PulsarData` with one-source fields, real `dmx`, columnar flags,
  the planetssb contract; P1/P2/P5/P6. Delete the v1 `TimingPulsar`
  internals (the second-PINT-pass code and its excuses).
- **V2-P2** `Mmat := −J` (R9.3–R9.6); M1; the N1 CI job.
- **V2-P3** Feather schema v1, round-trip, consumer-reader gates (P4).
- **V2-P4** Two-dtype dual (R11.3 rule 1); T15 on the full set.
- **V2-P5** Tempo2 checks H7/H8/S4.
- **V2-P7** R5.3b + `ECL` (§7.3, A.2.9) + gate P7. Changes numbers on every
  ecliptic par that is not `IERS2010`, which is most real data, so it lands
  before anything downstream is re-measured.
- **V2-P6** Docs: README rewritten against the new object (two-PTA claims
  attributed to MetaPulsar; the `from_tempo2` snippet shows
  `binary_conventions="tempo2"` for the EPTA case; a residuals-only look
  uses `Engine`, not `TimingPulsar`); REVIEW/CONFORMANCE/PARITY updated;
  deviation ledger extended.

## 15. Relations to the other packages

| Package | Relation |
|---|---|
| **Vela.jl / pyvela** | physics authority, oracle, fixture source. vela-jax is a separate GitHub repository (not a Vela.jl-organisation package, not a `pyvela` sub-package; §16.1). Cite Susobhanan 2025 ApJ 980 165 (`CITATION.cff`). The JSON oracle tables (T5, §16.7) remain the highest-leverage open ask. |
| **PINT** | timing-package and model-semantics authority; design-matrix **oracle** (R9.4), no longer product. |
| **nltiming** | consumer; owns the protocols, charts, priors, linearity policy, `engine_config`. §III.3. |
| **MetaPulsar** | the multi-PTA combiner; per-leg `VelaJaxEngine` dispatches on the `vela_jax` engine name, timing package per leg's `timing_package`; a vela-jax composite is a `PulsarJaxTimingEngine`. Converges on emitting `PulsarData` for composites (§III.4). |
| **Discovery** | consumes the fp64 engine through nltiming's delay callable and (later) the fp32 perturbative engine in its fp32 kernel; owns all noise bases. Reads the feather natively. |
| **Enterprise** | consumes the pulsar product (reads it exactly as a `FeatherPulsar`); owns signals/likelihoods. §III.5. |
| **JUG** | unrelated at runtime; two cited algorithms re-implemented with credit (§2). JUG remains the tempo2-compatible timing application — vela-jax never absorbs that job. |

License: GPL-3.0-or-later (translation of GPL-3 Vela.jl). Closed by the
Vela.jl author (§16.2): GPL-3 or later, no dual license.

## 16. Open questions (Vela.jl review asks)

Kept under their original numbers — `docs/REVIEW.md` answers those that now
have measurements instead of guesses.

1. **Home and name** — **closed.** Separate GitHub repository, installable on
   its own. Not a Vela.jl-organisation repo, not a `pyvela` sub-package, not
   co-owned inside MetaPulsar. The first Vela paper is the repository
   citation (`CITATION.cff`, README): Susobhanan 2025, ApJ 980, 165.
2. **License** — **closed.** GPL-3.0-or-later, as requested. No dual license.
3. **Binary Doppler term** — **closed as a physics question.** The term
   (`doppler = ΔREp · n̂`, ported as `drep * nhat`) is the binary Doppler the
   author added later; the `"Is this accurate enough?"` comment in
   `binary_dd_base.jl` predates it and is not reproduced here. Its reach is
   unchanged: the binary runs after dispersion, so a 1% error in the term is
   ~1 ps of residual. Not worth improving while it stays behind dispersion.
4. **Proper motion** — Vela's linear-then-renormalise vs astropy: measured
   equal to 0.4 ps at 8 mas/yr over 5 yr; tempo2 uses Vela's convention; one
   long-baseline high-PM fixture would close the J0437 case.
5. **DDK `K96`** — Vela's behaviour ported; `K96 N` raises.
6. **ELL1 truncation** — implemented as the A.4 flag and measured (22.9 µs on
   `J1227-6208`); confirmation wanted that tempo2's ELL1 truncation applies
   to ELL1H.
7. **Shared fixtures** — the JSON component-table export from `Vela.jl/test`;
   still the ask.
8. **Wideband** — not in v1/v2; freeze can keep DM-info columns on request.
9. **Live epochs** — T0/TASC live and gated; PEPOCH is the time origin and
   stays frozen.
10. **`PHOFF` forced free** — kept; nltiming requires the gauge column
    (R10.5).

---

# Addendum A — the tempo2 timing package and `binary_conventions`

## A.1 Two orthogonal choices

| | values | what it changes |
|---|---|---|
| **timing_package** | `pint` (default), `tempo2` | who reads par/tim and produces the frozen arrays |
| **`binary_conventions`** | `"pint"` (default), `"tempo2"` | whose binary-*input* conventions the chain applies |

`Engine.from_files`/`from_pint` are the PINT. `Engine.from_tempo2(par,
tim, *, binary_conventions=…)` is the tempo2 (libstempo, sandboxed —
tempo2 segfaults on real data often enough that the parent process must
survive it). Both take `binary_conventions`, so the flag is testable in
isolation. A tempo2-read EPTA file usually wants
`binary_conventions="tempo2"`; the README example MUST show that pairing.

## A.2 Authority under the tempo2

tempo2 owns: INCLUDE trees, `TIME`/`MODE`, site codes, the clock chain,
TT→TDB, JPL/site ephemeris vectors, and the phase connection (`TRACK -2` /
`-pn`). PINT owns: the timing-model meaning (BINARY family, masks, units,
`PHOFF`) and the R9.4 oracle matrix, both read off the *same* TDB par. Vela
owns every delay. Nothing of tempo2's delay tree enters the trace.

- **A.2.1** `tdbld` is `SAT + correction_tt + correction_tt_tb`, assembled in
  longdouble. It MUST NOT be `bbat`/`bat` (barycentric — Vela's
  `solar_system` would double-count the Roemer delay).
- **A.2.2** `ssb_obs_pos = earth_ssb + observatory_earth`, light-seconds;
  `ssb_obs_vel = earth_ssb[3:6] + siteVel`. tempo2 stores the site *position*
  in `observatory_earth[0:3]` and leaves `[3:6]` at zero; the rotational
  velocity is the separate `siteVel` field (`dm_delays.C:99`). Using the
  zero half shortens the freeze velocity by ~1.3% and reaches the residual
  through the doppler-corrected frequency (~100 ns on a real MSP at 1400 MHz).
  `obs_sun_pos` and planet columns are those bodies' SSB positions minus it.
  An ecliptic par: tempo2 has rotated *every* vector with its own obliquity
  (`ECLIPTIC_OBLIQUITY_VAL = 84381.4059″`); rotate back to ICRS with that
  same constant — not Vela's `OBL`, which rotates a line of sight, not a
  frame. A barycentric site (`bat` / `@`) is the SSB: `ssb_obs_pos` is
  zero and tempo2's `roemer` is already zero.
- **A.2.3 (gate)** Before any rotation, in tempo2's own frame,
  `psrPos·R − ½·PX·R⊥²` MUST reproduce tempo2's `roemer` to ≤ 1 ns (measured:
  ≤ 0.4 ps). Exact to third order in proper motion (`calculate_bclt.C`:
  `rcos1 + dt_pm` is the unnormalised line of sight, `dt_pmtt` its
  second-order normalisation term). The build **raises** if it fails; the fix
  is the vector mapping, never the physics.
- **A.2.4** The observing frequency handed to the chain is topocentric
  (`freq_ssb` already carries tempo2's Doppler shift and would double-count).
- **A.2.5** The TZR pseudo-TOA's geometry comes from PINT; it contributes one
  constant phase, which the always-free `PHOFF` absorbs.
- **A.2.6** Pulse numbers come from tempo2 **only** when tempo2 was given the
  phase connection, which means `TRACK -2` **and** `-pn` flags — both, not
  either. `formResiduals.C:2263` reads the flags only inside the `TRACK -2`
  branch, and the consequence is measured: bumping a `-pn` value by one turn
  on a tim with no `TRACK` changes tempo2's residuals by nothing at all, while
  the same edit under `TRACK -2` moves every affected residual by exactly one
  period. Most of Vela.jl's own fixtures carry decorative `-pn` flags and no
  `TRACK`, so treating a bare flag as a connection claims an authority tempo2
  never exercised. Otherwise the model defines them, as on the PINT. When
  tempo2 does own them, only the *origin* is taken from PINT (median
  difference against `model.phase(abs_phase=True).int`); every pulse-to-pulse
  difference stays tempo2's — that is the entire content of the phase
  connection.
- **A.2.7** PINT MUST NOT recompute clocks, positions or pulse numbers for
  the science TOAs (guard on `compute_TDBs`/`compute_posvels`/
  `compute_pulse_numbers`; the one-row TZR object is exempt).
- **A.2.8** tempo2 spells `FDJUMPn`; PINT reads `FDnJUMP` — respelled in the
  PINT copy of the par text only, so both codes see the same numbers. In the
  other direction, PINT writes `CLOCK` where `readParfile.C` reads only `CLK`:
  respelled in the *tempo2* copy. A PINT-written par that keeps `CLOCK`
  silently un-pins tempo2's clock chain.
- **A.2.9 (the obliquity is tempo2's).** tempo2 has no `ECL` keyword; it
  rotates every ephemeris vector into ecliptic coordinates with
  `ECLIPTIC_OBLIQUITY_VAL` = 84381.4059″ whatever the par says. A.2.2 undoes
  exactly that rotation, so the line of sight MUST be rotated back with the
  same constant — a dot product is rotation-invariant only if both sides use
  one frame. The engine therefore uses tempo2's obliquity when tempo2 reads,
  overriding `ECL` (§7.3), and **warns** when the par asked for another
  realisation: that is the right answer for a tempo2-fitted par, whose
  ELONG/ELAT were fitted in that frame, and a real ~120 ns inconsistency for a
  PINT-fitted one. It is a genuine PINT/tempo2 disagreement, not a defect.
  A.6.1 sets `ECL IERS2003` on the PINT copy so both packages use tempo2's
  default rather than absorbing the 0.1 mas into the freeze floor.
- **A.2.10 (H7 compares freezes, not package residual functions).**
  `Engine.residuals()` is the JAX delay chain on the freeze. Switching
  `timing_package` changes only the frozen arrays. Identical freezes
  (barycentric TOAs, same `tau`) make the JAX residuals bit-identical even
  when `PINT.Residuals` and libstempo still differ. A.6.1 therefore bounds
  clock-file, ephemeris-interpolation, planet-position, and convention-default
  disagreement in the freeze, not PINT vs tempo2 delay physics. When
  `PLANET_SHAPIRO Y`, `planet_pos` is interpolated independently by each
  package — a freeze field, tens of nanoseconds on Vela's `sim_sw`. Mixed
  engines turn planetary Shapiro off.

## A.3 Timescale

A tempo2 par is TCB unless it says otherwise: absent `UNITS` ⇒ TCB, `UNITS
SI` ⇒ TCB, duplicate active `UNITS` lines refused. Conversion is `tempo2 -gr
transform … tdb` on the text (then the `NE_SW` dedupe — old tempo2 builds
write it twice), **before tempo2 itself runs**, so its own
`correction_tt_tb` is TT→TDB rather than TT→TCB — one par text, one
timescale, everywhere. IFTE never enters JAX. The rules are MetaPulsar's; the
code is local (vela-jax never imports MetaPulsar).

## A.4 The `"tempo2"` conventions

One difference, resolved into a static closure at build:

**ELL1 truncation** (`ELL1model.C`), for ELL1/ELL1H/ELL1k:
`dre = a1(sinΦ + ½(ε₂sin2Φ − ε₁cos2Φ))`, `drep = a1 cosΦ`,
`drepp = −a1 sinΦ`. Measured difference vs Vela's cubic expansion:
3.6 ns at e = 2.5×10⁻⁶, **22.9 µs** at e = 1.2×10⁻³ — a tempo2-fitted par
must be evaluated under the conventions its values were fitted with.

`BinaryFacts` reports `ell1_t2`. This is not a parity program with
libstempo: it removes the known ELL1-input bias, and A.6.3 is the gate that
keeps it honest.

## A.5 Kept relaxation

`CORRECT_TROPOSPHERE Y` accepted with a warning on both timing packages (§1.2).

## A.6 Tempo2 checks (new in v2 — the evidence the design was owed)

- **A.6.1 (H7 — freeze comparison, mixed-engine-consistent files).**
  `r_pint` and `r_tempo2` are `Engine.residuals()` on two timing-package
  reads of the same observatory par/tim — the same JAX chain, two freezes
  (A.2.10). Same `binary_conventions`. Budget:

  ```
  RMS( r_pint − r_tempo2 ) ≤ 100 ns    — no relative term, no cushion
  ```

  The files MUST be ones both packages can freeze the same way. A pair is
  this gate only if all of the following hold:

  1. **Clock coverage.** Both packages apply the clock chain. A SAT before
     the observatory clock file (PINT extrapolates; tempo2 applies none) is
     not this gate. Replacement: `tests/data/sim_jump_clk`.
  2. **Shared ecliptic frame.** Equatorial, or `ECL IERS2003` so PINT uses
     tempo2's default obliquity (tempo2 has no `ECL` keyword; the pin is not
     a sky-coordinate transform). Relabelling `ECL` without transforming
     ELONG/ELAT is the H7 pin; MetaPulsar's mixed-engine path is the numeric
     transform `as_ICRS().as_ECL(ecl="IERS2003")`. Native `ECL IERS2010` is
     ~120 ns of Roemer and is not this gate.
  3. **Mixed-engine deterministic surface.** The par both packages read is
     the surface MetaPulsar writes for a pint+tempo2 stack:
     `PLANET_SHAPIRO N`, `SWM 0`, constant `NE_SW` only (no `NE_SW1` /
     `SWEPOCH`), `TIMEEPH FB90`, `T2CMETHOD IAU2000B`, `CORRECT_TROPOSPHERE
     N`, clock keyword `CLK`. `PLANET_SHAPIRO Y` puts independently
     interpolated planet positions into the freeze; PINT-only solar-wind
     derivatives and unset `TIMEEPH` are further package defaults. Native
     Vela `sim_sw` is not this gate. Replacement: `tests/data/sim_sw_aligned`.

  Recorded in `PARITY.md` (measured ~1.5 ns on all 19 rows). This is **not**
  a timing-model test and **not** `PINT.Residuals` vs libstempo. The
  load-bearing geometry check on a tempo2 read is the Roemer closure
  (A.2.3, ≤ 1 ns, measured 0.4 ps).
- **A.6.2 (H8 — `TRACK −2` is required, not permitted).** One fixture whose
  tim carries the phase connection, where the test *requires*
  `engine.pulse_number_source == "tempo2"` — and which is
  constructed so model-derived pulse numbers would differ (a deliberate
  near-half-turn ambiguity), so a silently dropped connection is ≥ half a
  period wrong, not invisible.
- **A.6.3 (S4 — the conventions flag gets a discriminating fixture).** One
  EPTA-style ELL1 fixture against libstempo on matched clocks:
  `RMS ≤ 50 ns` with `binary_conventions="tempo2"`. The fixture MUST
  discriminate: its eccentricity is large enough that the same comparison
  with `binary_conventions="pint"` **fails** the gate. The missing ELL1
  truncation is 3.6 ns at e = 2.5×10⁻⁶ and 22.9 µs at e = 1.2×10⁻³
  (A.4) — `J1227-6208` is already measured and qualifies; a near-circular
  EPTA ELL1 would pass 50 ns without the flag and prove nothing.
  The fixture is reduced to a plain ELL1 with the mixed-engine pins A.6.1
  already requires (`NE_SW 0`, `CORRECT_TROPOSPHERE N`) so the gate measures
  the ELL1 truncation rather than solar-wind or troposphere defaults.
  Deliberately loose otherwise — the first *external* constraint on A.4, not
  a parity program. If the gate needs a third tempo2 convention to pass,
  that is a deviation-ledger finding, **not a license to port more
  formBats**.

---

# Addendum B — the pulsar product

`PulsarData` is **the pulsar**: TOAs, residuals, design matrix, flags —
frozen, in one record, in the timing package's row order. It is simultaneously: (a) nltiming's
`protocols.PulsarData` + its ephemeris extras, satisfied structurally; (b) an
Enterprise pulsar in the `FeatherPulsar` sense (plain attributes with the
names Enterprise reads — no properties, no live timing object); (c) the in-memory
form of the feather file (schema v1, B.5). One record, all three roles, no
adapter code. (`PulsarData` is nltiming's type name; it is kept, not
renamed.)

## B.1 The one-source rule (R-B1)

Plainly: every array on the pulsar is frozen TOA columns, the reference
residual / design matrix, or a value read off the par.

- **R-B1.1** Every public array on `PulsarData` MUST be one of:
  - **(F)** a frozen TOA array (a `FrozenTOAs` field, or a TOA column
    captured at the same freeze: errors, flags, observatory codes, site
    arrival times), or
  - **(E)** the engine's reference evaluation: `residuals()`,
    `−residual_jacobian()` (R9.3), a field of the reference `Correction`
    at `θ★` (`delay`, `doppler`, `ssb_psr_pos`, `spin_frequency`), or the
    pre-binary barycentric snapshot of R-B1.4, or
  - **(M)** a scalar/table read off the parsed model *values* with no PINT
    computation: `name`, `dm`, the DMX window table, `pdist`, sky scalars.

  Calling any PINT/tempo2 *physics* method after the freeze
  (`get_barycentric_toas`, `barycentric_radio_freq`, `ssb_to_psb_xyz_ICRS`,
  `Residuals`, `designmatrix` — except as the R9.4 oracle) to populate a
  field is a spec violation, enforced by monkeypatch (P2).
- **R-B1.2** The v1 violations and their v2 sources, spelled out:

  | field | v1 source (violation) | v2 source |
  |---|---|---|
  | `toas` | `model.get_barycentric_toas(toas)` — second PINT pass | **(F+E)** `tdbld·86400 − delay_bary★` (R-B1.4): the barycentric arrival at PINT's pre-binary cutoff, from the chain's own reference delays. On the tempo2 timing package: tempo2's TDB minus *Vela's* pre-binary delay |
  | `freqs` | `model.barycentric_radio_freq(toas)` | **(F+E)** `freq_hz·(1 − doppler_bary★)/10⁶` MHz (R-B1.4): the solar-system Doppler at `θ★`, PINT's cutoff — not the full chain's `doppler_corrected_observing_frequency`, which includes the binary Doppler |
  | `dmx` | `{}` | **(M)** the real table (B.3.2) |
  | `flags` | per-TOA `get_flags()` walk | **(F)** columnar TOA capture at freeze |
  | `pos_t`, `residuals`, `Mmat` | engine ✔ | unchanged (E); `Mmat` per R9.3 |

- **R-B1.3** The identities are *tested as array equalities* (P1) — the
  fields the v1 test conspicuously skipped.
- **R-B1.4 (the pre-binary barycentric snapshot — the cutoff is PINT's).**
  `toas` is the **barycentric arrival**, not the corrected TOA. PINT's
  `get_barycentric_toas` — the array Enterprise writes into
  `PintPulsar.toas` — is TDB minus the delays *before the first binary
  component* (`include_last=False`, cutoff at the `pulsar_system` category):
  after solar system, solar wind and DM/DMX, **before** the binary Roemer
  and Shapiro, and before FD (PINT orders FD after the binary). Vela's
  `corrected_toa_value` subtracts *every* delay; for an ELL1/DD pulsar the
  two differ by the binary Roemer delay — **seconds** (`A1` is seconds),
  which is a wrong `toas` array by any measure and would put every TOA in a
  different ECORR epoch. The engine therefore records, during the
  reference pass, the accumulated delay and Doppler at PINT's cutoff:
  **after the dispersion stages and before the binary stage** when a binary
  is present (FD sits after the binary in `STAGE_ORDER`, so the snapshot
  excludes it), and after **all** delay stages for an isolated pulsar
  (matching PINT, whose empty cutoff sums every delay):

  ```
  toas  := tdbld·86400 − delay_bary★           # barycentric arrival, seconds
  freqs := freq_hz·(1 − doppler_bary★)/10⁶     # MHz; solar-system Doppler
  ```

  `doppler_bary★` at that point is the solar-system Doppler (solar wind and
  dispersion add none), matching PINT's `barycentric_radio_freq`; the full
  chain's Doppler additionally carries the binary term and MUST NOT be used
  here. The fully corrected TOA (`corrected_time` at `θ★` after every
  delay) is engine-internal and MUST NOT be published as `toas` — v2.0's
  first draft made exactly that mistake, and P1's PINT cross-check is what
  keeps it from coming back.

## B.2 Field inventory (R-B2)

`N` TOAs, in the timing package's order per §5.5. Sources per R-B1.1 tags.

| field | shape/type | units/semantics | src |
|---|---|---|---|
| `name` | str | PSR/PSRJ | M |
| `fitpars` | tuple[str] | engine `param_names` == `Mmat` columns | E |
| `setpars` | tuple[str] | model params not in `fitpars` | M |
| `toas` | (N,) f8 | barycentric arrival, **seconds** (MJD·86400 scale) — R-B1.4, PINT's pre-binary cutoff | F+E |
| `stoas` | (N,) f8 | site arrival, seconds (`SAT`·86400, from the timing package) | F |
| `toaerrs` | (N,) f8 | seconds | F |
| `residuals` | (N,) f8 | `engine.residuals()`, seconds, gauge-free | E |
| `freqs` | (N,) f8 | **MHz**, solar-system-Doppler corrected — R-B1.4 | F+E |
| `Mmat` | (N, n_fit) f8 | `−residual_jacobian()` — R9.3, cached per R9.6 | E |
| `flags` | dict[str, (N,) str] | columnar tim flags | F |
| `backend_flags` | (N,) str | Enterprise recipe verbatim (B.3.1) | F |
| `telescope` | (N,) str | observatory codes | F |
| `pos` | (3,) f8 | ICRS unit vector at POSEPOCH | M |
| `pos_t` | (N,3) f8 | reference `ssb_psr_pos` — the line of sight the chain used | E |
| `sunssb` | (N,6) f8 | lt-s; `[:, :3]` from frozen columns; `[:, 3:]` zeros | F |
| `planetssb` | (N,9,6) f8 | B.3.3 | F |
| `theta`, `phi` | f8 | polar/azimuth from `pos` | M |
| `pdist` | (2,) f8 | (kpc, err); PX-derived else `(1.0, 0.2)` | M |
| `dm` | f8 | par DM; `0.0` if absent — never raises (Enterprise's `AttributeError` here is a defect, not a contract) | M |
| `dmx` | dict \| None | B.3.2 | M |
| `state_id` | str | B.4 | E |

## B.3 Field semantics

### B.3.1 Flags and `backend_flags`

Captured at freeze from the timing package's columnar flag store (PINT: TOA-table
columns; tempo2: `flagvals` per name), missing values `""`. `backend_flags`
is Enterprise's recipe copied with its precedence quirk intact (`fe_be` base,
then `f, i, sys, g, group` overwrite where non-empty — last wins), cited to
`enterprise/pulsar.py:327`.

### B.3.2 `dmx`

Enterprise's shape, built from model values (no physics):

```python
{"DMX_0001": {"DMX": float, "DMXerr": float | None,
              "DMXR1": float, "DMXR2": float, "fit": bool}, ...}
```

`None` when the model has no DMX component; `fit` from the frozen/free state;
R1/R2 in MJD. The sole consumer is `WidebandTimingModel` (§III.2), which
requires `fit=True`, indexes `psr.fitpars.index(key)`, and windows on
`stoas/86400` — all of which now cohere because `fitpars`, `Mmat` and `stoas`
come from one freeze. An empty dict on a DMX pulsar is a refused state: the
real table, or `None`.

### B.3.3 `planetssb` and `sunssb`

The `(N, 9, 6)` layout is a 15-year-old accident; the *contract* implemented
is the one Enterprise's own PINT path defines and its consumers exercise
(§III.2): slots Mercury=0 … Pluto=8; only positions `[:, :, :3]` of slots
{2 Earth, 4 Jupiter, 5 Saturn, 6 Uranus, 7 Neptune} are ever read
(`utils.physical_ephem_delay`, `PhysicalEphemerisSignal`; Discovery
`solar.py` reads slot 2 and `sunssb[:, :3]` only); **velocities are read by
nothing**.

- Slots 2/4/5/6/7 positions: from the frozen columns
  (`obs_<planet>_pos + ssb_obs_pos`, whichever timing package froze them).
- **Venus (slot 1) is filled whenever the freeze carried it**
  (`PLANET_SHAPIRO` true) — we have it; NaN would be theater. Enterprise's
  PINT path leaves slot 1 NaN, which is why gate P6 *excludes* slot 1 from
  the Enterprise-PINT match: filling Venus is a superset, and nobody NaNs it
  "for parity".
- All velocities and remaining slots: NaN, matching Enterprise-PINT exactly.
- The docstring MUST carry the table of what Enterprise and Discovery
  actually read, so nobody "fixes" the NaNs.
- `sunssb[:, :3]` from `obs_sun_pos + ssb_obs_pos`; velocity half zeros.

### B.3.4 `dm` never raises

`0.0` when the par has no DM — a readable default, unlike the
`AttributeError` Enterprise's `_dm` leaves behind.

### B.3.5 Serialization is not optional

`to_feather`/`from_feather` per B.5, with lossless round-trip (P4) — the
thing Enterprise's own `save_feather` fails at today (it touches `_toas` on
an object that only has `toas`).

### B.3.6 Non-goals (the road back to `BasePulsar`, closed)

No `filter_data` (it desynchronizes `fitpars` from `Mmat` in Enterprise today
— §III.2; filtering means re-freezing, which is cheap here), no
`set_flags`, no pickle/deflate machinery, no `sort_data` (R5.5: this package
does not sort, and does not offer to), no mutable anything. `PulsarData` is frozen with read-only array views. The
consumer surface trending past ~700 lines is this rule's alarm (G7).

## B.4 `state_id` (R-B4)

A stable identifier of everything the residual depends on: the timing package,
`binary_conventions`, pulse-number source, TOA count, fitpars, exact
reference strings, stage list and the schema version. There is no row-order
component because there is no row order to record (R5.5). nltiming's context cache keys on it
(`_pulsar_state_fingerprint` prefers `state_id()` when present).

## B.5 Feather schema v1 (R-B5)

The on-disk contract, frozen exactly as Enterprise/Discovery read it today,
now with a name and a version:

```
columns:        toas stoas toaerrs residuals freqs backend_flags telescope
vector_columns: Mmat_i  sunssb_i  pos_t_i
tensor_columns: planetssb_i_j        (54 columns; NaN where B.3.3 says NaN)
flag columns:   flags_<name>
metadata json:  name dm dmx pdist pos phi theta fitpars setpars
                + schema="pulsardata-feather-v1"
                + state_id, timing_package, software="vela_jax", gauge (provenance dict)
                + reference_theta_exact (decimal strings), native_units
```

The metadata block is the quiet ecosystem win: a feather written by this
package carries everything nltiming's `LinearTimingEngine` needs (`fitpars`,
`Mmat`, exact reference strings, units, gauge provenance), so **a frozen
linear timing analysis can be rebuilt from the file alone, with no timing
package installed** — the "valid read requires only on-disk products" rule
nltiming already enforces for run manifests, extended one file to the left.
Consumers that predate the metadata ignore it; no column moved.

## B.6 `TimingPulsar` (R-B6) — composition, not inheritance

```python
class TimingPulsar:
    """One pulsar: a PulsarData record and the Engine that produced it."""
    data: PulsarData
    engine: Engine
    # the pulsar attribute names (toas, residuals, Mmat, flags, ...
    # — every B.2 field)
    # are forwarded to `data`; nothing is recomputed, nothing is copied.

    @classmethod
    def from_files(cls, par, tim, *, timing_package="pint", **engine_kwargs): ...
    @classmethod
    def from_pint(cls, model, toas, **engine_kwargs): ...  # already-built PINT pair, same as Engine.from_pint
    @classmethod
    def from_tempo2(cls, par, tim, **engine_kwargs): ...  # alias: timing_package="tempo2"
    def pint_model(self): ...
    def can_use_engines(self, engines="vela_jax", **_): ...
    def timing_engine(self, engines="vela_jax", *,
                      derivative_method="analytic",
                      nonlinear_params=None, **engine_kwargs): ...
    def state_id(self) -> str: ...
```

- **R-B6.1 (composition).** `TimingPulsar` *holds* a `PulsarData` and an
  `Engine` and forwards the pulsar attribute names. It does not inherit from the frozen
  dataclass (a frozen dataclass plus an engine plus methods fights the type:
  `__init__` order, accidental copies). The protocols are structural, so
  forwarding satisfies `isinstance` checks; a test asserts every B.2 name is
  forwarded and `is`-identical to the record's array.
- **R-B6.2** `timing_engine` returns the `VelaJaxTimingEngine` adapter (B.7).
  A hybrid `nonlinear_params` mode is executed by `Engine.perturbative`
  (R10.3), and the backend reports the executed mode. `derivative_method`
  per R10.2: both known values accepted and recorded, unknown values raise.
- **R-B6.3 (honest capability).** `can_use_engines` answers for exactly what
  this object is: a single-leg, single-timing-package vela-jax pulsar. A
  `{"pint": ..., "tempo2": ...}` mapping is honoured iff every value is
  `"vela_jax"`. Mixed-impl and multi-PTA requests belong to MetaPulsar, and
  the README MUST attribute the two-PTA results to MetaPulsar composites, not
  to this class.
- **R-B6.4 (kwargs are checked, not swallowed).** The keyword arguments
  nltiming actually passes (`tempo2_native`, `tempo2_jug_options`,
  `prime_sessions`, `verify_wiring`, `subtract_tzr`, plus the two named ones)
  are accepted; timing-package-specific ones are ignored *with their names checked* —
  an unknown kwarg raises, so a typo in a contract knob cannot silently
  no-op. `subtract_tzr=False` is asserted compatible (this engine never
  mean-subtracts; `export="none"`).

## B.7 `VelaJaxTimingEngine` and the chart-capability mirror (R-B7)

- **R-B7.1** `VelaJaxTimingEngine(engine)`: the twenty renaming lines (R10.1), plus
  `nonlinear_params` set by the caller that chose the mode (R10.3).
- **R-B7.2** The `_ChartCapability` mirror of nltiming's
  `BinaryChartCapability` stays structural (no nltiming import) but MUST
  enforce the same invariant the real dataclass does:
  `origin_certified=True` without `certification_ref` **raises** — the
  mirror may never be *weaker* than the contract it mirrors.
  `origin_certified` stays `False` until an actual origin-certification run
  is recorded; when it is, `certification_ref` names it.
- **R-B7.3** The nltiming-installed integration test asserts the real
  protocols by `isinstance`; the mandatory N1 CI job is what makes that a
  guard rather than a hope.

---

# Part III — The ecosystem, as if designed from the start

This part is proposal, not implementation, and by R14.0 it never blocks
Part II: each item is a follow-up PR in its own repo, and if every one of
them slips, the timing package is still honest.

## III.1 The target ownership table

| object | owner | JAX? | today's owner (the inversion) |
|---|---|---|---|
| frozen pulsar arrays (`PulsarData`) | the timing package (vela-jax per leg; MetaPulsar for composites; JUG/pyvela when they choose) | no | Enterprise (`PintPulsar`/`Tempo2Pulsar`), duplicated by Discovery's feather fork |
| `r(θ)`, `J`, `M` | the same object, same package | if the engine is | split: `M` from Enterprise's pulsar, `r(θ)` from a side-loaded engine |
| the interchange file | feather schema v1 = serialized `PulsarData` | — | de-facto Enterprise `save_feather`, unversioned, broken round-trip |
| protocols (`PulsarData`, `TimingPulsar`, `TimingEngine*`, gauge, charts) | nltiming (already written; already structural) | — | nltiming ✔ |
| linear-only T0 view | nltiming `LinearTimingEngine` over `PulsarData` (or a schema-v1 feather's metadata alone) | no | nltiming ✔ |
| EFAC/EQUAD/ECORR, GPs, ORFs, samplers | Discovery / Enterprise | theirs | ✔ |
| multi-PTA combination | MetaPulsar | per-leg | ✔ |
| tempo2-faithful application, GUI, fitter, live barycentric corrections | JUG | — | ✔ — and stays there; vela-jax never absorbs it |

The change is one row: the pulsar moves from the consumers to the timing
packages. Everything else is already where it belongs — which is why this is
a small redesign wearing a big thesis.

## III.2 What we learned reading the consumers (the facts that license the flip)

1. **Enterprise core + Discovery consume**: `name, toas, residuals, toaerrs,
   freqs, Mmat, fitpars, flags, backend_flags, telescope, pos, pos_t,
   planetssb[{2,4,5,6,7}, :3], sunssb[:3], dm, dmx, stoas` — and nothing
   else. `planetssb` velocities: zero readers. `setpars`, `pdist`, `theta`,
   `phi`: extensions-only.
2. **Names bind by `hasattr`** in both frameworks (`selection_func`;
   `makedelay`); in Discovery a vanished attribute silently becomes a sampled
   parameter. The schema is *the* API and must be versioned — which is what
   B.5 does.
3. **Sorting**: Enterprise `_isort` (stable, barycentric key) is applied at
   the property layer and deliberately *not* serialized. Discovery does not
   assume sorted input either — its `quantize` argsorts internally and returns
   bins in the caller's order, as Enterprise's `create_quantization_matrix`
   does. Nothing downstream requires a sorted file, so §5.5 does not produce
   one.
4. **nltiming already defines the protocols** and already treats
   `pulsar.Mmat` as the analytic source of record. The disagreement is not in
   its design; it is in what today's pulsars can supply.
5. **The consistency gate (`validate_engine_against_pulsar`) has zero call sites** —
   the clearest single symptom of the inversion: the consistency check
   exists, and cannot be turned on, because no current pulsar/engine pair
   would pass it in general.

## III.3 nltiming

nltiming's design survives contact with this redesign almost untouched — it
was built *around* a pulsar and an engine that could disagree, and removing
the disagreement removes workarounds, not architecture.

- **Turn the gate on — scoped.** Call `validate_engine_against_pulsar`
  inside `TimingSpec.for_pulsar` **for engine-emitted single-leg pulsars**
  (detectable by `state_id`/schema provenance, or simply by the engine and
  pulsar advertising the same `software`), where it is free and a hard error
  with nothing grandfathered. It MUST NOT be applied as-written to
  composites: a MetaPulsar mix of `−J` legs and linearized legs will not
  satisfy `engine.design_matrix() == pulsar.Mmat` globally, and III.4 says
  mixed `M` is allowed. The composite check is defined separately:
  **block-equality on the rows/columns of each vela-jax leg**, other-engine
  legs gated by their own existing validators.
- **Let the routes converge.** With `pulsar.Mmat == −J`,
  `derivative_method` stops selecting between two truths and becomes manifest
  metadata. Keep the knob (the manifest must describe what ran; engines
  without a residual Jacobian still differ), but document that on a
  `JaxTimingEngine`-built pulsar the routes are one matrix.
- **Withdraw `feature_binar_plus_models.md`.** Its own text names the cost:
  *"the charter line this does cross is 'nltiming is not a reimplementation
  of the timing ecosystem'."* The proposal's `HybridTimingEngine` is
  vela-jax's `PerturbativeEngine` (same formula, delta-exact rather than
  hand-linearized, certified against a full nonlinear parent it would not
  have); its `PintTimingPulsar` is `vela_jax.TimingPulsar`. The piece to
  keep is the vocabulary: `nonlinear_params` modes stay nltiming-owned and
  engine-executed, exactly as today.
- **Protocol honesty, two lines**: declare the real `timing_engine(...)`
  kwargs in the `TimingPulsar` protocol (the six the call site passes), and
  add `pos` to the `PulsarData` protocol (Discovery ORFs and
  `BasisCommonGP` read it; the protocol omits it).
- **Later, once feathers carry the B.5 metadata block**: `load_run` /
  `LinearTimingEngine` can rebuild a frozen linear analysis from the feather
  alone — that on-disk rule extended one file to the left.

## III.4 MetaPulsar

MetaPulsar is already two-thirds of a `PulsarData` writer: the Enterprise
duck surface, `to_feather`, `timing_engine()` returning composite engines,
`state_id`, exact reference-token bookkeeping. Convergence, not rebuild:

- **Emit the object.** `MetaPulsar.pulsar_data()` returning the same frozen
  record vela-jax emits per leg — concatenated rows, canonical `fitpars`,
  composite `Mmat` — so a composite and a single-leg pulsar are the same
  type downstream. The property duck becomes a thin view over it; the
  feather writer becomes `PulsarData.to_feather`.
- **Keep `sort=False`.** MetaPulsar already refuses to sort and has a test
  that builds ECORR on a shuffled pulsar; §5.5 brings vela-jax into line with
  that rather than the other way round. `_isort`/`_iisort` can become the
  identity slices they already are by default and then disappear — a
  simplification, not a reordering.
- **Composite `Mmat` is mixed by design**: a vela-jax leg contributes `−J`
  columns; other-engine legs keep their own matrices; the per-leg
  `Offset_<pta>` gauge columns and unit canonicalization stay exactly the
  MetaPulsar machinery that exists. The composite consistency check is the
  III.3 block-equality, not the global gate.
- The `vela_jax` engine name, timing-package-per-leg dispatch, and
  `PulsarJaxTimingEngine` composites are already merged in MetaPulsar and are
  the model for how any future engine family joins.

## III.5 Enterprise

Enterprise's job shrinks to what it is good at: signals and likelihoods.

- **Accept the record.** `Pulsar(...)` grows one more duck check — an object
  satisfying the `PulsarData` attribute set is taken as-is (`FeatherPulsar`
  semantics: plain attributes; Enterprise sorts at its property layer if it
  wants to, on read, as it does today). That is the entire
  "adapter": zero lines of translation, because B.2 was written against
  what Enterprise actually reads.
- **`PintPulsar`/`Tempo2Pulsar` become legacy constructors** for the classic
  workflow — kept, frozen, not extended. New pipelines get their pulsar from
  the timing side (vela-jax, MetaPulsar) or from a feather.
- **Defects worth fixing regardless** (all verified in the current tree):
  `filter_data` prunes `Mmat` columns but not `fitpars` (the misalignment
  reaches `WidebandTimingModel` and `get_coefficients`);
  `FeatherPulsar.save_feather` touches `_toas` and breaks feather→feather
  round-trips; the `"ELONG" and "ELAT" in ...` truthy-constant parse at
  three sites; `Tempo2Pulsar._get_planetssb` mutating the live tempo2 object
  (`DMASSPLANET*=0` + `formbats()`); `psr.dm` raising `AttributeError` on
  DM-less pars. Schema v1 sidesteps all five for new data; the fixes matter
  for the legacy path's remaining lifetime.
- **Stop serializing what nothing reads**: planet velocities and slots 0/3/8
  can be declared reserved-NaN in schema v1 (they already are, from the PINT
  path) rather than "data".

## III.6 Discovery

Nothing required. Discovery already consumes exactly the schema (its
`Pulsar` is the feather record, minus `telescope`), quantizes ECORR by value
rather than by adjacency so row order does not reach it, and receives
`residual_delta_jax` through nltiming's delay callable — the architecture
this whole redesign exists to feed. Optional niceties: accept an in-memory
`PulsarData` without the file hop; keep the documented
enterprise-vs-discovery ECORR degree-of-freedom difference documented.

## III.7 JUG, Vela.jl/pyvela, PINT

- **JUG** keeps its job: tempo2-native `r(θ)`, the fitter, the GUI, and the
  live barycentric corrections — recomputed as the sky position moves during
  a fit — the cases where the emission epoch must move. It never becomes a
  vela-jax dependency and vice versa. If JUG one day emits
  `(PulsarData, TimingEngine)`, it plugs into the same consumers with no new
  seams — that is the test that the contract is right, not a work item.
- **Vela.jl / pyvela**: the §16 asks stand. The new one: `SPNTA` could emit
  `PulsarData` too — pyvela already holds every field — making Vela.jl a
  first-class writer for Julia-side workflows without touching the JAX
  port. The component-level JSON oracle tables remain the single
  highest-leverage item anyone in this ecosystem can build.
- **PINT** stays the default timing package and the model-semantics authority. Its design
  matrix is demoted from product to oracle here (R9.4); upstreaming the TZR
  and delay-feedback terms into PINT's `designmatrix` would close the T9 gap
  at the source and is worth an issue, not a fork.

## III.8 What this buys, concretely

- `validate_engine_against_pulsar` goes from zero call sites to always-on
  where it can be exact, with a defined composite variant where it cannot.
- The Enterprise- and Discovery-likelihood timing blocks marginalize the
  *tangent of the residual they sample* — `M = −J` exactly — instead of a
  matrix 10⁻³ away in phase with a different gauge story.
- A feather file becomes a versioned, self-describing product that can
  rebuild its own frozen linear analysis.
- One pulsar object type from a single-leg file, a two-PTA composite, or (in
  time) any other timing backend — and the "does the engine match the
  pulsar?" question stops being a question, because there is nothing left to
  match.

---

# Appendix — disposition of the reviews

## A. The v1 prototype review (structural)

| demand | disposition |
|---|---|
| delete/write REVIEW.md, CONFORMANCE.md | they exist; kept and extended (V2-P6) |
| `toas`/`freqs` from the freeze, not `get_barycentric_toas` | **accepted** — R-B1.2, gate P1 |
| fill `dmx` or drop it | **accepted** — B.3.2 (filled), gate P5 |
| sort policy, pick one and test it | **accepted, then reversed** — the policy is *no* sorting (R5.5, gate P3). Sorting at the freeze was v2.0's answer and it was wrong: neither consumer needs it (both quantize by value), and it grows a permutation protocol across a package boundary that is the identity on most real files and therefore untestable end to end |
| tighten PINT–tempo2 gate to a clock floor; drop the 10 µs cushion | **accepted** — A.6.1, gate H7; freeze comparison on mixed-engine-consistent files; ECL pinned to IERS2003 so PINT uses tempo2's obliquity |
| one required `TRACK −2` fixture | **accepted** — A.6.2, gate H8 |
| fp32 build-time reference channel, then Discovery's kernel | **accepted** — R11.3 rule 1 (fp64 reference channel, computed once at compile), gate T15; the Discovery-kernel smoke (T16) follows it |
| don't grow `TimingPulsar` toward `PintPulsar` | **accepted structurally** — B.3.6 bans the unread surface; what *is* implemented is what the consumers measurably read, which is small |
| don't import JUG for `nonlinear_params` | already true; the perturbative engine is the executor (R10.3) |
| don't port more formBats for S4 | **accepted with one gate** — A.6.3 constrains the existing flag; a failure is a ledger entry, not a porting license |
| don't make vela-jax the multi-PTA combiner | **accepted** — R-B6.3; README attribution fixed (V2-P6) |
| `derivative_method` accepted-and-ignored | **dissolved** — R10.2/R9.3: the routes coincide; unknown values raise |
| `_ChartCapability` weaker than the real dataclass | **accepted** — R-B7.2 invariant + mandatory N1 CI job |
| "no Venus" / "ghost docs" | **rebutted with citations** — §I.1; Venus is frozen and now slotted when present (B.3.3) |

## B. The v2 draft review (packaging and precision)

| point | disposition |
|---|---|
| 1. two documents is a defect | **accepted** — this document is self-contained; the v0.2 draft is withdrawn; v1's Part II numbering is preserved so `docs/*` anchors (§7.2, §11.3 rule 1, §16.x, A.4, R3.x…) stay valid |
| 2. `TimingPulsar(PulsarData)` fights the type | **accepted** — composition, R-B6.1: holds `data` + `engine`, forwards the pulsar attribute names, `is`-identity tested |
| 3. `M = −J` construction cost unnamed | **accepted** — R9.6: `J` computed once and cached, cost named; T0-only consumers use feather + `LinearTimingEngine`; substituting PINT `M` back is a named spec violation |
| 4. composite gate underspecified | **accepted** — III.3/III.4: the always-on gate is scoped to engine-emitted single-leg pulsars; composites get block-equality on vela-jax legs |
| 5. multi-site sort equivalence overstated | **superseded** — the equivalence argument is moot now that §5.5 sorts nothing at all; there are no two orders to be equivalent |
| 6. Part III must not block Part II | **accepted as a rule** — R14.0 shipping rule |
| 7. Venus vs P6 Enterprise-PINT match | **accepted** — P6 explicitly excludes slot 1; B.3.3 says fill it and why |

## C. The merged-spec review (physics and dialect)

| point | disposition |
|---|---|
| `toas` identity is the corrected TOA, not the barycentric arrival | **accepted — the load-bearing fix.** R-B1.4 defines the pre-binary barycentric snapshot at PINT's `get_barycentric_toas` cutoff; B.1.2, B.2 and gate P1 updated, with a PINT cross-check on a binary fixture so a wrong cutoff (seconds) can never pass; `freqs` is cut at the same point (solar-system Doppler, not the full chain's) |
| H7 could be read as a timing-model test | **accepted** — A.6.1 / A.2.10: `Engine.residuals()` on two freezes of mixed-engine-consistent files (clock coverage, shared ecliptic frame, `PLANET_SHAPIRO N`, no PINT-only solar-wind derivatives); the Roemer closure (A.2.3) is the load-bearing geometry check |
| S4's fixture could pass without the A.4 flag | **accepted** — A.6.3 requires a discriminating eccentricity: the gate MUST fail under `"pint"` conventions and pass under `"tempo2"`; `J1227-6208` (22.9 µs) qualifies |
| README: a residuals-only look uses `Engine` | **accepted** — R9.6 and V2-P6 |
| Part I lost the Vela-facing introduction | **accepted** — I.0 added in Vela's nouns; this document is the full version |
| G7 will be pressed by Addendum B | **accepted** — G7 restated as a smell threshold; B.3.6's ~700-line consumer alarm is the operative budget |
| two meanings of "host"; "trace" in prose | **accepted** — "host" is retired; the terms paragraph uses *timing package* (who reads par/tim) and *build time* (longdouble reductions); "trace" is kept to the JAX rules |
| Vela's names first (`TOACorrection`, `corrected_toa_value`, `doppler_corrected_observing_frequency`, `is_barycentered`, `Component`) | **accepted** — aliases introduced at first use (§5.4, §7, §7.1) and a glossary added in Vela/PINT nouns, with the three arrival times distinguished |
| CS register (ABI, dyad, impl token, constant-fold, victim, three hats, arrow flip, charter, live BCLT) | **accepted** — swapped for timing language throughout Part I, Addendum B and Part III; "schema v1" kept as the on-disk version name; `PulsarData` kept as nltiming's type name and described as *the pulsar* |
| N1 CI job / `CORRECT_TROPOSPHERE` / H8 naming | already as reviewed — unchanged |
