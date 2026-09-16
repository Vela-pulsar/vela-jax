# vela-jax

**Vela.jl's timing-delay engine, in JAX.** One pulsar's par and tim in; a
`jit`-able, `jacfwd`-able residual `r(θ)` in seconds out, plus a frozen pulsar
record that Enterprise, Discovery and nltiming read directly. A gradient-based
sampler (NumPyro NUTS inside Discovery, or PTMCMC through Enterprise) can then
sample nonlinear timing parameters jointly with the noise and GW model, against
a design matrix that is the tangent of the residual it samples.

> **Status.** Implements [`SPEC.md`](SPEC.md) v2.5; where this README and the
> spec disagree, the spec wins. Reviewers start at [`docs/REVIEW.md`](docs/REVIEW.md).
>
> **float64 is required.** Set `JAX_ENABLE_X64=1`, or run
> `jax.config.update("jax_enable_x64", True)` before any JAX array exists.
> The engine refuses to build otherwise.

```python
import jax; jax.config.update("jax_enable_x64", True)
from vela_jax import Engine

engine = Engine.from_files("J1909-3744.par", "J1909-3744.tim")
engine.param_names               # free PINT parameters, in PINT order
r = engine.residuals()           # r(θ★), seconds, gauge-free
d = engine.residual_delta(δ)     # r(θ★+δ) - r(θ★); δ in PINT units
J = engine.residual_jacobian()   # jacfwd of the same function, computed once
M = engine.design_matrix()       # -J: the design matrix in fitter sign
```

`Engine` is the residual. `TimingPulsar` is the frozen pulsar record built on
top of it, and building it computes the Jacobian (an `N × n_par` forward-mode
pass with its own compile). Use `Engine` when you want residuals and
`TimingPulsar` when you want the pulsar.

```python
from vela_jax import TimingPulsar
import nltiming

psr = TimingPulsar.from_files("psr.par", "psr.tim", timing_package="tempo2",
                              binary_conventions="tempo2")
timing = nltiming.TimingSpec(engines="vela_jax").for_pulsar(psr)
psr.to_feather("psr.feather")    # schema v1; Discovery and Enterprise read it
```

A complete walkthrough from par/tim to a timing corner plot is in
[`examples/sampling_timing_parameters.ipynb`](examples/sampling_timing_parameters.ipynb).

## Install

This repository is private, and most of the stack around it is unreleased
work on branches. PyPI releases will not reproduce anything below.

```bash
pip install -e .                       # engine + PINT as the timing package; pulls psrdata
pip install -e '.[dev,oracle,tempo2]'  # everything the test tiers need
```

Every other package is installed from a checkout with `--no-deps`, so that pip
does not replace a required branch with a release:

| Package | Repository | Branch | Needed for |
|---|---|---|---|
| **vela-jax** | `vhaasteren/vela-jax` (private) | `main` | this package |
| **psrdata** | `nanograv/psrdata` | `main` | the pulsar record and feather schema; a declared dependency, pip installs it |
| **PINT** | `vhaasteren/PINT` | `metapulsar` | the default timing package: FDJUMPDM sign fix, a longdouble fix, Jodrell MkII clock chains |
| libstempo | `vhaasteren/libstempo` | `feat/vela-jax` | the tempo2 timing package (`Engine.from_tempo2`): exposes `siteVel`, `correction_tt`, `correction_tt_tb` |
| tempo2 | | | the C library and `$TEMPO2` runtime, for a tempo2 read |
| nltiming | `vhaasteren/nltiming` | `main` | sampling timing parameters: `TimingSpec`, priors, charts, the samplers |
| Discovery | `vhaasteren/discovery` | `feat/class-tracking` | the NUTS path: `transport.class_tracking` and the `origin=` keyword |
| MetaPulsar | `vhaasteren/metapulsar` | `main` | several PTA datasets in one timing model, one vela-jax leg per PTA |
| Vela.jl | `vhaasteren/Vela.jl` | `vela-jax` | tests only: `pyvela` as the parity oracle, and the fixture par/tim files |

```bash
git clone -b metapulsar          git@github.com:vhaasteren/PINT.git
git clone -b feat/vela-jax       git@github.com:vhaasteren/libstempo.git   # tempo2 timing package only
git clone -b feat/class-tracking git@github.com:vhaasteren/discovery.git
git clone                        git@github.com:vhaasteren/nltiming.git
git clone                        git@github.com:vhaasteren/metapulsar.git
pip install -e PINT -e discovery -e nltiming -e metapulsar --no-deps
pip install -e libstempo                                     # needs tempo2 headers and libs
```

`psrdata` is pinned to a moving `main`; pin a commit for a reproducible
environment. Discovery's decentering chart lives on its metamath kernel path,
so call `ds.config(kernels="metamath")` before building a model.

A stock libstempo fails loudly on a tempo2 read, by design:

```
Tempo2Error: this libstempo does not expose `siteVel`, which is the observatory
velocity. ... Install a libstempo that has it; vela-jax will not guess these.
```

tempo2 forms the observatory velocity as `earth_ssb[3:6] + siteVel`. Reading
the zero half of `observatory_earth` instead drops the Earth's rotation, 1.3%
of the observatory velocity, invisible in a Roemer closure check and about
100 ns of dispersion delay through the Doppler-corrected frequency. Every
substitute is wrong in a way that produces plausible numbers, so the engine
raises.

### Tests

```bash
make fast     #  11 s   37 tests: no par/tim read at all
make test     # 121 s  247 tests: behaviour, one fixture per binary family
make full     # 277 s  343 tests: every fixture, the Vela.jl oracle, tempo2
make oracle   # parity against pyvela   (oracle extra + Julia)
make tempo2   # the tempo2 checks       (tempo2 extra + tempo2)
make check    # black + ruff + fast
```

The tiers are markers, not directories: `slow` means breadth, never a looser
budget. Fixtures are Vela.jl's own `pyvela/examples`; point `VELA_JAX_FIXTURES`
at a copy or keep a Vela.jl checkout beside this one. Every test skips cleanly
without its optional dependency. The nltiming integration is a mandatory CI
job: this package implements nltiming's protocols structurally, without
importing them, so nothing else would notice the shape drifting.

## Documents

| | |
|---|---|
| [`SPEC.md`](SPEC.md) | the normative design, Part I for humans and Part II down to code |
| [`docs/REVIEW.md`](docs/REVIEW.md) | start here to review: how to read the port, where it differs from Vela.jl, open questions |
| [`docs/CONFORMANCE.md`](docs/CONFORMANCE.md) | Vela file to module map, every spec requirement and gate, the test inventory |
| [`docs/PARITY.md`](docs/PARITY.md) | measured agreement with Vela.jl, per fixture, reproducible |

## What owns what

| | owns |
|---|---|
| **PINT** or **tempo2** | par/tim, clock corrections, TDB, JPL ephemeris interpolation, pulse numbers |
| **PINT** | the timing-model meaning (BINARY family, masks, units, `PHOFF`), and the design-matrix oracle |
| **vela-jax** | the deterministic delay chain and the residual, as a JAX function |
| **Discovery / Enterprise** | EFAC/EQUAD/ECORR and every red-noise and DM Gaussian process |
| **nltiming** | priors, coordinate charts, whitening, likelihood assembly |

The JAX graph never calls PINT, astropy or erfa; it sees frozen arrays only
(`tau`, `phi_ref`, `spin_coeffs`, `ssb_obs_pos/vel`, `obs_sun_pos`, planet
positions, frequencies, masks). A test monkeypatches PINT to raise and runs a
jitted evaluation.

Noise lines are stripped from the par before either timing package sees it, so
PINT never builds `EcorrNoise` and nothing permutes the TOAs. GP delay
components (`WaveX`, `PL*NoiseGP`, ...) are refused, not ignored: Discovery owns
those bases.

## Physics

Every component is a translation of one Vela.jl file, named in the module
docstring ([the full map](docs/CONFORMANCE.md#vela-source-map)). Each is a pure
function `(frozen, correction, params) -> correction`, threaded through a fixed
tuple in `pyvela.model.pint_components_to_vela` order:

```
solar_system -> solar_wind -> dispersion_taylor -> dispersion_piecewise
             -> binary.{ELL1,ELL1H,ELL1k,DD,DDH,DDS,DDK}
             -> frequency_dependent -> frequency_dependent_jump
             -> spindown -> phase_offset -> phase_jump
```

Then `r = (ψ - ψ_tzr) / F_spin`, with the TZR pseudo-TOA as row `R-1` of the
same arrays and no mean removed. The divisor is the pulsar-frame spin
frequency, as PINT (`calctype="taylor"`) and tempo2 use. Vela divides by the
Doppler-shifted frequency instead, which differs by `r·(v/c)`; this is the one
deliberate departure from Vela (SPEC §8, G2).

Where Vela and PINT disagree, Vela wins and the difference is gated per
fixture. Refusals (`UnsupportedModelError`) name the offender and the supported
set: BT/BTX/DDGR/unresolved T2, wideband, glitches, chromatic components,
`FDJUMPLOG N`, overlapping DMX, a fitted JUMP selecting no TOA, DDK without
astrometry or with `K96 N`, and every GP delay.

## Precision

Two quantities are too large for float64. Both are reduced once, at build
time, in numpy `longdouble`. This is the part of the port that is not a
transcription.

**The spin phase.** `F0·(t - PEPOCH)` is about 3×10¹⁰ turns, whose float64
ulp is 20 ns of residual. Vela carries it in `Double64`; JAX has no such type.
The build evaluates the reference spin series at the undelayed TDB time and
hands the trace its Taylor expansion in the delay:

```
phi_ref = φ(τ) - N - φ(τ_tzr)                       # longdouble, then float64
ψ       = phi_ref + c₁ξ + c₂ξ²/2 + ...  +  Σ δF_k t^(k+1)/(k+1)!
```

with `ξ = -delay` (a few thousand seconds at most) and `c_m = φ⁽ᵐ⁾(τ)`. Nothing
large is multiplied inside the trace, and the residual stays exact for a fast
spinning-down pulsar: on the `sim_dd` fixture (`F1 = -4.2×10⁻⁸ Hz/s`) the
remainder is 0.12 ns, which is Vela's own float64. `residual_delta(0)` is
exactly zero, because `r(θ★)` and `r(θ★+δ)` are the same graph.

**The orbital phase.** `2π(t - T0)/PB` unreduced carries a 0.4 ps sawtooth at
10⁴ orbits. The build divides out an exact integer orbit count and keeps it, so
a live `PB` stays exact: `Φ = 2π[Δt_red/PB - n·δPB/PB]`. The integer comes back
in one place, the unwrapped true anomaly that drives `OMDOT`. Missing it was a
2.7 ms error on `J0955-6150`; it is now 2.4 ps.

Parameters enter as deltas in PINT units. `reference_theta_exact()` returns
decimal strings, and `precision_critical_params()` names the axes (`F0`, `T0`,
`TASC`, `PB`, `FB0`, the epochs) whose absolute value a caller must keep as
text and perturb additively.

## Single precision

`engine.perturbative(mode, dtype=jnp.float32)` gives a float32-capable engine
over a restricted live set (binary family, sky, proper motion, parallax), with
every other axis served by the fp64-baked design matrix:

```
Δr = -M·δ_lin - (ΔD - (F_tzr/F_i)·ΔD_tzr)
```

The full residual cannot run in float32 (the Roemer delay is 500 s and float32
resolves it to 3×10⁻⁵ s), but every delta can, provided no absolute quantity
is ever formed. [`perturbative/dual.py`](src/vela_jax/perturbative/dual.py) is
a value type carrying `(reference, perturbation)` whose every operation is the
cancellation-free difference identity (`sin` by half-angle, `log` by `log1p`,
`sqrt` by `1/(1+√(1+ε))`, `atan2` as an angle difference, Kepler solved in the
difference variable). The component chain is written against the small
[`numerics`](src/vela_jax/numerics.py) interface, so the perturbative engine
reuses the same physics code; there is no second copy to keep in sync.

float32 timing is optional. A PTA likelihood is dominated by the matrix work,
not the `O(n_TOA)` residual, so the recommended path is the float64 engine plus
a cast of `r` and `M` at the likelihood boundary. The perturbative engine
exists for a Discovery kernel whose working dtype is already float32.
`engine.perturbative(...).certify()` compares against the fp64 parent on
posterior-scale deltas and reports the worst case
([`docs/PARITY.md`](docs/PARITY.md#the-perturbative-engines-validity-domain)).

## Timing packages: PINT or tempo2

PINT is the default reader. For an EPTA/IPTA product it is often the wrong
one: the file needs tempo2's `INCLUDE` handling, its clock chain, its
`TRACK -2` pulse numbers. `Engine.from_tempo2` lets tempo2 do that job and only
that job: it freezes tempo2's arrays and runs the same component chain over
them.

```python
engine = Engine.from_tempo2("psr.par", "psr.tim", binary_conventions="tempo2")
engine.timing_package, engine.source_units    # ("tempo2", "TCB"), or "TDB" if it said so
```

A tempo2 par is TCB unless it says otherwise (no `UNITS` line means TCB, and
`UNITS SI` is tempo2's spelling of it), so the text is normalised to TDB with
`tempo2 -gr transform ... tdb` before either code parses it, including before
tempo2 itself runs, so that its own `correction_tt_tb` is TT to TDB. One par
text, one timescale, everywhere. The rules and the `NE_SW` sanitation are
MetaPulsar's; the code is local, because vela-jax never imports MetaPulsar.

The load-bearing check is one line of geometry: in tempo2's own frame, before
any rotation, `psrPos·R - ½·PX·R⊥²` must reproduce tempo2's own `roemer`. It
does, to 0.4 ps or better on every fixture tested, and the build raises if it
does not.

Three things are deliberately not tempo2's, each documented at its call site:
the observing frequency handed to the chain is topocentric (Vela applies its
own Doppler shift); the TZR pseudo-TOA's geometry is PINT's (one constant
phase, which `PHOFF` absorbs); and pulse numbers come from tempo2 only when
tempo2 was told the phase connection, `TRACK -2` and `-pn` flags both,
re-referenced onto PINT's fiducial. Without that, the model defines them. A
`-pn` flag on its own is decoration: `formResiduals.C` reads the flags only
inside the `TRACK -2` branch, and bumping a `-pn` value without `TRACK` changes
tempo2's residuals by nothing at all.

Two read details are normalised so both codes read the same numbers: PINT
writes `FDnJUMP` where tempo2 writes `FDJUMPn`, and PINT writes `CLOCK` where
tempo2 reads only `CLK`. Each is respelled in the other code's copy of the par
text. The second is not cosmetic: an unpinned clock chain was 234 ns of PINT
versus tempo2 residual difference.

## Conventions: whose binary inputs

`binary_conventions` is an orthogonal flag. `"pint"` (the default) is Vela's
own. `"tempo2"` uses tempo2's `ELL1model.C` Roemer truncation, first order in
the Laplace-Lagrange parameters, no harmonics in either derivative. That is
three functions in [`binary/ell1.py`](src/vela_jax/binary/ell1.py).

It is resolved at build time into a different closure, never a traced
predicate. It is not a parity program with libstempo: it removes the known
ELL1-input bias so that a tempo2-fitted ELL1 par is evaluated under the
conventions its values were fitted with.

## The pulsar

`Engine` is the backend. `PulsarData` is the pulsar: TOAs, residuals, design
matrix and flags, frozen, in one record. It is nltiming's `PulsarData`
protocol, an Enterprise pulsar in the `FeatherPulsar` sense (plain attributes,
no live timing object), and the in-memory form of the feather file, with no
adapter between them.

**One source.** Every array on the record is a frozen TOA column, the engine's
own reference evaluation, or a value read off the par, never a second pass
over PINT or tempo2. The PINT methods that could do it (`get_barycentric_toas`,
`barycentric_radio_freq`, `ssb_to_psb_xyz_ICRS`, `designmatrix`, `Residuals`)
are monkeypatched to raise while the record is built. `toas` is the
barycentric arrival at PINT's pre-binary cutoff, reconstructed from the chain's
own delays and cross-checked against PINT to 3 ps; `freqs` is cut at the same
place, so it carries the solar-system Doppler and not the binary's; `pos_t` is
the line of sight the chain actually evaluated.

**One row order.** Row `i` of every array is row `i` of PINT's or tempo2's TOA
table. This package does not sort, filter or otherwise permute TOAs, and
publishes no permutation. Discovery's `quantize` and Enterprise's
`create_quantization_matrix` each argsort internally and write ECORR bin
membership back in the caller's row order, so neither needs sorted input. A
consumer that wants time order sorts when it reads.

**One matrix.** `Mmat` is `-residual_jacobian()`: the tangent of the residual
the sampler moves, TZR-aware and delay-feedback-aware, gated column by column
against finite differences. PINT's analytic matrix is kept as an oracle
(`design_matrix(source="pint")`) and compared in phase. nltiming's
`"analytic"` and `"autodiff"` routes are the same matrix here, and
`validate_engine_against_pulsar` passes by construction.

`psr.timing_engine(...)` returns a [`VelaJaxTimingEngine`](src/vela_jax/backend.py):
twenty lines that rename `param_names` to `fitpars` and `param_units` to
`native_units`. Neither module imports nltiming; the protocols are structural
and `runtime_checkable`. `nonlinear_params="binary"` is not a stub on this
path: the perturbative engine is that residual formula, in float64 it
reproduces the full engine to about 10⁻¹³ s, and the backend reports the mode
it executed.

**One pulsar, one timing package, one leg.** `can_use_engines` answers for
exactly that. Combining several PTAs' data for one pulsar is
[MetaPulsar](https://github.com/vhaasteren/metapulsar)'s job, one vela-jax leg
per PTA, each choosing its timing package from its own `timing_package`.

## Measured

Full tables, with the command that regenerates them, in
[`docs/PARITY.md`](docs/PARITY.md). Against Vela.jl through pyvela, over the 19
supported fixtures in `pyvela/examples`:

| | budget (SPEC §12) | measured |
|---|---|---|
| T4 `‖r - r_Vela‖∞` | RMS ≤ 1 ns, max ≤ 10 ns | ≤ 124 ps, median 1.7 ps |
| T7 `‖Δr - Δr_Vela‖∞` at ±1σ | RMS ≤ 1 ns | ≤ 105 ps, median 2.0 ps |
| T14 perturbative fp64 | ≤ 10⁻¹² s | ≤ 5.0×10⁻¹³ s, all binaries |
| T15 perturbative fp32 | rtol 10⁻⁵ | passes on all binaries, `J2302+4442` included |
| M1 `-Mmat` vs finite differences | 10⁻⁶ relative | ≤ 7.5×10⁻⁷, every column |
| H7 PINT vs tempo2 clock floor | RMS ≤ 100 ns | ~1.5 ns on all 19; solar-wind row 1.20 ns |
| S4 libstempo, discriminating ELL1 | ≤ 50 ns under `"tempo2"` | 3.3 ns, and 12 072 ns under `"pint"`, as it must be |

On real EPTA DR1 and NANOGrav 9-year data for `J1853+1303`, NUTS through
Discovery's likelihood recovers every sampled timing parameter within 1.7σ of
the par value with zero divergences, on both timing packages and on the
two-PTA combination ([`examples/nuts_j1853.py`](examples/nuts_j1853.py)).

## Known gaps

* **The perturbative assembly is first order in ΔD.** Exact for the delay
  itself, but it uses the reference Doppler factor; the dropped term is about
  4×10⁻⁴·(Δr)², invisible below a millisecond of residual change
  ([`docs/PARITY.md`](docs/PARITY.md#the-perturbative-engines-validity-domain)).
* **No component-level oracle tables.** SPEC §12 T5/T6 asked for JSON tables
  exported from `Vela.jl/test`; parity is gated end to end only
  ([`docs/REVIEW.md`](docs/REVIEW.md#what-needs-discussion)).
* **The inert-parameter list is a judgement.** Every parameter a par sets must
  be consumed, known-inert, or refused by name (SPEC R5.3b). `RM`, `NHARMS`,
  `DMDATA`, the DispersionDMX info line `DMX`, and the read-side settings
  (`TIMEEPH`, `T2CMETHOD`, `DILATEFREQ`) are classified inert because PINT
  applies them before the freeze or they are not a delay; that reasoning is
  worth checking. `TIMEEPH`/`T2CMETHOD` are inert, not pinned: the host
  already consumed them.
* **ELL1H's Shapiro harmonics differ from tempo2's.** On `J1227-6208` the
  libstempo comparison sits at 1.7 µs under `"tempo2"` conventions, from the
  orthometric Shapiro parametrisation rather than the Roemer truncation. SPEC
  §16.6 asks Vela.jl whether tempo2's ELL1 truncation should apply to ELL1H.
* **PINT's matrix is an oracle, and disagrees where it should.** It ignores
  the TZR row's own parameter dependence and the feedback of one delay into
  later components, and divides by the constant `F0`. T9 compares in phase,
  modulo a constant, at 10⁻³.
* **The gauge direction is declared.** nltiming's gauge check asks that the
  named gauge column span the direction an unmeasurable phase offset moves.
  The residual divisor here is the pulsar-frame spin Taylor series `F(t)`,
  the same one PINT and tempo2 use (§8, G2), and the design matrix is `-J`
  of that residual, so the `PHOFF` column is `1/F(t_i)`. PINT's and tempo2's
  matrices divide by the constant `F0` instead, so their column is exactly
  constant. The two differ by `F1·T/F0`, about 2e-9 on J0613-0200 and 4e-8
  on B1937+21 over twenty years, either side of nltiming's 1e-8 tolerance.
  The backend declares `gauge_direction()` so nltiming tests the exact
  column rather than the constant. (Vela.jl's topocentric divisor, a part in
  1e4, is not used here.)
* Not yet ported: wideband TOAs, `SolarWindDispersionX`, `FDJumpDM`,
  `DispersionJump`, chromatic components, glitches, troposphere. A par with
  `CORRECT_TROPOSPHERE Y` is accepted with a warning rather than refused: the
  delay is about 10 ns and near-constant, it cancels out of every residual
  difference, and Vela does not model it either.

## Layout

```
src/vela_jax/
  engine.py          Engine: build, residuals, Jacobian, design matrix, facts
  pipeline.py        the stage tuple
  freeze.py          PINT reads, and the freeze both timing packages share
  read_tempo2.py     tempo2 reads: columns, closure, pulse numbers
  tcb.py             UNITS rules and the TCB->TDB transform
  precision.py       the two longdouble build-time reductions (SPEC §4)
  params.py          ParamLayout: PINT-unit deltas -> internal-unit values
  numerics.py        the overloadable algebra the physics is written against
  correction.py      Vela's TOACorrection state
  taylor.py          Vela's factorial-convention Taylor series
  units.py           PINT units -> Vela's [T^n] system (pyvela port)
  constants.py       Vela's constants, verbatim, with the source line
  astrometry.py solarwind.py dispersion.py frequency_dependent.py
  spindown.py phase.py                            one Vela .jl file each
  binary/            orbit.py dd.py ell1.py facts.py
  perturbative/      dual.py certify.py           the float32 story
  pulsar_data.py     PulsarData: the record, and feather schema v1
  backend.py pulsar.py                            the consumer surface
tests/               three tiers, see above
examples/            sampling_timing_parameters.ipynb  nuts_j1853.py  nuts_fp32.py  parity_report.py
```

## License

GPL-3.0-or-later, see [LICENSE](LICENSE). vela-jax is a translation of GPL-3
Vela.jl code; this is the license its author requested.

## Citing

Please cite the first Vela.jl paper:

Abhimanyu Susobhanan, 2025, *Bayesian Pulsar Timing and Noise Analysis with
Vela.jl: An Overview*, ApJ 980, 165,
https://doi.org/10.3847/1538-4357/adaaec
([arXiv:2412.15858](https://arxiv.org/abs/2412.15858)).

A machine-readable record is in [`CITATION.cff`](CITATION.cff).
