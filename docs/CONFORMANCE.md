# Conformance

What [`SPEC.md`](../SPEC.md) asks for, and where it is. Three tables: the Vela
source map, the normative requirements, and the test gates. A fourth lists the
refusals, and a fifth the test inventory.

Legend: **done** — implemented as specified. **changed** — implemented
differently, deliberately; the reason is in the row and in
[`REVIEW.md`](REVIEW.md#where-this-differs-from-velajl). **open** — not implemented.

---

## Vela source map

Every physics module names its Vela.jl source in its own docstring. The whole
map in one place:

| Vela.jl | vela-jax | notes |
|---|---|---|
| `src/toa/toa.jl` (`TOACorrection`) | [`correction.py`](../src/vela_jax/correction.py) | `efac`/`equad2` fields dropped: noise is Discovery's |
| `src/residuals/residuals.jl` | [`pipeline.py`](../src/vela_jax/pipeline.py) `form_residuals` | TZR is row `R−1` of the same arrays |
| `src/model/solarsystem.jl` | [`astrometry.py`](../src/vela_jax/astrometry.py) | the `iszero(pm)` short-circuit is dropped; `ecliptic_to_icrs` / `icrs_to_ecliptic` are `ecliptic_to_equatorial` / `equatorial_to_ecliptic` with the par's `ECL`, not Vela's hard-coded `OBL` |
| `src/model/solarwind.jl` | [`solarwind.py`](../src/vela_jax/solarwind.py) | `SolarWindDispersion` only; `SolarWindDispersionX` refused |
| `src/model/dispersion.jl`, `component.jl` | [`dispersion.py`](../src/vela_jax/dispersion.py) | plus the infinite-frequency guard, below |
| `src/model/frequency_dependent.jl` | [`frequency_dependent.py`](../src/vela_jax/frequency_dependent.py) | barycentric frequency, as Vela |
| `src/model/binary/orbit.jl` | [`binary/orbit.py`](../src/vela_jax/binary/orbit.py) | Mikkola with "substitute, then select" for the traced branches |
| `binary_dd_base.jl`, `binary_dd.jl`, `binary_ddh.jl`, `binary_dds.jl`, `binary_ddk.jl` | [`binary/dd.py`](../src/vela_jax/binary/dd.py) | one `DDState`, `shapiro_params` dispatched at build; ecliptic DDK rotates ICRS vectors into KOM's frame before `I0`/`J0` |
| `binary_ell1_base.jl`, `binary_ell1.jl`, `binary_ell1h.jl`, `binary_ell1k.jl` | [`binary/ell1.py`](../src/vela_jax/binary/ell1.py) | the three polynomials transcribed term by term |
| `src/model/spindown.jl` | [`spindown.py`](../src/vela_jax/spindown.py) | **changed**: the `F_`/`F0` Double64 split becomes a full build-time longdouble reduction |
| `src/model/phase_offset.jl`, `src/model/jump.jl` | [`phase.py`](../src/vela_jax/phase.py) | JUMP × constant `F0`, as Vela |
| `GeometricUnits` `taylor_horner*` | [`taylor.py`](../src/vela_jax/taylor.py) | plus `factorial_series` for the spin tail |
| `solarsystem.jl` constants, `frequency_dependent.jl` `νref` | [`constants.py`](../src/vela_jax/constants.py) | copied verbatim with the source line |
| `pyvela/model.py` `pint_components_to_vela` | [`pipeline.py`](../src/vela_jax/pipeline.py) `STAGE_ORDER` | |
| `pyvela/model.py` `fix_params` (delay part) | [`freeze.py`](../src/vela_jax/freeze.py) `prepare_model` | |
| `pyvela/model.py` `read_mask`, `is_exclusive_mask` | [`freeze.py`](../src/vela_jax/freeze.py) | |
| `pyvela/dmx.py` `get_dmx_mask` | [`freeze.py`](../src/vela_jax/freeze.py) `dmx_index` | |
| `pyvela/toas.py` `pint_toa_to_vela` | [`freeze.py`](../src/vela_jax/freeze.py) `freeze` | vectorised over rows |
| `pyvela/parameters.py` `get_scale_factor`, `get_unit_conversion_factor` | [`units.py`](../src/vela_jax/units.py) | ported with the Julia bridge removed |
| — | [`numerics.py`](../src/vela_jax/numerics.py), [`perturbative/`](../src/vela_jax/perturbative/) | no Vela counterpart; see §11 below |
| — | [`read_tempo2.py`](../src/vela_jax/read_tempo2.py), [`tcb.py`](../src/vela_jax/tcb.py) | SPEC Addendum A |
| — | [`backend.py`](../src/vela_jax/backend.py), [`pulsar_data.py`](../src/vela_jax/pulsar_data.py), [`pulsar.py`](../src/vela_jax/pulsar.py) | SPEC Addendum B; 517 lines together, the whole consumer surface. The record *type* and the feather schema are not here: they are `psrdata`, shared with MetaPulsar and nltiming, so a feather written by this package is not its private format. |

**Forbidden sources honoured.** No code, and no formula, is taken from
`jug/delays/combined.py`, `jug/fitting/derivatives_*.py`, JUG's tempo2 graphs
or tempo2 C — with two cited exceptions, both named in the spec: the
build-time longdouble and orbit-count reductions (§2, "algorithm only,
re-implemented"), and the tempo2 ELL1 Roemer truncation of Addendum A.4,
whose identity is cited to `ELL1model.C`.

---

## Components (SPEC §7.2)

| # | PINT component | stage | status |
|---|---|---|---|
| 1 | `AstrometryEcliptic` / `AstrometryEquatorial` (+ `SolarSystemShapiro`) | `solar_system` | done — **optional** (a par with no astrometry, `pure_rotator`, builds as it does in pyvela) and **`ECL`-aware**: the obliquity is resolved per engine (R7.3), not Vela's hard-coded IERS2010 |
| 2 | `SolarWindDispersion` | `solar_wind` | done |
| 3 | `DispersionDM` | `dispersion_taylor` | done |
| 4 | `DispersionDMX` | `dispersion_piecewise` | done |
| 5 | `Binary*` | `binary.<family>` | done — all seven families |
| 6 | `FD` | `frequency_dependent` | done |
| 7 | `FDJump` (`FDJUMPLOG Y`) | `frequency_dependent_jump` | done |
| 8 | `Spindown` | `spindown` | done (**changed**: build-time longdouble reduction) |
| 9 | `PhaseOffset` | `phase_offset` | done |
| 10 | `PhaseJump` | `phase_jump` / `phase_jump_exclusive` | done, both mask kinds |

---

## Requirements

### §3 Architecture

| | requirement | status | where |
|---|---|---|---|
| R3.1 | the trace never calls PINT / astropy / erfa | done | `test_engine.py::test_the_trace_never_calls_pint` |
| R3.2 | stages are a static tuple, unrolled, not `lax.scan` | done | `pipeline.py::run_chain` |
| R3.3 | all rows processed as `(R,)`/`(R,3)` arrays; no `vmap` over pulsars | done | three-vectors are 3-tuples of `(R,)`, mirroring Vela's `NTuple{3}` |
| R3.4 | per-row branching uses frozen boolean arrays and `where` | done | `numerics.where`; the "substitute, then select" rule in `mikkola` |
| R3.5 | the TZR pseudo-TOA is row `R−1` | done | `freeze.py::freeze` |
| R3.6 | every physics module computes through `numerics`, never `jnp` | done | `numerics.py`; the dual (R11.5) is what it buys |
| R3.7 | the finite-frequency guard | done | `correction.py::inverse_freq_sqr`, `freeze.py` |

### §4 Precision

| | requirement | status | where |
|---|---|---|---|
| R4.2 | build-time epoch reduction; live `PEPOCH`/`POSEPOCH`/`DMEPOCH` refused | done | `precision.py`, `freeze.py::validate_model` |
| R4.3 | reference phase from `tdbld`, not `get_mjds()`; `F0` from the exact token | done | `precision.py::reference_phase`; PINT holds `F0` as longdouble and it is used as such |
| R4.4 | spin phase in the trace | **changed** | the whole reference series is folded into the build-time constants, not only `F0★` — `precision.py::spin_coefficients`, `spindown.py` |
| R4.5 | binary time argument and orbit-count reduction | done, **extended** | `precision.py::reduce_orbits`; the integer is restored in the unwrapped true anomaly (`binary/dd.py`) — a spec omission that was a 2.7 ms error |
| R4.6 | TZR goes through the same stages, `phi_ref = 0`, `ssb_obs_pos = 0` when barycentred | done | `freeze.py`, and the `is_tzr` guards in `phase.py`, `dispersion.py`, `frequency_dependent.py` |
| R4.7 | `precision_critical_params()` | done | `engine.py` |
| R4.8 | `require_x64()` at construction, never enabling it silently | done | `config.py`; `require_longdouble()` added alongside |

### §5 Freeze

| | requirement | status | where |
|---|---|---|---|
| R5.2 | noise strip, superset of MetaPulsar's classifier | done | `freeze.py::NOISE_NAMES`; `test_freeze.py` gates the residual identity |
| R5.3 | `prepare_model` (PEPOCH, epochs, PHOFF, H4→STIGMA, zeroables, empty JUMPs, T2) | done | `freeze.py::prepare_model` |
| R5.3b | parameter accountability: consumed, inert, inert-at-zero, pinned, or refused by name | done | `freeze.py::account_for_parameters`, `INERT_PARAMS`, `PINNED_PARAMS`; gate P7 |
| R5.4 | `FrozenTOAs` fields and units | done, **extended** | `freeze.py`; adds `spin_coeffs` (R4.4) and `finite_freq` (below) |
| R5.5.1 | the freeze never sorts, argsorts, filters or permutes PINT's or tempo2's rows | done | `freeze.py::freeze`, `freeze.py::NO_REORDERING`; gate P3 |
| R5.5.2 | no permutation is published: no `data_order`, no `toa_index`, no `_isort` | done | `test_pulsar_data.py` asserts the *absence* on the engine, the record and the timing-package columns |
| R5.5.3 | a consumer that wants another order produces it on read | done | `test_feather.py` shows Enterprise applying its own `_isort` to our unpermuted file |

### §6 Parameters

| | requirement | status | where |
|---|---|---|---|
| §6.1 | `param_names` = PINT `free_params`; unconsumed free parameter refused | done | `engine.py` raises `UnsupportedModelError` naming them |
| §6.2 | internal units via pyvela's scale factors, not hand-typed | done | `units.py` |
| R6.3 | `ParamLayout` / `Params` | done, **changed** | `Params` is an attribute namespace rather than a `NamedTuple`; `build()` takes a `wrap` hook, which is how the dual is injected |

### §9 API

| | requirement | status | where |
|---|---|---|---|
| R9.1 | `residual_delta_jax` jitted, frozen arrays closed over | done | `engine.py` |
| R9.2 | `residuals(θ)` forms the delta in `Decimal` | done | `params.py::delta_from_theta` |
| R9.3 | `design_matrix()` is `−residual_jacobian()`, fitter sign, `param_names` order | done | `engine.py`; gate M1 |
| R9.4 | PINT's matrix demoted to an oracle, row-permuted into freeze order | done | `design_matrix(source="pint")`; gate T9, in phase at 10⁻³ |
| R9.5 | no zero-delta short-circuit in the traced path | done | only `residual_delta` (numpy) short-circuits |
| R9.6 | the Jacobian is computed once and cached; `pulsar_data()` triggers it | done | `engine.py`; the README says a residuals-only look uses `Engine` |

### §11 Single precision

| | requirement | status | where |
|---|---|---|---|
| R11.2 | the live nonlinear set | done | `perturbative/__init__.py`; modes `binary`, `binary+`, `binary+astrometry`, `astrometry` |
| R11.3 | the reference channel is float64 always, cast where it enters the perturbation channel | done | `perturbative/dual.py` (`Pert._cast` marks every site); closes `J2302+4442` ([PARITY](PARITY.md#the-perturbative-engine-spec-12-t14-and-t15)) |
| R11.4 | the difference identities | done | `perturbative/dual.py`, one method each |
| §11.5–11.7 (archived draft) | hand-written delta kernels per component | **changed** | not written: the chain runs over the dual instead, so the identities apply to *every* component without a second copy — now normative as SPEC §11.5 |
| §11.8 | assembly | done, **extended** | the TZR term carries `F_tzr/F_i`; without it a fast spin-down pulsar is 2–4% wrong |
| §11.9 | `certify()` | done | plus `quadratic_coefficient`, which diagnoses the assembly's one approximation |

### §10 nltiming interface

| | requirement | status | where |
|---|---|---|---|
| R10.1 | `VelaJaxTimingEngine` renames and forwards; recomputes nothing | done | `backend.py` |
| R10.2 | `derivative_method` honoured and recorded; unknown values raise | done | `backend.py`, `pulsar.py`; `test_pulsar.py` |
| R10.3 | `nonlinear_params` is the *executed* mode, reported to nltiming | done | `pulsar.py` → `Engine.perturbative`; `test_nltiming_integration.py` |
| R10.4 | registered under both native packages, `_IMPL_FAMILY = "pint"` | done | nltiming `engine_config`; `test_nltiming_integration.py` |
| R10.5 | gauge-free; `PHOFF` always free; the gauge column is the phase direction | done, **with an nltiming change** | `backend.py::gauge_direction` — see the ledger row below |

### §12 / Addendum B: the pulsar product

| | requirement | status | where |
|---|---|---|---|
| R-B1.1 | every public array is frozen state, the reference evaluation, or a model value | done | `pulsar_data.py::PulsarData.from_engine`; gate P2 monkeypatches the PINT physics methods |
| R-B1.4 | `toas`/`freqs` from the **pre-binary** barycentric snapshot, PINT's cutoff | done | `pipeline.py::barycentric_cut`, `Engine.reference_barycentric`; gate P1 cross-checks against PINT on binary fixtures |
| R-B2 | the field inventory | done | `pulsar_data.py` |
| B.3.1 | columnar flags at freeze; Enterprise's `backend_flags` recipe verbatim | done | `freeze.py::flag_columns`, `pulsar_data.py::backend_flags` |
| B.3.2 | the real `dmx` table, `None` without the component | done | `pulsar_data.py::dmx_table`; gate P5 |
| B.3.3 | `planetssb`/`sunssb`, Venus filled, velocities NaN | done | `pulsar_data.py::PLANET_SLOTS`; gate P6 |
| B.3.4 | `dm` never raises | done | `pulsar_data.py` |
| B.3.6 | no `filter_data`, `set_flags`, pickling, `sort_data`; frozen, read-only | done | `PulsarData` is a frozen dataclass with read-only views; `test_pulsar_data.py` |
| R-B4 | `state_id` removed (psrdata SPEC v1); nltiming fingerprints record content | done | `pulsar_data.py` |
| R-B5 | feather schema v1, with the metadata block | done | `PulsarData.to_feather`/`from_feather`; gate P4 |
| R-B6.1 | composition: holds the record and the engine, forwards the names, `is`-identical | done | `pulsar.py`; `test_pulsar_data.py` |
| R-B6.3 | `can_use_engines` answers only for a single-leg vela-jax pulsar | done | `pulsar.py` |
| R-B6.4 | timing-engine kwargs accepted but *checked*; an unknown one raises | done | `pulsar.py::ACCEPTED_KWARGS` |
| R-B7.2 | the chart-capability mirror enforces the real dataclass's invariant | done | `backend.py::_ChartCapability` raises without `certification_ref` |

### Addenda

Addendum A (tempo2) and Addendum B (the pulsar product) are implemented in
full. A.2.3 (the Roemer closure) is enforced at *build* time, not only in a
test: `Tempo2Error` if the frozen columns do not reproduce tempo2's own `roemer`
to 1 ns. A.2.6 is implemented **more strictly than its wording** — see the
ledger.

---

## Gates (SPEC §12)

| # | gate | status | test |
|---|---|---|---|
| T1 | freeze refuses missing planets / pulse numbers / `tdbld`; refuses each §1.2 item | done | `test_freeze.py` |
| T2 | `residual_delta(0)` exactly zero | done | `test_engine.py`, all 14 fixtures |
| T3 | vs PINT `Residuals` | done, **rebudgeted** | `test_engine.py`; 10⁻⁷ s per fixture, 10⁻⁶ for ELL1H/DDH where PINT's orthometric Shapiro genuinely differs from Vela's |
| T4 | vs `SPNTA.time_residuals` | done | `test_oracle_pyvela.py`; [PARITY](PARITY.md) |
| T5 | per-component Vela JSON tables | **open** | needs a Vela.jl export — [REVIEW](REVIEW.md#what-needs-discussion) |
| T6 | Mikkola vs a table | done, **differently** | `test_orbit.py` checks `u − e sin u = l` directly on a grid (1e-13) and 2π-equivariance, which needs no table |
| T7 | `residual_delta(δ)` vs `SPNTA` | done | `test_oracle_pyvela.py`; every free axis, not only the binary |
| T8 | Jacobian vs central differences | done | `test_engine.py`; the step is chosen per column so the comparison is not measuring the residual's own 10⁻¹³ s floor |
| T9 | PINT's oracle matrix vs `−J`, identically-linear columns | done, **rebudgeted** | `test_engine.py`; compared in *phase* (PINT divides by constant `F0`, Vela by the instantaneous doppler-shifted frequency), modulo one constant per column, at 10⁻³ — see [REVIEW](REVIEW.md#residual-design-matrix-gauge) |
| T10 | the trace never calls PINT | done | `test_engine.py` |
| T11 | noise-line strip is residual-neutral | done | `test_freeze.py` |
| T12 | live `T0`/`TASC`/`PB` vs pyvela | done | `test_oracle_pyvela.py` |
| T13 | orbit-count reduction vs unreduced | done | `test_precision.py` |
| T14 | perturbative fp64 vs the full engine | done, all 8 | `test_perturbative.py`; capped at 20 µs of residual change, above which the assembly's second-order term dominates and is [bounded separately](PARITY.md#the-perturbative-engines-validity-domain) |
| T15 | perturbative fp32, **full fixture set incl. `J2302+4442`** | done, all 8 | `test_perturbative.py`; R11.3 rule 1 closed the DDS gap |
| T16 | fp32 inside a Discovery fp32 likelihood | done, smoke | `examples/nuts_fp32.py`: the fp32 perturbative engine drives a Discovery likelihood whose linear algebra is `working=float32`, and the `SINI`/`M2` posterior matches the fp64 run within MC error |

### v2 gates (SPEC §12, new)

| # | gate | status | test |
|---|---|---|---|
| P1 | `toas`/`freqs` identities, and `toas` against PINT's barycentric arrival | done | `test_pulsar_data.py`; exact array equality, and ≤ 10⁻⁷ s vs PINT on binary fixtures (measured ≤ 3×10⁻¹² s) |
| P2 | one-source guard: PINT physics methods raise during the record build | done | `test_pulsar_data.py` |
| P3 | the freeze reproduces PINT's or tempo2's rows unchanged; nothing publishes a permutation | done | `test_pulsar_data.py` |
| P4 | feather round-trip lossless; read by stock Discovery *and* Enterprise readers | done | `test_feather.py` |
| P5 | `dmx` vs the par; windows partition the `stoas` they claim | done | `test_pulsar_data.py` |
| P6 | `planetssb` slots {2,4,5,6,7} vs Enterprise-PINT; velocities NaN; Venus filled | done | `test_pulsar_data.py`, ≤ 10⁻¹⁰ ls |
| M1 | `−Mmat` vs central differences, **all** columns | done | `test_engine.py`; worst 7.5×10⁻⁷ ([PARITY](PARITY.md#the-design-matrix-spec-12-m1--r93)) |
| H7 | PINT–tempo2 freeze floor, RMS ≤ 100 ns, mixed-engine-consistent files, ECL IERS2003 so PINT uses tempo2's obliquity | done | `test_read_tempo2.py`; ~1.5 ns on all 19; `sim_jump`/`sim_sw` replaced |
| H8 | `TRACK −2` required, with a whole-turn tripwire | done | `test_tempo2_gates.py` |
| S4 | libstempo, discriminating ELL1, ≤ 50 ns under `"tempo2"` and failing under `"pint"` | done | `test_tempo2_gates.py`; 3.3 ns vs 12 072 ns |
| N1 | nltiming installed: protocols, validators, gauge assert, `TimingSpec` end to end, hybrid manifest | done | `test_nltiming_integration.py`, and a **mandatory** CI job (`.github/workflows/ci.yml`) |
| P7 | R5.3b: `A0`/`B0`/`SWM 1`/wideband refused by name; frozen bare `DMX` inert; `ECL` read and tracking PINT | done | `test_freeze.py`; the `ECL` shift matches PINT's to <0.1% |

### Gates added beyond the spec

| gate | why | test |
|---|---|---|
| the geometry closure | the tempo2's whole correctness rests on one vector composition | `test_read_tempo2.py`, and at build time |
| PINT never re-clocks the science TOAs | the point of the tempo2 is that PINT does *not* redo that work | `test_read_tempo2.py` |
| `from_pint` on the injected table reproduces the engine bit for bit | isolates "the host supplied bad arrays" from "the physics is wrong" | `test_read_tempo2.py` |
| the dual's identities vs a longdouble oracle | a naive float64 difference is only good to 10⁻¹⁶, i.e. exactly the cancellation the identities exist to avoid | `test_perturbative.py` |
| the nltiming protocols, by `isinstance` | vela-jax does not import nltiming, so nothing else would notice a drift | `test_nltiming_integration.py` |
| `CLOCK` is respelled `CLK` for tempo2 | PINT's spelling silently un-pins tempo2's clock chain: 234 ns | `test_read_tempo2.py` |
| the site velocity comes from tempo2's `siteVel` | `observatory_earth[3:6]` is zero, and the Roemer closure cannot see it | `test_read_tempo2.py` |
| ecliptic DDK annual parallax is in KOM's frame | mixed ICRS/`KOM` is 2.2 μs on `sim_ddk.as_ECL()`; equatorial `sim_ddk` cannot see it | `test_ddk.py` |
| a bare `-pn` flag is not a phase connection | tempo2 reads the flags only under `TRACK −2`; claiming otherwise advertises an authority it never exercised | `test_tempo2_gates.py` |

---

## Refusals (SPEC §1.2)

Each raises `UnsupportedModelError` naming the offender and the supported set.

| refused | where |
|---|---|
| a non-zero parameter no stage evaluates (`A0`, `B0`, …) | `freeze.py::account_for_parameters` (R5.3b) |
| `SWM 1` / `SWM 2` (PINT's You+2007 solar wind) | `freeze.py::PINNED_PARAMS` (`TIMEEPH`/`T2CMETHOD` are inert ingest flags, not pinned) |
| fitted bare `DMX` (the info line, not `DMX_NNNN`) | `engine.py` unconsumed free parameter; frozen `DMX` is `INERT_PARAMS` |
| wideband TOAs (the tim carries DM measurements) | `freeze.py::account_for_parameters` |
| `BINARY BT` / `BTX` / `BT_piecewise` / `DDGR` / unresolved `T2` | `binary/__init__.py::resolve_family`, `freeze.py::validate_model` |
| `WaveX`, `DMWaveX`, `CMWaveX`, `PL*NoiseGP`, `Glitch`, `ChromaticCM/CMX`, `SimpleExponentialDip`, `FDJumpDM`, `DispersionJump`, `SolarWindDispersionX` | `freeze.py::ALLOWED_COMPONENTS` (an allow-list, so a new PINT component is refused by default) |
| `FDJUMPLOG N` | `freeze.py::validate_model` |
| overlapping or incomplete DMX coverage | `freeze.py::dmx_index` |
| a *fitted* JUMP selecting no TOA (a frozen one is dropped with a warning) | `freeze.py::_drop_empty_jumps` |
| DDK without astrometry, DDK with H3/STIGMA, DDK with `K96 N` | `freeze.py::validate_model` |
| free `PEPOCH` / `POSEPOCH` / `DMEPOCH` | `freeze.py::validate_model` |
| a free parameter no component consumes | `engine.py` |
| a binary par without exactly one of `PB`/`FB0` | `binary/__init__.py::uses_fbx` |
| **not** refused: `CORRECT_TROPOSPHERE Y` | accepted with a warning — see [REVIEW](REVIEW.md#troposphere-and-a-par-with-no-sky) |

Wideband TOAs **are** refused explicitly (R5.3b). Until v2.2 they were merely
not read — the freeze takes the narrowband columns, so a wideband par built and
returned narrowband residuals without saying so, which contradicted SPEC §1.2
and is the same class of silence as the dropped parameters above.

---

## Test inventory

127 test functions, 343 cases, in three tiers — `make fast` (37 cases, ~11 s,
no par/tim read), `make test` (247, ~120 s, one fixture per binary family)
and `make full` (343, ~280 s, everything). The counts come from
`pytest --collect-only`; the wall clocks were measured on an 8-core aarch64
container. The tiers are markers, so a gate cannot weaken between them:
`slow` is breadth over the remaining fixtures, never a looser budget. Every
file skips cleanly without its optional dependency.

| file | tests | covers |
|---|---:|---|
| `test_engine.py` | 10 | T2, T3, T8, T9, M1, T10; facts and the exact-θ round trip |
| `test_freeze.py` | 19 | T1, T11; the noise classifier, the refusals |
| `test_precision.py` | 6 | §4: the epoch reduction, `phi_ref`, `spin_coeffs`, T13 |
| `test_taylor.py` | 3 | the factorial convention, against PINT's own `taylor_horner` |
| `test_orbit.py` | 4 | T6; Mikkola, its 2π-equivariance, its gradient at the singular inputs |
| `test_ddk.py` | 4 | ecliptic DDK annual-parallax frame; ICRS vs `as_ECL()` on `sim_ddk` |
| `test_perturbative.py` | 8 | T14, T15; the dual's identities, its Kepler solve, the assembly's validity domain |
| `test_oracle_pyvela.py` | 3 | T4, T7, T12 — the Vela.jl oracle (`oracle` marker) |
| `test_tcb.py` | 1 | the `UNITS` rules and the TCB→TDB transform |
| `test_read_tempo2.py` | 15 | Addendum A: columns, closure, no re-clocking, pulse numbers, **H7** |
| `test_tempo2_gates.py` | 5 | Addendum A.6: **H8** and **S4** |
| `test_binary_conventions.py` | 6 | Addendum A.4: the ELL1 truncation |
| `test_pulsar_data.py` | 17 | Addendum B: **P1, P2, P3, P5, P6**, the record's shape and forwarding |
| `test_feather.py` | 3 | Addendum B.5: **P4**, schema v1, both consumers' readers |
| `test_pulsar.py` | 12 | Addendum B.6: composition, dispatch, kwargs, `derivative_method` |
| `test_backend.py` | 6 | Addendum B.7: the protocol shape, chart facts, the mirror's invariant |
| `test_nltiming_integration.py` | 9 | **N1** — the same, against the real nltiming protocols |
