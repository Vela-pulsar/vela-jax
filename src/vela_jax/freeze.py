"""The PINT read path, and the freeze both timing packages share.

PINT stays exactly what it is in pyvela: the thing that reads par/tim, applies
clock corrections, computes TDB, interpolates the JPL ephemerides and assigns
pulse numbers. :func:`load_pint` is that path end to end.

:func:`prepare_model`, :func:`freeze` and the mask helpers below are shared:
:mod:`vela_jax.read_tempo2` produces the same ``(model, toas)`` pair from
tempo2 instead, and hands it to the same :func:`freeze`. Everything here
happens once, in numpy; the JAX graph never calls PINT, astropy or erfa again.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from pint.models import PhaseOffset
from pint.models.parameter import MJDParameter
from psrdata.partext import NOISE_NAMES, is_noise_line, strip_noise_lines  # noqa: F401

from . import numerics as vm
from .binary import SUPPORTED as SUPPORTED_BINARIES
from .config import require_longdouble
from .constants import DAY_S
from .errors import FreezeError, UnsupportedModelError
from .precision import (
    OrbitReduction,
    reduce_orbits,
    reference_phase,
    seconds_since_epoch,
    spin_coefficients,
)

# --- 1. noise strip --------------------------------------------------------
#
# The classifier is `psrdata.partext.is_noise_line`, shared with MetaPulsar.
# It used to be a local copy, and the two copies disagreed: this one spelled
# the tempo2 EFAC family `TNEF`/`TNEQ` and caught `TRES`/`DMRES`, MetaPulsar's
# spelled them `TNEFAC`/`TNEQUAD` and did not -- while a cross-repo test
# asserted the two produced byte-identical output. The shared function is the
# union of both.


def has_value(model, name: str) -> bool:
    """A parameter is present in the model *and* actually set.

    A model helper, not a text rule: it reads a parsed PINT model, so it stays
    here rather than moving to psrdata with the noise classifier it used to
    sit beside.
    """
    return name in model and model[name].quantity is not None


# --- 2. ingest -------------------------------------------------------------


def load_pint(par, tim, **pint_kwargs):
    """Read a delay-only ``(model, toas)`` pair the engine can freeze.

    The par is parsed and checked *before* the tim is read. Reading TOAs is
    the expensive half -- clock corrections, TDB, one JPL ephemeris
    interpolation per observatory group -- and an unsupported component is
    visible from the par alone, so a `BINARY BT` file is refused in
    milliseconds instead of after six seconds of work nobody wanted.
    """
    from pint.models import get_model, get_model_and_toas

    par = Path(par)
    kwargs = dict(planets=True, allow_T2=True, allow_tcb=True, add_tzr_to_model=True)
    kwargs.update(pint_kwargs)
    with tempfile.TemporaryDirectory(prefix="vela_jax_") as tmp:
        stripped = Path(tmp) / par.name
        stripped.write_text(strip_noise_lines(par.read_text()))
        validate_model(
            get_model(
                str(stripped),
                allow_T2=kwargs["allow_T2"],
                allow_tcb=kwargs["allow_tcb"],
            )
        )
        model, toas = get_model_and_toas(str(stripped), str(tim), **kwargs)
    refuse_wideband(toas)
    prepare_model(model, toas)
    toas.compute_pulse_numbers(model)
    return model, toas


# --- 3. prepare_model (pyvela `fix_params`, delay part) --------------------

#: Vela's zeroable-if-unset list (pyvela `fix_params`), plus the DD-family
#: shape parameters SPEC §5.3 adds.
# fmt: off
ZEROABLE = (
    "M2", "SINI", "PBDOT", "XPBDOT", "A1DOT", "EPS1DOT", "EPS2DOT",
    "H3", "STIGMA", "LNEDOT", "EDOT", "OMDOT", "GAMMA", "DR", "DTH",
)
# fmt: on

ALLOWED_COMPONENTS = frozenset(
    {
        "AstrometryEquatorial",
        "AstrometryEcliptic",
        "SolarSystemShapiro",
        "SolarWindDispersion",
        "DispersionDM",
        "DispersionDMX",
        "FD",
        "FDJump",
        "Spindown",
        "PhaseOffset",
        "PhaseJump",
        "AbsPhase",
        "TroposphereDelay",
    }
    | {f"Binary{name}" for name in SUPPORTED_BINARIES}
)


def prepare_model(model, toas) -> None:
    """pyvela ``fix_params``, delay part only (SPEC §5.3). Mutates ``model``."""
    if model["PEPOCH"].value is None:
        raise FreezeError("PEPOCH is required; it is the engine's whole time origin.")

    for name in model.params:
        param = model[name]
        if (
            name.endswith("EPOCH")
            and isinstance(param, MJDParameter)
            and param.value is None
        ):
            param.quantity = model["PEPOCH"].quantity

    if "PhaseOffset" not in model.components:
        model.add_component(PhaseOffset())
    model["PHOFF"].frozen = False
    model["PHOFF"].uncertainty_value = 0.1

    if (
        "H4" in model
        and model["H4"].quantity is not None
        and model["STIGMA"].quantity is None
    ):
        model["STIGMA"].quantity = model["H4"].quantity / model["H3"].quantity
        model["STIGMA"].frozen = model["H4"].frozen
        model["H4"].frozen = True

    for name in ZEROABLE:
        if name in model and model[name].quantity is None:
            model[name].value = 0

    if "TroposphereDelay" in model.components and model["CORRECT_TROPOSPHERE"].value:
        # Vela leaves the troposphere commented out, so this engine does not
        # model it either. Refusing the whole par would block most EPTA/IPTA
        # files for a delay of order 10 ns that is a near-constant offset and
        # cancels out of every residual *difference*, so it is disabled with a
        # warning instead. (SPEC 1.2 lists this as a refusal; accepting it is a
        # deliberate, documented relaxation.)
        warnings.warn(
            "CORRECT_TROPOSPHERE Y: this engine does not model a tropospheric "
            "delay, so absolute residuals carry it as an offset of order 10 ns; "
            "residual differences are unaffected",
            stacklevel=3,
        )
        model["CORRECT_TROPOSPHERE"].value = False

    _drop_empty_jumps(model, toas)
    validate_model(model)


def _drop_empty_jumps(model, toas) -> None:
    """A frozen JUMP selecting no TOA is dropped; a fitted one is an error."""
    component = model.components.get("PhaseJump")
    if component is None:
        return
    for param in list(component.get_jump_param_objects()):
        if param.key is None or len(param.select_toa_mask(toas)):
            continue
        if not param.frozen:
            raise UnsupportedModelError(
                f"fitted JUMP {param.name}, which selects no TOA (it would be "
                "an all-zero design column); freeze or remove it"
            )
        warnings.warn(f"dropping frozen {param.name}: it selects no TOA", stacklevel=3)
        model.remove_param(param.name)
    if not model.components["PhaseJump"].get_jump_param_objects():
        model.remove_component("PhaseJump")


def validate_model(model) -> None:
    """Refuse everything v1 does not implement, by name."""
    for name in model.components:
        if name in ALLOWED_COMPONENTS:
            continue
        if name.startswith("Binary"):
            raise UnsupportedModelError(
                f"BINARY {model.BINARY.value}", SUPPORTED_BINARIES
            )
        raise UnsupportedModelError(
            f"PINT component {name}", sorted(ALLOWED_COMPONENTS)
        )
    if model.BINARY.value is not None and str(model.BINARY.value).upper() == "T2":
        raise UnsupportedModelError("BINARY T2 unresolved by PINT", SUPPORTED_BINARIES)
    if "FDJump" in model.components and not model["FDJUMPLOG"].value:
        raise UnsupportedModelError("FDJUMPLOG N")
    if "BinaryDDK" in model.components:
        if not (
            "AstrometryEcliptic" in model.components
            or "AstrometryEquatorial" in model.components
        ):
            raise UnsupportedModelError("DDK without an astrometry component")
        if "H3" in model and model["H3"].quantity is not None and model["H3"].value:
            raise UnsupportedModelError("DDK with H3/STIGMA")
        if "K96" in model and model["K96"].value is False:
            raise UnsupportedModelError("DDK with K96 N (Vela always applies PM terms)")
    for name in ("PEPOCH", "POSEPOCH", "DMEPOCH"):
        if (
            name in model
            and model[name].quantity is not None
            and not model[name].frozen
        ):
            raise UnsupportedModelError(f"a free {name} (SPEC §4.2 refuses it)")


# --- 3b. parameter accountability (R5.3b) ---------------------------------

#: Parameters whose *value* cannot reach the delay chain's output. Every name
#: here is inert for a stated reason, not because nothing happened to break.
#:
#: The rule they exist for: a par may only set a parameter this engine either
#: **consumes**, or knows to be inert, or **refuses by name**. Silently
#: dropping a parameter PINT honours is the one failure mode a fixture suite
#: cannot catch, because a fixture that does not set the parameter looks
#: identical either way -- which is exactly how ``ECL`` (~100 ns of Roemer
#: delay on a typical EPTA/IPTA ``IERS2003`` par) survived a full parity pass.
# fmt: off
INERT_PARAMS = {
    # identity and bookkeeping
    "PSR", "PSRJ", "PSRB", "NTOA", "START", "FINISH", "TRES", "CHI2", "CHI2R",
    "NITS", "EPHVER", "MODE", "INFO", "TRACK",
    "DMDATA",                     # "was the fit done with per-TOA DMs?"; not a delay
    # ingest-side: PINT (or tempo2) honours these *before* the freeze, so they
    # are already baked into tdbld and the ephemeris columns we read.
    # TIMEEPH / T2CMETHOD belong here, not on PINNED_PARAMS: any value the
    # host accepted is already in the freeze, so pinning IAU2000B / FB90
    # would refuse a par the physics has already consumed.
    "EPHEM", "CLOCK", "UNITS", "TIMEEPH", "T2CMETHOD", "DILATEFREQ",
    "PLANET_SHAPIRO",             # read at build as a static stage flag
    # consumed by the freeze rather than by a stage
    "PEPOCH", "TZRMJD", "TZRFRQ", "TZRSITE", "BINARY", "FDJUMPLOG",
    "SWEPOCH", "DMEPOCH", "POSEPOCH",
    "H4",                         # folded into STIGMA by prepare_model
    "CORRECT_TROPOSPHERE",        # relaxation: disabled with a warning (SPEC 1.2)
    # not a delay
    "RM",                         # rotation measure: polarimetry, not timing
    "DMX",                        # DispersionDMX info line; delay is DMX_NNNN
    # PINT's ELL1H harmonic count. Vela subtracts the a0/b1/a2 harmonics
    # analytically instead, which is the T3 budget's documented ELL1H/DDH
    # widening; the number itself never reaches this chain.
    "NHARMS",
    # only meaningful alongside a component this engine refuses outright
    "ORBWAVE_EPOCH", "SWP",
}
# fmt: on

#: Parameters that are inert **only** at a particular value, with the value
#: this engine's physics assumes. Anything else is refused by name.
#: This is not the ingest-convention list: ``T2CMETHOD`` / ``TIMEEPH`` are
#: on ``INERT_PARAMS`` because the host already applied them.
PINNED_PARAMS = {
    # Vela implements the spherical solar wind only (`solarwind.jl`); SWM 1/2
    # is PINT's You et al. (2007) model, which is a different delay.
    "SWM": 0,
}


def _is_zero(param) -> bool:
    """A numeric delay amplitude of zero. Bools and strings are not amplitudes.

    ``float(False) == 0.0``, so without the bool guard a flag sitting at N
    (PINT ``boolParameter``) would look inert and skip the offender check.
    ``DMDATA N`` used to take that path; it is now on ``INERT_PARAMS``.
    """
    value = getattr(param, "value", None)
    if value is None:
        return True
    if isinstance(value, (bool, str)):
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def refuse_wideband(toas) -> None:
    """SPEC 1.2 / R5.3b: do not silently return narrowband residuals.

    Wideband is a property of the TOAs (they carry DM measurements), not of
    a par component. ``DispersionJump`` is a separate allow-list refusal and
    must not stand in for this check.
    """
    if toas is not None and getattr(toas, "wideband", False):
        raise UnsupportedModelError(
            "wideband TOAs (the tim carries DM measurements). This engine "
            "freezes the narrowband columns only, so it would return "
            "narrowband residuals for a wideband data set without saying so",
            ["narrowband TOAs"],
        )


def account_for_parameters(model, toas, consumed) -> None:
    """Refuse a par that sets something this engine would silently ignore.

    ``validate_model`` allow-lists *components* and ``Engine`` refuses *free*
    parameters no stage reads. Between the two sat a real gap: a **frozen**
    parameter that PINT applies and no stage here consumes just disappeared.
    Measured on the Vela fixtures, against PINT's own residuals: ``A0 = 1e-5``
    moves PINT by 14 microseconds and this engine by nothing; so does ``B0``;
    ``ECL IERS1992`` moves PINT by 1.3 microseconds.

    A numeric parameter sitting at zero is genuinely inert -- every one of
    these enters its delay additively -- so it is accepted in silence. A
    non-zero value that nothing consumes is refused, named, with the supported
    set, exactly like an unsupported component.
    """
    from pint.models.parameter import funcParameter

    refuse_wideband(toas)

    offenders = []
    for name in model.params:
        param = model[name]
        if name in consumed or name in INERT_PARAMS:
            continue
        # A derived view of other parameters (ELL1's ECC/OM/T0, DDK's
        # KINIAU/KOMIAU), never an independent input.
        if isinstance(param, funcParameter):
            continue
        if name in PINNED_PARAMS:
            if param.value is not None and param.value != PINNED_PARAMS[name]:
                offenders.append(f"{name} {param.value}")
            continue
        if name.startswith(("DMXR1_", "DMXR2_")):
            continue  # window bounds; consumed by `dmx_index`
        if _is_zero(param):
            continue
        offenders.append(f"{name} {param.value}")

    if offenders:
        raise UnsupportedModelError(
            "parameters set in the par that no component of this engine "
            f"evaluates: {sorted(offenders)}. Ignoring them would change the "
            "residual silently",
            sorted(consumed | INERT_PARAMS),
        )


# --- 4. masks --------------------------------------------------------------


def read_mask(toas, params) -> np.ndarray:
    """pyvela ``read_mask``: one boolean row per mask parameter."""
    rows = []
    for param in params:
        row = np.zeros(len(toas), dtype=bool)
        row[param.select_toa_mask(toas)] = True
        if not row.any():
            raise UnsupportedModelError(
                f"mask parameter {param.name}, selecting no TOA"
            )
        rows.append(row)
    return np.array(rows, dtype=bool) if rows else np.zeros((0, len(toas)), bool)


def is_exclusive(mask: np.ndarray) -> bool:
    return bool(np.all(mask.sum(axis=0) <= 1))


def exclusive_index(mask: np.ndarray) -> np.ndarray:
    """1-based selector index per TOA, 0 where nothing selects it."""
    if not is_exclusive(mask):
        raise ValueError("mask is not exclusive")
    index = np.zeros(mask.shape[1], dtype=np.int32)
    rows, cols = np.nonzero(mask)
    index[cols] = rows + 1
    return index


def sat_mjd_longdouble(toas) -> np.ndarray:
    """Site arrival times as longdouble MJD.

    ``toas.get_mjds()`` returns float64 days, which at MJD 55000 quantises to
    ~0.6 us -- three orders of magnitude above the engine's error budget.
    ``get_mjds(high_precision=True)`` returns an *object* array of scalar
    ``Time``s instead of numbers, so it is gathered into one ``Time`` array
    and converted with PINT's own two-part reduction.
    """
    from astropy.time import Time
    from pint.pulsar_mjd import time_to_longdouble

    times = Time([entry for entry in toas.table["mjd"]])
    return np.asarray(time_to_longdouble(times), dtype=np.longdouble)


def dmx_index(model, toas, prefix="DMX_") -> np.ndarray:
    """pyvela ``get_dmx_mask``: the DMX window index of every TOA.

    Vectorised over the whole table. The per-TOA ``find_prefix_bytime`` loop
    this replaces is O(N x windows) in Python and cost ~2 s on a 20-year DMX
    par; it is kept in ``tests/test_freeze.py`` as the oracle this must match
    bit for bit.

    The window predicate is inclusive at both ends, which is
    ``find_prefix_bytime``'s, and the time compared is the SAT MJD in
    longdouble rather than ``toas.table["mjd"]``'s float64 -- the same instant,
    read at the precision the rest of the freeze uses.
    """
    keys = np.array(sorted(model.get_prefix_mapping(prefix)), dtype=np.int32)
    if keys.size == 0:
        raise UnsupportedModelError(f"model has no {prefix} ranges")
    r1 = np.array(
        [model[f"DMXR1_{key:04d}"].value for key in keys], dtype=np.longdouble
    )
    r2 = np.array(
        [model[f"DMXR2_{key:04d}"].value for key in keys], dtype=np.longdouble
    )
    mjd = sat_mjd_longdouble(toas)
    inside = (mjd[:, None] >= r1[None, :]) & (mjd[:, None] <= r2[None, :])

    count = inside.sum(axis=1)
    if (count == 0).any():
        raise UnsupportedModelError(
            f"a TOA in no {prefix} range (TOA {int(np.argmax(count == 0))}); "
            f"{prefix.rstrip('_')} must cover every TOA"
        )
    if (count > 1).any():
        raise UnsupportedModelError(
            f"overlapping {prefix} ranges (at TOA {int(np.argmax(count > 1))})"
        )
    return keys[np.argmax(inside, axis=1)].astype(np.int32)


# --- 5. frozen arrays ------------------------------------------------------


class FrozenBinary(NamedTuple):
    dt_red: Any
    n_orb: Any
    period_ref_s: float


class FrozenTOAs(NamedTuple):
    """Everything the trace reads. ``R = N + 1`` rows; row ``R-1`` is the TZR."""

    tau: Any
    phi_ref: Any
    spin_coeffs: tuple
    freq_hz: Any
    finite_freq: Any
    is_tzr: Any
    is_bary: Any
    ssb_obs_pos: vm.Vec3
    ssb_obs_vel: vm.Vec3
    obs_sun_pos: vm.Vec3
    planet_pos: dict
    dmx_index: Any = None
    jump_index: Any = None
    jump_masks: tuple = ()
    fdjump_masks: tuple = ()
    fdjump_exp: tuple = ()
    binary: FrozenBinary | None = None

    @property
    def n_rows(self) -> int:
        return int(self.tau.shape[0])

    @property
    def n_toas(self) -> int:
        return self.n_rows - 1


class TOAColumns(NamedTuple):
    """TOA-table columns captured at the same freeze, in the same row order.

    The trace never reads these; :class:`~vela_jax.pulsar_data.PulsarData`
    does (Addendum B, source tag **F**). Capturing them here rather than
    re-querying the timing package later is the point: residuals and ``Mmat``
    come from the same timing-model evaluation, and a second pass over a
    PINT/tempo2 object is a different freeze.
    """

    tdb_seconds: Any  # longdouble seconds, TDB (``tdbld * 86400``)
    sat_seconds: Any  # longdouble seconds, site arrival time
    toaerrs: Any  # float64 seconds
    telescope: Any  # observatory codes
    flags: dict  # columnar tim flags, missing values ""
    ssb_obs_pos: Any  # (N, 3) light-seconds
    obs_sun_pos: Any  # (N, 3) light-seconds, observatory -> Sun
    body_pos: dict  # observatory -> body, (N, 3) light-seconds


#: The five bodies Vela applies a planetary Shapiro delay for.
PLANET_COLUMNS = ("jupiter", "saturn", "venus", "uranus", "neptune")

#: Bodies whose SSB positions the *pulsar record* wants, Earth included.
#: Enterprise reads slots {2 Earth, 4 Jupiter, 5 Saturn, 6 Uranus, 7 Neptune};
#: Venus is frozen too and slotted when present (SPEC B.3.3).
EPHEMERIS_BODIES = ("earth",) + PLANET_COLUMNS


def _column_vec3(table, name, unit):
    values = table[name].quantity.to_value(unit)
    return tuple(np.asarray(values[:, k], dtype=float) for k in range(3))


def _stack_rows(main, tzr):
    return jnp.asarray(np.concatenate([np.atleast_1d(main), np.atleast_1d(tzr)]))


def _stack_vec3(main, tzr) -> vm.Vec3:
    return tuple(_stack_rows(main[k], tzr[k]) for k in range(3))


def _as_array(vec3) -> np.ndarray:
    """A ``Vec3`` of ``(N,)`` components as one ``(N, 3)`` array."""
    return np.stack([np.asarray(component) for component in vec3], axis=1)


def _require_columns(toas, label) -> None:
    for column in ("tdbld", "ssb_obs_pos", "ssb_obs_vel", "obs_sun_pos"):
        if column not in toas.table.colnames:
            raise FreezeError(f"{label} has no '{column}' column")
    if not toas.planets:
        raise FreezeError(f"{label} was not read with planets=True")


def flag_columns(toas) -> dict[str, np.ndarray]:
    """The tim flags as one string column per flag name, missing values ``""``.

    Captured at freeze (SPEC B.3.1). A per-TOA ``get_flags()`` walk after the
    fact would be a second read of the timing-package object, and when
    tempo2 read the files it would be a read of a *different* object than
    the one that produced the geometry.
    """
    n = len(toas)
    columns: dict[str, list[str]] = {}
    for index, row in enumerate(toas.get_flags()):
        for key, value in row.items():
            columns.setdefault(key, [""] * n)[index] = str(
                getattr(value, "value", value)
            )
    return {key: np.array(value, dtype=str) for key, value in columns.items()}


#: **This package never reorders TOAs.** Row ``i`` of every frozen array, every
#: residual, every Jacobian row, the pulsar record and the feather file is row
#: ``i`` of the TOA table -- PINT's or tempo2's, whichever read the
#: files. There is no argsort at the freeze and no permutation anywhere.
#:
#: v2.0 of the spec briefly required a stable argsort on ``tdbld`` here, on the
#: grounds that Enterprise sorts lazily through ``_isort`` and Discovery
#: "assumes sorted input". Both halves are wrong, and the code says so:
#: Enterprise's ``create_quantization_matrix`` (``signals/utils.py:1149``) and
#: Discovery's ``quantize`` (``signals.py:51``) each ``argsort`` *internally*
#: and write bin membership back into the caller's row order, so neither needs
#: sorted input for ECORR; Enterprise's one order-sensitive path,
#: ``quant2ind(as_slice=True)``, checks contiguity itself and falls back to
#: index arrays. MetaPulsar has kept ``sort=False`` throughout and gates it
#: (``tests/optional/test_enterprise_ecorr_unsorted.py`` shuffles a pulsar and
#: builds ECORR on it).
#:
#: What the sort *did* buy was a permutation protocol across a package
#: boundary: the engine published one row order, MetaPulsar's leg published
#: another, and a consumer that compared ``engine.residuals()`` against a
#: composite's rows without going through the adapter got a plausible, wrong
#: likelihood. Two public orders for one freeze is the bug the sort claimed to
#: prevent. A consumer that wants time order sorts when it reads.
NO_REORDERING = True


def freeze(model, toas, layout, *, family=None, use_fbx=False):
    """``(FrozenTOAs, TOAColumns)`` for ``N`` TOAs plus the TZR pseudo-TOA.

    Row ``i`` is the TOA table's row ``i`` (see :data:`NO_REORDERING`); the TZR
    pseudo-TOA is appended as row ``R-1`` (R3.5). Nothing here permutes.
    """
    require_longdouble()
    _require_columns(toas, "TOAs")
    if toas.get_pulse_numbers() is None:
        raise FreezeError("pulse numbers are missing; call toas.compute_pulse_numbers")

    pepoch = model["PEPOCH"].value
    tzr = model.get_TZR_toa(toas)
    _require_columns(tzr, "the TZR TOA")

    tdb_mjd_ld = np.asarray(toas.table["tdbld"].value, dtype=np.longdouble)
    tau_ld = seconds_since_epoch(tdb_mjd_ld, pepoch)
    tau_tzr_ld = seconds_since_epoch(tzr.table["tdbld"].value, pepoch)[0]

    pulse_number = (
        toas.table["pulse_number"].value - toas.table["delta_pulse_number"].value
    )
    all_tau_ld = np.append(np.asarray(tau_ld), np.longdouble(tau_tzr_ld))
    phi_ref = reference_phase(tau_ld, pulse_number, tau_tzr_ld, layout.f_ld)
    spin_coeffs = tuple(
        jnp.asarray(c) for c in spin_coefficients(all_tau_ld, layout.f_ld)
    )

    ls = u.lightsecond
    pos = _column_vec3(toas.table, "ssb_obs_pos", ls)
    pos_tzr = _column_vec3(tzr.table, "ssb_obs_pos", ls)
    vel = _column_vec3(toas.table, "ssb_obs_vel", ls / u.s)
    vel_tzr = _column_vec3(tzr.table, "ssb_obs_vel", ls / u.s)
    sun = _column_vec3(toas.table, "obs_sun_pos", ls)
    sun_tzr = _column_vec3(tzr.table, "obs_sun_pos", ls)

    body_pos: dict[str, np.ndarray] = {}
    for body in EPHEMERIS_BODIES:
        column = f"obs_{body}_pos"
        if column in toas.table.colnames:
            body_pos[body] = _as_array(_column_vec3(toas.table, column, ls))

    planet_pos = {}
    if "PLANET_SHAPIRO" in model and model["PLANET_SHAPIRO"].value:
        for body in PLANET_COLUMNS:
            column = f"obs_{body}_pos"
            if column not in toas.table.colnames:
                raise FreezeError(f"PLANET_SHAPIRO is on but '{column}' is missing")
            planet_pos[body] = _stack_vec3(
                _column_vec3(toas.table, column, ls),
                _column_vec3(tzr.table, column, ls),
            )

    n_toas = len(toas)
    bary = np.concatenate(
        [
            np.all(np.stack(pos, axis=1) == 0.0, axis=1),
            np.all(np.stack(pos_tzr, axis=1) == 0.0, axis=1),
        ]
    )
    is_tzr = np.zeros(n_toas + 1, dtype=bool)
    is_tzr[-1] = True

    # PINT's infinite observing frequency (a TOA, or a TZR with no TZRFRQ,
    # that carries no dispersive delay) is replaced by a finite placeholder;
    # the dispersive stages select those rows off. See `inverse_freq_sqr`.
    freq_hz = np.concatenate(
        [toas.get_freqs().to_value(u.Hz), tzr.get_freqs().to_value(u.Hz)]
    )
    finite_freq = np.isfinite(freq_hz) & (freq_hz != 0.0)
    freq_hz = np.where(finite_freq, freq_hz, 1.0e9)

    frozen = dict(
        tau=_stack_rows(tau_ld.astype(np.float64), np.float64(tau_tzr_ld)),
        phi_ref=_stack_rows(phi_ref, 0.0),
        spin_coeffs=spin_coeffs,
        freq_hz=jnp.asarray(freq_hz),
        finite_freq=jnp.asarray(finite_freq),
        is_tzr=jnp.asarray(is_tzr),
        is_bary=jnp.asarray(bary),
        ssb_obs_pos=_stack_vec3(pos, pos_tzr),
        ssb_obs_vel=_stack_vec3(vel, vel_tzr),
        obs_sun_pos=_stack_vec3(sun, sun_tzr),
        planet_pos=planet_pos,
    )

    if "DispersionDMX" in model.components:
        frozen["dmx_index"] = jnp.asarray(
            np.append(dmx_index(model, toas), 0).astype(np.int32)
        )

    if "PhaseJump" in model.components:
        mask = read_mask(toas, model.components["PhaseJump"].get_jump_param_objects())
        padded = np.concatenate([mask, np.zeros((mask.shape[0], 1), bool)], axis=1)
        if is_exclusive(mask):
            frozen["jump_index"] = jnp.asarray(exclusive_index(padded))
        else:
            frozen["jump_masks"] = tuple(jnp.asarray(row) for row in padded)

    if "FDJump" in model.components:
        component = model.components["FDJump"]
        names = [n for n in component.fdjumps if model[n].quantity is not None]
        mask = read_mask(toas, [model[n] for n in names])
        padded = np.concatenate([mask, np.zeros((mask.shape[0], 1), bool)], axis=1)
        frozen["fdjump_masks"] = tuple(jnp.asarray(row) for row in padded)
        frozen["fdjump_exp"] = tuple(int(component.get_fd_index(n)) for n in names)

    if family is not None:
        frozen["binary"] = _freeze_binary(model, layout, all_tau_ld, use_fbx=use_fbx)

    toa_columns = TOAColumns(
        tdb_seconds=tdb_mjd_ld * np.longdouble(DAY_S),
        # Longdouble, not `get_mjds()`'s float64: the SAT is what every
        # consumer (`stoas`, the canonical writer, PINT re-ingest) reads back
        # out, and float64 days quantise it to ~0.6 us.
        sat_seconds=sat_mjd_longdouble(toas) * np.longdouble(DAY_S),
        toaerrs=np.asarray(toas.get_errors().to_value(u.s), dtype=float),
        telescope=np.asarray(toas.get_obss()),
        flags=flag_columns(toas),
        ssb_obs_pos=_as_array(pos),
        obs_sun_pos=_as_array(sun),
        body_pos=body_pos,
    )
    return FrozenTOAs(**frozen), toa_columns


def _freeze_binary(model, layout, all_tau_ld, *, use_fbx) -> FrozenBinary:
    epoch_name = "TASC" if has_value(model, "TASC") else "T0"
    epoch_rel = layout.ref_internal[epoch_name]
    period = (
        1.0 / float(model["FB0"].quantity.to_value(u.Hz))
        if use_fbx
        else float(model["PB"].quantity.to_value(u.day)) * DAY_S
    )
    reduction: OrbitReduction = reduce_orbits(all_tau_ld, epoch_rel, period)
    return FrozenBinary(
        dt_red=jnp.asarray(reduction.dt_red),
        n_orb=jnp.asarray(reduction.n_orb),
        period_ref_s=reduction.period_ref_s,
    )


# --- 6. free-parameter families -------------------------------------------


def prefix_members(model, prefix, *, include_base=None) -> tuple[str, ...]:
    """Prefix-parameter names in index order, stopping at the first unset one.

    pyvela's ``_get_multiparam_elements`` breaks at the first ``None``; Vela's
    Taylor series are dense, so a gap would silently shift every later term.
    """
    try:
        mapping = model.get_prefix_mapping(prefix)
    except ValueError:
        mapping = {}
    names = [mapping[i] for i in sorted(mapping)]
    if include_base:
        names = [include_base] + [n for n in names if n != include_base]
    members = []
    for name in names:
        if name not in model or model[name].quantity is None:
            break
        members.append(name)
    return tuple(members)


def build_delta_sequences(model) -> dict[str, tuple[str, ...]]:
    """Families whose *deltas*, not values, the trace uses.

    Only the spin series: its reference part is baked into ``phi_ref`` and
    ``spin_coeffs`` at load time, so the trace carries the free deltas alone.
    """
    return {"dF": prefix_members(model, "F", include_base="F0")}


def build_sequences(model) -> dict[str, tuple[str, ...]]:
    """The prefix families the stages Horner or index over."""
    sequences: dict[str, tuple[str, ...]] = {}
    if "DispersionDM" in model.components:
        sequences["DM"] = prefix_members(model, "DM", include_base="DM")
    if "DispersionDMX" in model.components:
        sequences["DMX_"] = prefix_members(model, "DMX_")
    if "SolarWindDispersion" in model.components:
        sequences["NE_SW"] = prefix_members(model, "NE_SW", include_base="NE_SW")
    if "FD" in model.components:
        sequences["FD"] = prefix_members(model, "FD")
    if has_value(model, "FB0"):
        sequences["FB"] = prefix_members(model, "FB", include_base="FB0")
    if "PhaseJump" in model.components:
        sequences["JUMP"] = tuple(
            p.name for p in model.components["PhaseJump"].get_jump_param_objects()
        )
    if "FDJump" in model.components:
        sequences["FDJUMP"] = tuple(
            n
            for n in model.components["FDJump"].fdjumps
            if model[n].quantity is not None
        )
    return sequences
