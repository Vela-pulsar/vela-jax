"""Assembling the component chain, in pyvela's order.

The stage tuple is fixed at build time and the trace is an unrolled Python
loop over it: the stages are heterogeneous, so ``lax.scan`` does not apply, and
unrolling is what lets each component stay a plain readable function.

The chain also declares which parameters it *reads*. A free PINT parameter no
stage consumes is refused rather than silently ignored — the usual cause is a
noise or GP parameter that survived the strip.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from .astrometry import solar_system
from .binary import (
    binary_stage,
    ddr_consumed,
    resolve_ddr_config,
    resolve_family,
    uses_fbx,
)
from .constants import OBL, obliquity_radians
from .correction import Correction
from .dispersion import dispersion_piecewise, dispersion_taylor
from .frequency_dependent import frequency_dependent, frequency_dependent_jump
from .phase import phase_jump, phase_jump_exclusive, phase_offset
from .solarwind import solar_wind
from .spindown import spindown

Stage = Callable[[object, Correction, object], Correction]

_EQUATORIAL = ("RAJ", "DECJ", "PMRA", "PMDEC")
_ECLIPTIC = ("ELONG", "ELAT", "PMELONG", "PMELAT")

# fmt: off
_BINARY_COMMON = ("A1", "A1DOT", "PB", "PBDOT")
_DD_KEPLER = ("T0", "ECC", "EDOT", "OM", "OMDOT", "GAMMA", "DR", "DTH")
_ELL1_KEPLER = ("TASC", "EPS1", "EPS2", "EPS1DOT", "EPS2DOT")
#: Per-family consumed names for the seven pre-DDR families. **DDR is
#: deliberately absent**: its consumed set is mode-dependent
#: (``binary.ddr_consumed``), and indexing this table with ``"DDR"`` is a
#: ``KeyError`` on dispatch.
_BINARY_EXTRA = {
    "DD":    _DD_KEPLER + ("M2", "SINI"),
    "DDH":   _DD_KEPLER + ("H3", "STIGMA"),
    "DDS":   _DD_KEPLER + ("M2", "SHAPMAX"),
    "DDK":   _DD_KEPLER + ("M2", "KIN", "KOM", "PX"),
    "ELL1":  _ELL1_KEPLER + ("M2", "SINI"),
    "ELL1H": _ELL1_KEPLER + ("H3", "STIGMA"),
    "ELL1k": ("TASC", "EPS1", "EPS2", "OMDOT", "LNEDOT", "M2", "SINI"),
}
# fmt: on


#: Stage order, by slot name. Vela's own order (pyvela
#: ``pint_components_to_vela``). ``binary_conventions`` only selects the
#: ELL1 Roemer truncation (:mod:`vela_jax.binary.ell1`).
STAGE_ORDER = (
    "solar_system",
    "solar_wind",
    "dispersion_taylor",
    "dispersion_piecewise",
    "binary",
    "frequency_dependent",
    "frequency_dependent_jump",
    "spindown",
    "phase_offset",
    "phase_jump",
)

CONVENTIONS = ("pint", "tempo2")


def validate_conventions(name: str) -> str:
    if name not in CONVENTIONS:
        raise ValueError(
            f"binary_conventions must be one of {CONVENTIONS}; got {name!r}"
        )
    return name


class Chain(NamedTuple):
    stages: tuple[tuple[str, Stage], ...]
    consumed: frozenset[str]
    family: str | None
    use_fbx: bool
    ecliptic: bool
    conventions: str = "pint"
    #: How many stages are inside PINT's barycentric cutoff (SPEC R-B1.4).
    bary_cut: int = 0

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.stages)


def binary_family(model) -> str | None:
    """The Vela binary family name, or ``None`` for an isolated pulsar."""
    return resolve_family(model) if model.BINARY.value is not None else None


def build_chain(
    model,
    sequences,
    delta_sequences,
    frozen,
    *,
    conventions: str = "pint",
    obliquity: float | None = None,
) -> Chain:
    """The stage tuple for a prepared PINT model.

    Stages are collected by slot name and then laid out in ``STAGE_ORDER``.
    ``conventions`` selects the ELL1 truncation, not the stage order.
    """
    validate_conventions(conventions)
    components = model.components
    stages: dict[str, tuple[str, Stage]] = {}
    consumed: set[str] = set()

    ecliptic = "AstrometryEcliptic" in components
    has_astrometry = ecliptic or "AstrometryEquatorial" in components
    # The obliquity is the par's ``ECL`` unless the timing package has
    # already rotated its own vectors with a different one (Addendum A:
    # tempo2 does, and ignores ``ECL``). Resolved here, never in the trace.
    # DDK annual-parallax uses the same value so I0/J0 live in KOM's frame.
    resolved = (
        obliquity_radians(model["ECL"].value)
        if obliquity is None and "ECL" in model
        else (OBL if obliquity is None else obliquity)
    )
    if has_astrometry:
        # A par with no astrometry at all is a bare rotator: pyvela adds no
        # SolarSystem component either, and every TOA is treated as barycentred.
        planet_shapiro = bool(
            "PLANET_SHAPIRO" in model and model["PLANET_SHAPIRO"].value
        )
        stages["solar_system"] = (
            "solar_system",
            lambda f, c, p: solar_system(
                f,
                c,
                p,
                ecliptic=ecliptic,
                planet_shapiro=planet_shapiro,
                obliquity=resolved,
            ),
        )
        consumed |= set(_ECLIPTIC if ecliptic else _EQUATORIAL) | {"PX", "POSEPOCH"}
        if ecliptic:
            consumed.add("ECL")

    if "SolarWindDispersion" in components and not (
        model["NE_SW"].value == 0 and model["NE_SW"].frozen
    ):
        stages["solar_wind"] = ("solar_wind", solar_wind)
        consumed |= set(sequences.get("NE_SW", ())) | {"SWEPOCH"}

    if "DispersionDM" in components:
        stages["dispersion_taylor"] = ("dispersion_taylor", dispersion_taylor)
        consumed |= set(sequences.get("DM", ())) | {"DMEPOCH"}

    if "DispersionDMX" in components:
        stages["dispersion_piecewise"] = ("dispersion_piecewise", dispersion_piecewise)
        consumed |= set(sequences.get("DMX_", ()))

    family = binary_family(model)
    use_fbx = False
    if family is not None:
        use_fbx = uses_fbx(model)
        ddr_config = (
            resolve_ddr_config(
                model, use_fbx=use_fbx, ecliptic=ecliptic, obliquity=resolved
            )
            if family == "DDR"
            else None
        )
        stages["binary"] = (
            f"binary.{family}",
            binary_stage(
                family,
                use_fbx=use_fbx,
                ecliptic=ecliptic,
                ell1_t2=conventions == "tempo2",
                obliquity=resolved,
                ddr_config=ddr_config,
            ),
        )
        if family == "DDR":
            # `ddr_consumed` *substitutes* for the common/extra union: there is
            # no `_BINARY_EXTRA["DDR"]`, and the seven-family `use_fbx` subtract
            # of {PB, PBDOT} would drop names DDR's PB chart just added.
            consumed |= ddr_consumed(ddr_config)
            if use_fbx:
                consumed |= set(sequences.get("FB", ()))
                consumed -= {"FB0"}
        else:
            consumed |= set(_BINARY_COMMON) | set(_BINARY_EXTRA[family])
            if use_fbx:
                consumed |= set(sequences.get("FB", ()))
                consumed -= {"PB", "PBDOT"}
            if family == "DDK":
                consumed |= set(_ECLIPTIC[2:] if ecliptic else _EQUATORIAL[2:])

    if "FD" in components:
        stages["frequency_dependent"] = ("frequency_dependent", frequency_dependent)
        consumed |= set(sequences.get("FD", ()))

    if "FDJump" in components:
        stages["frequency_dependent_jump"] = (
            "frequency_dependent_jump",
            frequency_dependent_jump,
        )
        consumed |= set(sequences.get("FDJUMP", ()))

    stages["spindown"] = ("spindown", spindown)
    consumed |= set(delta_sequences.get("dF", ()))

    stages["phase_offset"] = ("phase_offset", phase_offset)
    consumed.add("PHOFF")

    if "PhaseJump" in components:
        exclusive = frozen.jump_index is not None
        stages["phase_jump"] = (
            "phase_jump",
            phase_jump_exclusive if exclusive else phase_jump,
        )
        consumed |= set(sequences.get("JUMP", ()))

    slots = tuple(slot for slot in STAGE_ORDER if slot in stages)
    ordered = tuple(stages[slot] for slot in slots)
    if len(ordered) != len(stages):
        missing = set(stages) - set(STAGE_ORDER)
        raise KeyError(f"stages missing from STAGE_ORDER: {missing}")
    return Chain(
        ordered,
        frozenset(consumed),
        family,
        use_fbx,
        ecliptic,
        conventions,
        barycentric_cut(slots),
    )


#: The delay slots inside PINT's ``get_barycentric_toas`` cutoff on a binary
#: pulsar: solar system, solar wind and the two dispersion stages.
PRE_BINARY_SLOTS = (
    "solar_system",
    "solar_wind",
    "dispersion_taylor",
    "dispersion_piecewise",
)


def barycentric_cut(slots: tuple[str, ...]) -> int:
    """How many stages PINT's barycentric arrival time includes (R-B1.4).

    ``PulsarData.toas`` is the *barycentric arrival*, which PINT defines as
    TDB minus the delays before the first ``pulsar_system`` component:
    after solar system, solar wind and DM/DMX, before the binary Roemer and
    Shapiro, and before FD (PINT orders FD after the binary). Vela's
    ``corrected_toa_value`` instead subtracts *every* delay, and on an
    ELL1/DD pulsar the two differ by the binary Roemer delay -- seconds, since
    ``A1`` is in light-seconds -- which is a wrong ``toas`` array by any
    measure and would put every TOA in a different ECORR epoch.

    The cut is therefore taken after the dispersion stages when a binary is
    present (FD sits after the binary in ``STAGE_ORDER``, so the snapshot
    excludes it), and after every delay stage for an isolated pulsar --
    matching PINT, whose empty cutoff sums them all.
    """
    if "binary" not in slots:
        return sum(1 for slot in slots if slot not in _PHASE_SLOTS)
    last = max(
        (index for index, slot in enumerate(slots) if slot in PRE_BINARY_SLOTS),
        default=-1,
    )
    return last + 1


_PHASE_SLOTS = ("spindown", "phase_offset", "phase_jump")


def run_chain(frozen, chain: Chain, params, *, snapshot: bool = False):
    """Run every stage in order and return the accumulated correction.

    With ``snapshot=True`` the return is ``(correction, barycentric)``, the
    second being the accumulated correction at PINT's barycentric cutoff
    (:func:`barycentric_cut`) -- the one extra thing the pulsar record needs
    out of the reference pass, so that ``toas``/``freqs`` come from the chain
    rather than from a second PINT pass (SPEC R-B1.4).
    """
    corr = Correction.initial(frozen.n_rows)
    bary = None
    for index, (_, stage) in enumerate(chain.stages):
        if snapshot and index == chain.bary_cut:
            bary = corr
        corr = stage(frozen, corr, params)
    if not snapshot:
        return corr
    return corr, (corr if bary is None else bary)


def form_residuals(frozen, corr):
    """The phase residual over the pulsar-frame spin frequency, in seconds.

    Gauge-free, no mean removed -- Vela's ``form_residuals`` in that respect.

    **The divisor is ``spin_frequency``, not Vela's
    ``doppler_shifted_spin_frequency``.** Vela.jl divides by the topocentric
    (doppler-shifted) instantaneous frequency; PINT (``Residuals``, default
    ``calctype="taylor"``) and tempo2 divide by the pulsar-frame Taylor
    series. The two differ by ``r * (v/c)`` -- ~1e-4 of the residual, so it
    hides on a well-timed pulsar and surfaces as 87 ns rms on AEI-DR2 EPTA
    J0613-0200, whose residuals are 86 us. These residuals are handed to
    Enterprise/Discovery likelihoods next to PINT-formed ones, so parity with
    the two timing packages wins over the Vela identity here (§8, G2).

    ``design_matrix`` follows automatically -- it is ``-jacfwd`` of this --
    but :meth:`Engine.gauge_direction` is hand-written and tracks it by hand.
    """
    psi = corr.phase
    return (psi[:-1] - psi[-1]) / corr.spin_frequency[:-1]
