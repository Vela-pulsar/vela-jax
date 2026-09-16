"""tempo2 as the timing package that reads the files, physics unchanged.

PINT is this package's default timing package. For EPTA/IPTA products it is
often the wrong one: the file needs tempo2's INCLUDE handling, its clock
chain, its ``TRACK -2`` pulse numbers and its site/ephemeris vectors. This
module lets tempo2 open the file and then hands the *same* frozen arrays to
the *same* component chain. Nothing here evaluates a delay.

    par + tim
      -> strip noise lines, normalise UNITS to TDB (tcb.py)
      -> tempo2 (libstempo, sandboxed): clocks, TT->TDB, SPK, TRACK -2
      -> PINT parses the same TDB par: BINARY family, masks, units
      -> inject tempo2's columns into a PINT TOAs table
      -> Engine(model, toas)                      <- unchanged from here on

The one place tempo2's own numbers are *not* used is the delay: Vela's
component chain recomputes Roemer, Shapiro, dispersion and the binary from the
frozen geometry. That is the point -- tempo2 reads, PINT/Vela physics.

Three deliberate departures from tempo2's own bookkeeping, each documented at
its call site:

* the observing frequency handed to the chain is the *topocentric* one. Vela
  applies its own Doppler shift, and tempo2's ``freq_ssb`` has that shift
  baked in already, so passing it would double-count;
* the TZR pseudo-TOA's geometry comes from PINT. The TZR row contributes one
  constant phase, which the free ``PHOFF`` absorbs;
* pulse numbers are taken from tempo2 only when tempo2 was actually *told*
  the phase connection -- ``TRACK -2`` **and** ``-pn`` flags -- and are
  re-referenced onto PINT's fiducial even then. See
  :func:`resolve_pulse_numbers`.
"""

from __future__ import annotations

import io
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import Column
from astropy.time import Time
from psrdata.partext import (
    respell_clock_for_tempo2,
    respell_fdjump_for_pint,
    strip_noise_lines,
)

from .constants import DAY_S, TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC
from .errors import Tempo2Error
from .freeze import prepare_model
from .tcb import normalize_to_tdb

#: The frozen columns must reproduce tempo2's own Roemer delay to this, in
#: tempo2's own frame. It is a units-and-composition check, so the budget is
#: float64 noise on a ~500 s quantity, not a physics tolerance.
ROEMER_TOLERANCE_S = 1e-9

#: tempo2 / PINT spellings of a barycentric observatory. The site is the SSB:
#: ``observatory_earth`` is zero, ``roemer`` is zero, and adding ``earth_ssb``
#: would invent a ~500 s delay tempo2 never applied.
BARYCENTRIC_SITES = ("bat", "@")

#: Bodies Vela's ``solar_system`` can apply a Shapiro delay for, and the
#: libstempo attribute holding each one's SSB position (light-seconds).
PLANET_ATTRS = {
    "jupiter": "jupiter_ssb",
    "saturn": "saturn_ssb",
    "venus": "venus_ssb",
    "uranus": "uranus_ssb",
    "neptune": "neptune_ssb",
}


def prepare_par(par) -> tuple[str, str, str]:
    """``(tempo2_text, pint_text, source_units)`` from a par file on disk.

    The two texts differ only in two keyword spellings -- ``FDJUMP`` for PINT,
    ``CLK`` for tempo2 -- so both codes read the same numbers, including the
    same clock realisation. Noise lines are stripped from both for the same
    reason as when PINT reads (PINT would build ``EcorrNoise`` and permute
    the TOAs, and neither engine needs them for a delay).
    """
    stripped = strip_noise_lines(Path(par).read_text())
    normalized, source_units = normalize_to_tdb(stripped)
    return (
        respell_clock_for_tempo2(normalized),
        respell_fdjump_for_pint(normalized),
        source_units,
    )


# --- tempo2 ----------------------------------------------------------------


def load_pulsar(par_text: str, tim, **kwargs):
    """Open ``(par_text, tim)`` with tempo2 through ``libstempo.sandbox``.

    tempo2 segfaults on real data often enough that the parent process must
    survive it. Native ``libstempo.tempopulsar`` is not a path this package
    offers.
    """
    try:
        from libstempo.sandbox import tempopulsar
    except ImportError as exc:  # pragma: no cover - environment
        raise ImportError(
            "reading with tempo2 needs libstempo and a working tempo2: "
            "pip install 'vela-jax[tempo2]'"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="vela_jax_t2_") as tmp:
        par_path = Path(tmp) / "engine.par"
        par_path.write_text(par_text)
        kwargs.setdefault("dofit", False)
        return tempopulsar(parfile=str(par_path), timfile=str(tim), **kwargs)


def ecliptic_to_icrs(vectors: np.ndarray) -> np.ndarray:
    """Undo tempo2's ``equ2ecl`` on an ``(N, 3)`` block of vectors.

    tempo2 rotates *every* ephemeris vector into ecliptic coordinates when the
    par uses ELONG/ELAT (``readEphemeris.C``, ``get_obsCoord.C``), because its
    own line of sight is ecliptic there. Vela's ``solar_system`` instead keeps
    the geometry in ICRS and rotates the line of sight itself, so the vectors
    have to be rotated back -- with tempo2's obliquity, since tempo2 is what
    rotated them.
    """
    angle = np.deg2rad(TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC / 3600.0)
    cos_e, sin_e = np.cos(angle), np.sin(angle)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return np.stack([x, cos_e * y - sin_e * z, sin_e * y + cos_e * z], axis=1)


def _decode(values) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in ("S", "O"):
        return np.char.decode(values.astype("S"), "ascii")
    return values.astype(str)


@dataclass(frozen=True)
class Tempo2Columns:
    """tempo2's per-TOA state, in the units PINT's TOA table uses."""

    tdbld: np.ndarray  # longdouble MJD, observatory TDB (barycentric when the site is)
    sat_mjd: np.ndarray  # longdouble site arrival time, for masks and DMX
    freq_mhz: np.ndarray
    error_us: np.ndarray
    telescope: np.ndarray
    flags: dict[str, np.ndarray]
    pulse_number: np.ndarray
    phase_connected: bool  # tempo2 was given the pulse numbers, not asked for them
    ssb_obs_pos: np.ndarray  # (N, 3) light-seconds
    ssb_obs_vel: np.ndarray  # (N, 3) light-seconds / s
    sun_ssb_pos: np.ndarray  # (N, 3) light-seconds
    planet_ssb_pos: dict[str, np.ndarray]
    freq_ssb_hz: np.ndarray  # tempo2's barycentric frequency, for diagnostics
    roemer_residual: np.ndarray  # see `read_columns`; must be picoseconds
    n_deleted: int
    ecliptic: bool


#: The libstempo properties this module needs and stock 2.5.2 does not have.
#: They are three one-line memoryviews over fields tempo2 already fills; the
#: PR that adds them is against ``vallis/libstempo``, not a fork.
_REQUIRED_LIBSTEMPO = {
    "siteVel": (
        "the observatory velocity. tempo2 keeps the site *position* in "
        "`observatory_earth[0:3]` and leaves `[3:6]` at zero, so reading that "
        "half instead is silently wrong: it drops the Earth's rotation, which "
        "is 1.3% of the observatory velocity and ~100 ns of dispersion delay "
        "through the doppler-corrected observing frequency"
    ),
    "correction_tt": (
        "the site clock chain to TT. Without it there is no observatory TDB; "
        "`toas()`/`bat` are barycentric and would double-count the Roemer delay"
    ),
    "correction_tt_tb": "the TT to TDB/TCB correction, the other half of that sum",
}


def _require(pulsar, name: str, _unused: str) -> None:
    """Refuse a libstempo that predates the three properties this read needs.

    A refusal, never a fallback: every plausible substitute for these fields
    is wrong in a way that produces plausible numbers (see the messages
    above), which is the failure mode worth spending an exception on.
    """
    if hasattr(pulsar, name):
        return
    raise Tempo2Error(
        f"this libstempo does not expose `{name}`, which is "
        f"{_REQUIRED_LIBSTEMPO[name]}. Install a libstempo that has it "
        "(vallis/libstempo, the siteVel/clock-correction properties); "
        "vela-jax will not guess these."
    )


def _plain(values) -> np.ndarray:
    """A float64 array from a libstempo column, with or without units.

    ``tempopulsar(units=True)`` returns astropy quantities for `freqs`,
    `toaerrs` and friends; the frozen arrays are plain numbers in tempo2's own
    units, so the quantity is unwrapped rather than converted.
    """
    return np.asarray(getattr(values, "value", values), dtype=float)


def _track_mode(pulsar) -> int:
    """tempo2's ``TRACK`` as an int; ``0`` when the par does not set it.

    ``TRACK -2`` is what says tempo2 was *given* the phase connection rather
    than asked to work it out, so a par without the line is not phase
    connected and must not be read as if it were.

    Deliberately not ``"TRACK" in pulsar``: the sandbox proxy defines
    ``__getitem__`` and no ``__contains__``, so ``in`` falls back to the old
    numeric iteration protocol and issues RPCs forever.
    """
    try:
        value = pulsar["TRACK"].val
    except Exception:  # absent from the par: libstempo raises, the proxy may not
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def read_columns(pulsar, model) -> Tempo2Columns:
    """Pull tempo2's frozen state out of a libstempo ``tempopulsar``.

    ``tdbld`` is assembled here rather than taken from a diagnostic, because
    the obvious-looking field is the wrong one: ``bbat`` is *barycentric*, and
    feeding it to Vela's ``solar_system`` would double-count the Roemer delay.
    PINT's ``tdbld`` is the observatory TDB, which is
    ``SAT + (TT correction) + (TT->TB correction)``.

    ``roemer_residual`` is the load-bearing check on everything else here.
    In tempo2's own frame, before any rotation, its ``roemer``
    (``calculate_bclt.C``) is

        psrPos . R  -  parallax delay

    to third order in proper motion: ``rcos1 + dt_pm`` is the unnormalised
    line of sight dotted with ``R``, and ``dt_pmtt`` is exactly the
    second-order term of normalising it, which is what ``psrPos`` already has.
    So this residual must be picoseconds. If the units are wrong, or the site
    offset is missing, or the wrong vector was picked, it is microseconds --
    and it says so with no obliquity or ephemeris question mixed in.

    A barycentric site (``bat`` / ``@``) is the SSB: tempo2's ``roemer`` is
    already zero, ``observatory_earth`` is zero, and ``earth_ssb`` is still
    the Earth's position. ``R`` is therefore zero, not ``earth_ssb``.
    """
    from .units import reference_internal

    ecliptic = "AstrometryEcliptic" in model.components
    parallax = (
        reference_internal(model["PX"], model["PEPOCH"].value)
        if "PX" in model and model["PX"].quantity is not None
        else 0.0
    )
    _require(pulsar, "siteVel", "the observatory velocity")
    _require(pulsar, "correction_tt", "the site clock chain to TT")
    _require(pulsar, "correction_tt_tb", "the TT to TDB/TCB correction")

    # Two reads that make tempo2 compute rather than just hand back a struct
    # field, so both happen before the geometry views are taken rather than in
    # the middle of them.
    #
    # `pulsenumbers()` and not the `pulse_number` property: `obsn[].pulseN` is
    # filled by `formResiduals`, which a `dofit=False` construction has not
    # run, so the raw view is all zeros. libstempo says as much -- the property
    # is documented as deprecated in favour of this call.
    pulse_number = np.asarray(
        pulsar.pulsenumbers(removemean=False), dtype=np.longdouble
    )
    freq_ssb_hz = _plain(pulsar.ssbfreqs())

    keep = np.asarray(pulsar.deleted) == 0
    n_deleted = int((~keep).sum())

    long_ = np.longdouble
    sat = np.asarray(pulsar.stoas, dtype=long_)
    tdbld = sat + (
        np.asarray(pulsar.correction_tt, dtype=long_)
        + np.asarray(pulsar.correction_tt_tb, dtype=long_)
    ) / long_(DAY_S)

    earth = np.asarray(pulsar.earth_ssb, dtype=float)
    site = np.asarray(pulsar.observatory_earth, dtype=float)
    telescope = _decode(pulsar.telescope())
    ssb_obs_pos = earth[:, :3] + site[:, :3]
    # tempo2 keeps the site *position* in observatory_earth[0:3] and leaves
    # [3:6] at zero; the site velocity is a separate field, and tempo2 itself
    # forms the observatory velocity as `earth_ssb[3:6] + siteVel`
    # (`dm_delays.C:99`). Reading the zero half instead drops the Earth's
    # rotation -- 1.3% of the observatory velocity, which is invisible in the
    # Roemer closure (a position check) and shows up as ~100 ns of dispersion
    # delay through the doppler-corrected observing frequency. Gate A.6.1 is
    # what caught it.
    ssb_obs_vel = earth[:, 3:6] + np.asarray(pulsar.siteVel, dtype=float)
    barycentric = np.isin(telescope, BARYCENTRIC_SITES)
    if np.any(barycentric):
        ssb_obs_pos = np.where(barycentric[:, None], 0.0, ssb_obs_pos)
        ssb_obs_vel = np.where(barycentric[:, None], 0.0, ssb_obs_vel)
    line_of_sight = np.asarray(pulsar.psrPos, dtype=float)
    projected = np.sum(line_of_sight * ssb_obs_pos, axis=1)
    perpendicular = np.sum(ssb_obs_pos * ssb_obs_pos, axis=1) - projected * projected
    roemer_residual = (
        projected
        - 0.5 * parallax * perpendicular
        - np.asarray(pulsar.roemer, dtype=float)
    )

    sun_ssb_pos = np.asarray(pulsar.sun_ssb, dtype=float)[:, :3]
    planet_ssb_pos = {
        body: np.asarray(getattr(pulsar, attr), dtype=float)[:, :3]
        for body, attr in PLANET_ATTRS.items()
    }
    if ecliptic:
        ssb_obs_pos = ecliptic_to_icrs(ssb_obs_pos)
        ssb_obs_vel = ecliptic_to_icrs(ssb_obs_vel)
        sun_ssb_pos = ecliptic_to_icrs(sun_ssb_pos)
        planet_ssb_pos = {k: ecliptic_to_icrs(v) for k, v in planet_ssb_pos.items()}

    flags = {str(key): _decode(pulsar.flagvals(key))[keep] for key in pulsar.flags()}
    return Tempo2Columns(
        tdbld=tdbld[keep],
        sat_mjd=sat[keep],
        freq_mhz=_plain(pulsar.freqs)[keep],
        error_us=_plain(pulsar.toaerrs)[keep],
        telescope=telescope[keep],
        flags=flags,
        pulse_number=pulse_number[keep],
        phase_connected=(_track_mode(pulsar) == -2 and "pn" in set(pulsar.flags())),
        ssb_obs_pos=ssb_obs_pos[keep],
        ssb_obs_vel=ssb_obs_vel[keep],
        sun_ssb_pos=sun_ssb_pos[keep],
        planet_ssb_pos={k: v[keep] for k, v in planet_ssb_pos.items()},
        freq_ssb_hz=freq_ssb_hz[keep],
        roemer_residual=roemer_residual[keep],
        n_deleted=n_deleted,
        ecliptic=ecliptic,
    )


# --- PINT TOAs -------------------------------------------------------------


def build_toas(columns: Tempo2Columns, model) -> Any:
    """A PINT ``TOAs`` object carrying tempo2's columns, in tempo2's order.

    Built from a TOA list rather than by re-reading the ``.tim``: that keeps
    tempo2 the single authority on which TOAs exist and in what order, and it
    means PINT's clock-correction path is never entered for a science TOA. The
    columns PINT would have computed (``tdb``, ``tdbld``, the ephemeris
    vectors, ``pulse_number``) are written in directly.
    """
    from pint.toa import TOA, TOAs

    n = len(columns.tdbld)
    # Two-part MJD, not `float(sat_mjd)`: tempo2's SAT is longdouble, and
    # collapsing it to a float64 day number throws away ~20 ns -- the whole
    # precision budget -- before PINT ever sees it. `TOA.__init__` unpacks a
    # 2-tuple as `arg1, arg2 = MJD` and hands both parts to `Time`.
    day = np.floor(columns.sat_mjd)
    toa_list = [
        TOA(
            (float(day[i]), float(columns.sat_mjd[i] - day[i])),
            error=columns.error_us[i],
            obs=str(columns.telescope[i]),
            freq=columns.freq_mhz[i],
            flags={
                key: str(value[i])
                for key, value in columns.flags.items()
                if str(value[i])
            },
        )
        for i in range(n)
    ]
    toas = TOAs(toalist=toa_list)
    if len(toas) != n:
        raise Tempo2Error(
            f"PINT built {len(toas)} TOAs from {n} tempo2 rows; the tables "
            "would not line up"
        )

    day = np.floor(columns.tdbld)
    toas.table["tdb"] = Time(
        np.asarray(day, dtype=float),
        np.asarray(columns.tdbld - day, dtype=float),
        format="mjd",
        scale="tdb",
    )
    toas.table["tdbld"] = np.asarray(columns.tdbld)

    ls = u.lightsecond
    toas.table["ssb_obs_pos"] = Column(data=columns.ssb_obs_pos, unit=ls)
    toas.table["ssb_obs_vel"] = Column(data=columns.ssb_obs_vel, unit=ls / u.s)
    toas.table["obs_sun_pos"] = Column(
        data=columns.sun_ssb_pos - columns.ssb_obs_pos, unit=ls
    )
    for body, position in columns.planet_ssb_pos.items():
        toas.table[f"obs_{body}_pos"] = Column(
            data=position - columns.ssb_obs_pos, unit=ls
        )
    # Overwritten by resolve_pulse_numbers(); present so that PINT's own
    # phase machinery sees a well-formed table.
    toas.table["pulse_number"] = np.asarray(columns.pulse_number)
    toas.table["delta_pulse_number"] = np.zeros(n)

    toas.planets = True
    toas.ephem = str(model.EPHEM.value) if model.EPHEM.value else None
    # get_TZR_toa() reads this; it is the only PINT clock path this read uses.
    toas.clock_corr_info = {"include_bipm": True}
    toas.vela_jax_timing_package = "tempo2"
    return toas


def resolve_pulse_numbers(toas, model, columns: Tempo2Columns) -> str:
    """Fill the ``pulse_number`` column, and report where it came from.

    tempo2's ``pulseN`` is authoritative exactly when tempo2 was *given* the
    phase connection, which means ``TRACK -2`` **and** ``-pn`` flags -- both,
    not either. ``formResiduals.C:2263`` reads the flags only inside the
    ``TRACK -2`` branch, and we measured the consequence: bumping a ``-pn``
    value by one turn on a tim without ``TRACK -2`` changes nothing at all,
    while the same edit with ``TRACK -2`` moves every affected residual by
    exactly one period. Treating a bare ``-pn`` flag as a connection would
    therefore claim an authority tempo2 never exercised -- most of Vela.jl's
    own fixtures carry decorative ``-pn`` flags and no ``TRACK``.

    Without the connection, ``pulseN`` is tempo2's own count from its own
    origin and can disagree with the model by whole turns, so the model
    defines the pulse numbers instead, exactly as when PINT reads the files.

    Even when tempo2 owns them, the *origin* is PINT's: PINT counts from
    ``TZRMJD`` (``abs_phase=True``) and tempo2 from elsewhere, and Vela's
    residual is ``phase - pulse_number``, so the wrong origin leaves ~1e8
    turns of accumulated phase where float64 has no room for a residual. Only
    that one constant is taken from PINT, as a median; every pulse-to-pulse
    difference stays tempo2's, which is the whole content of the phase
    connection. A TOA where the two codes genuinely disagree about a wrap
    survives as a whole turn in the residual -- which is what one wants to see.
    """
    reference = np.asarray(model.phase(toas, abs_phase=True).int, dtype=np.longdouble)
    if not columns.phase_connected:
        toas.table["pulse_number"] = np.asarray(reference)
        return "model"
    offset = np.round(np.median(columns.pulse_number - reference))
    toas.table["pulse_number"] = np.asarray(columns.pulse_number - offset)
    return "tempo2"


# --- the whole ingest ------------------------------------------------------


def tempo2_obliquity(model) -> float:
    """tempo2's obliquity, and a warning when the par asked for another.

    ``readParfile.C`` has no ``ECL`` keyword: tempo2 rotates *every* ephemeris
    vector into ecliptic coordinates with its own compiled-in
    ``ECLIPTIC_OBLIQUITY_VAL`` regardless of what the par says.
    :func:`ecliptic_to_icrs` undoes exactly that rotation, so the line of
    sight must be rotated back with the same constant or the composition is
    not the identity -- a 0.1 mas frame twist, ~100 ns of Roemer delay, from
    nothing but two constants disagreeing across two files.

    A par whose ``ECL`` names a different realisation is therefore evaluated
    in tempo2's frame here, which is the right answer for a tempo2-fitted par
    (its ELONG/ELAT were fitted in that frame) and a real, small inconsistency
    for a PINT-fitted one. It says so rather than deciding quietly.
    """
    import numpy as np

    from .constants import obliquity_radians

    tempo2_value = float(np.deg2rad(TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC / 3600.0))
    requested = model["ECL"].value if "ECL" in model else None
    if requested is not None and abs(obliquity_radians(requested) - tempo2_value) > 0:
        warnings.warn(
            f"the par says ECL {requested}, but tempo2 has no ECL keyword and "
            f"rotated every ephemeris vector with its own "
            f'{TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC}" obliquity; the line of '
            "sight is rotated back with the same constant, so this engine "
            "reproduces tempo2 rather than the par's stated frame",
            stacklevel=3,
        )
    return tempo2_value


def load(par, tim, *, pulsar=None, **pulsar_kwargs):
    """``(model, toas, package, columns)`` when tempo2 reads the files."""
    from pint.models import get_model

    if "sandbox" in pulsar_kwargs:
        raise TypeError(
            "tempo2 reads always go through libstempo.sandbox; "
            "there is no native-libstempo path"
        )
    tempo2_text, pint_text, source_units = prepare_par(par)
    model = get_model(io.StringIO(pint_text))
    if pulsar is None:
        pulsar = load_pulsar(tempo2_text, tim, **pulsar_kwargs)

    columns = read_columns(pulsar, model)
    worst = float(np.max(np.abs(columns.roemer_residual)))
    if worst > ROEMER_TOLERANCE_S:
        raise Tempo2Error(
            f"tempo2's own Roemer delay is not reproduced by the frozen "
            f"geometry columns (worst {worst * 1e9:.3g} ns > "
            f"{ROEMER_TOLERANCE_S * 1e9:g} ns). The vector mapping is wrong, "
            "not the physics."
        )
    if columns.n_deleted:
        warnings.warn(
            f"tempo2 marked {columns.n_deleted} TOAs deleted (START/FINISH or "
            "a tim directive); the engine follows tempo2 and drops them",
            stacklevel=2,
        )

    toas = build_toas(columns, model)
    prepare_model(model, toas)
    source = resolve_pulse_numbers(toas, model, columns)
    if source == "model":
        warnings.warn(
            "no phase connection was given to tempo2 (it reads -pn flags "
            "only under TRACK -2), so pulse numbers come from the timing "
            "model, as they would when PINT reads the files",
            stacklevel=2,
        )

    package = {
        "timing_package": "tempo2",
        "source_units": source_units,
        "par_text": pint_text,
        "pulse_number_source": source,
        "obliquity": tempo2_obliquity(model),
    }
    return model, toas, package, columns
