"""Building a :class:`psrdata.PulsarData` from a vela-jax engine.

The record *type* is psrdata's -- a frozen array record must not require a JAX
engine to exist, and every producer in this stack emits the same one. What
lives here is the part that is this package's: reading the three things off
the PINT model that no engine evaluates (the DMX window table, the parallax
distance, the sky scalars), and the parameter facts for every set parameter.

**The same-evaluation rule (R-B1).** Every array on the record is a frozen TOA
column, the engine's reference evaluation, or a value read off the parsed
model -- never a second pass over PINT or tempo2. v1 of this package computed
``toas`` with ``model.get_barycentric_toas`` and ``freqs`` with
``model.barycentric_radio_freq``, which when tempo2 read the files meant PINT
physics over injected columns: not the chain, not tempo2, not tested. Both now
come from the chain's own reference delays at PINT's barycentric cutoff
(:meth:`vela_jax.engine.Engine.reference_barycentric`), and a test
monkeypatches those PINT methods to raise while the record is built.
"""

from __future__ import annotations

from typing import Any

import astropy.units as u
import numpy as np
from pint.models.parameter import boolParameter, funcParameter, strParameter
from psrdata import (
    DEFAULT_DISTANCE_KPC,
    PLANET_SLOTS,
    ParameterFact,
    PulsarData,
    backend_flags,
)
from psrdata.record import PHASE_OFFSET_RE, PHASE_OFFSET_UNITS

from .params import exact_string, exact_uncertainty_string


def dmx_table(model) -> dict[str, dict[str, Any]] | None:
    """Enterprise's ``dmx`` shape, built from model values only (B.3.2).

    ``None`` when the model has no DMX component. An empty dict on a pulsar
    that *has* DMX is a refused state -- it is what v1 shipped, and it silently
    disables ``WidebandTimingModel``.
    """
    if "DispersionDMX" not in model.components:
        return None
    table: dict[str, dict[str, Any]] = {}
    for name in model.components["DispersionDMX"].params:
        if not name.startswith("DMX_") or model[name].quantity is None:
            continue
        index = name[len("DMX_") :]
        uncertainty = model[name].uncertainty_value
        table[name] = {
            "DMX": float(model[name].value),
            "DMXerr": None if uncertainty is None else float(uncertainty),
            "DMXR1": float(model[f"DMXR1_{index}"].value),
            "DMXR2": float(model[f"DMXR2_{index}"].value),
            "fit": not model[name].frozen,
        }
    return table


def distance(model) -> tuple[float, float]:
    """``(kpc, err)`` from PX when the par has one; Enterprise's fallback else."""
    parallax = model["PX"] if "PX" in model else None
    if parallax is None or not parallax.value:
        return DEFAULT_DISTANCE_KPC
    milliarcsec = float(parallax.quantity.to_value(u.mas))
    kpc = 1.0 / milliarcsec
    error = float(parallax.uncertainty_value or 0.0)
    return kpc, (kpc * error / milliarcsec if error else 0.2 * kpc)


def _unit_label(param, *, name: str, param_units: dict[str, str]) -> str | None:
    match = PHASE_OFFSET_RE.match(name)
    if match:
        return PHASE_OFFSET_UNITS[match.group(1)]
    if name in param_units:
        text = str(param_units[name]).strip()
        return "dimensionless" if text in ("", "1") else text
    units = getattr(param, "units", None)
    if units is None:
        return None
    text = str(units).strip()
    if text in ("", "1"):
        return "dimensionless"
    return text


def _fact_for_param(
    name: str,
    param,
    *,
    theta_exact: dict[str, str],
    param_units: dict[str, str],
) -> ParameterFact | None:
    """One :class:`ParameterFact`, or ``None`` if the parameter has no value."""
    if isinstance(param, funcParameter):
        return None
    if name in theta_exact:
        value = theta_exact[name]
    elif (
        getattr(param, "value", None) is None
        and getattr(param, "quantity", None) is None
    ):
        return None
    elif isinstance(param, (strParameter, boolParameter)) or isinstance(
        getattr(param, "value", None), str
    ):
        value = str(param.value)
    else:
        try:
            value = exact_string(param)
        except (TypeError, ValueError, AttributeError):
            if param.value is None:
                return None
            value = str(param.value)

    units = _unit_label(param, name=name, param_units=param_units)
    if isinstance(param, (strParameter, boolParameter)) or (
        isinstance(getattr(param, "value", None), str) and name not in theta_exact
    ):
        units = None

    match = PHASE_OFFSET_RE.match(name)
    uncertainty = None if match else exact_uncertainty_string(param)
    if match:
        units = PHASE_OFFSET_UNITS[match.group(1)]
    return ParameterFact(str(value), units, uncertainty)


def parameter_facts(model, engine) -> tuple[tuple[str, ...], dict[str, ParameterFact]]:
    """Facts for every set parameter; ``fitpars`` is a subset of ``setpars``."""
    theta_exact = dict(engine.reference_theta_exact())
    param_units = dict(engine.param_units)
    facts: dict[str, ParameterFact] = {}
    for name in model.params:
        param = model[name]
        fact = _fact_for_param(
            name, param, theta_exact=theta_exact, param_units=param_units
        )
        if fact is not None:
            facts[name] = fact
    for name in engine.param_names:
        match = PHASE_OFFSET_RE.match(name)
        if match:
            # The column is a delta around the stored residuals (R-3.2.2).
            facts[name] = ParameterFact("0", PHASE_OFFSET_UNITS[match.group(1)])
            continue
        if name in facts:
            continue
        units = param_units.get(name, "dimensionless")
        if str(units).strip() in ("", "1"):
            units = "dimensionless"
        facts[name] = ParameterFact(theta_exact[name], units)
    setpars = tuple(name for name in model.params if name in facts)
    extra = tuple(name for name in engine.param_names if name not in set(setpars))
    return setpars + extra, facts


def build_pulsar_data(engine) -> PulsarData:
    """Every field from the freeze or the reference evaluation (R-B1)."""
    model = engine.pint_model
    columns = engine.toa_columns
    n = engine.toa_count

    delay_bary, _ = engine.reference_barycentric()
    rows = engine.toa_rows()
    reference = engine.reference_correction()

    # (F+E) the barycentric arrival at PINT's pre-binary cutoff, from the
    # chain's own reference delays. NOT the fully corrected TOA: those differ
    # by the binary Roemer delay, which is seconds.
    toas = np.asarray(columns.tdb_seconds - delay_bary, dtype=float)

    observatory = columns.ssb_obs_pos
    sunssb = np.zeros((n, 6))
    sunssb[:, :3] = columns.obs_sun_pos + observatory
    planetssb = np.full((n, 9, 6), np.nan)
    for body, position in columns.body_pos.items():
        planetssb[:, PLANET_SLOTS[body], :3] = position + observatory

    flags = dict(columns.flags)
    coords = model.get_psr_coords().icrs
    pos = np.asarray(coords.cartesian.xyz.value, dtype=float)
    setpars, facts = parameter_facts(model, engine)

    return PulsarData(
        name=str(model["PSR"].value or ""),
        fitpars=tuple(engine.param_names),
        setpars=setpars,
        parameters=facts,
        toas=toas,
        stoas=rows.stoas,
        toaerrs=rows.toaerrs,
        residuals=engine.residuals(),
        freqs=rows.freqs,
        Mmat=engine.design_matrix(),
        flags=flags,
        backend_flags=backend_flags(flags, n),
        telescope=np.asarray(columns.telescope, dtype=str),
        pos=pos,
        pos_t=np.stack([np.asarray(c)[:n] for c in reference.ssb_psr_pos], axis=1),
        sunssb=sunssb,
        planetssb=planetssb,
        theta=float(np.pi / 2.0 - coords.dec.rad),
        phi=float(coords.ra.rad),
        pdist=distance(model),
        dm=float(model["DM"].value) if "DM" in model else 0.0,
        dmx=dmx_table(model),
        timing_package="vela_jax",
        partim_compatibility=engine.timing_package,
        residual_centering=engine.residual_centering,
        producer="vela_jax",
    )


__all__ = ["build_pulsar_data", "dmx_table", "distance", "parameter_facts"]
