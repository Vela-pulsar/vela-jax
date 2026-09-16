"""PINT units to Vela's internal ``[T^n]`` system.

Ported from ``pyvela/parameters.py`` (``get_scale_factor`` /
``get_unit_conversion_factor``) with the Julia bridge removed, so that a
parameter's internal value is exactly the number pyvela hands Vela:

    internal_value = pint_value * unit_conversion_factor(param)

with the single exception of ``MJDParameter``, which pyvela represents as
seconds since ``PEPOCH``; see :func:`reference_internal`.
"""

from __future__ import annotations

import re

import astropy.units as u
from astropy.time import Time
from pint import DMconst
from pint.models.parameter import Parameter, compute_effective_dimensionality

from .constants import DAY_S
from .errors import UnsupportedModelError

FDJUMP_RX = re.compile(r"^FD(\d+)JUMP(\d+)")

_DM_LIKE_PREFIXES = ("CM", "CMWXSIN_", "CMWXCOS_", "CMX_", "DMJUMP", "DMEQUAD")
_DIMENSIONLESS_NAMES = (
    "TNREDAMP",
    "TNREDGAM",
    "TNDMAMP",
    "TNDMGAM",
    "TNCHROMAMP",
    "TNCHROMGAM",
    "TNCHROMIDX",
)
_DIMENSIONLESS_PREFIXES = ("EFAC", "EQUAD", "ECORR", "DMEFAC", "FD")


def scale_factor(param: Parameter):
    """pyvela ``get_scale_factor``: PINT units to a pure ``[T^n]`` quantity."""
    if param.tcb2tdb_scale_factor is not None:
        return param.tcb2tdb_scale_factor
    if isinstance(param.quantity, Time):
        return 1
    prefix = getattr(param, "prefix", None)
    if param.name == "CM" or prefix in _DM_LIKE_PREFIXES:
        return DMconst
    if (
        param.name in _DIMENSIONLESS_NAMES
        or prefix in _DIMENSIONLESS_PREFIXES
        or FDJUMP_RX.match(param.name)
    ):
        return 1
    raise UnsupportedModelError(f"parameter {param.name} (no known scale factor)")


def unit_conversion_factor(param: Parameter) -> float:
    """pyvela ``get_unit_conversion_factor``: PINT value to internal value."""
    factor = scale_factor(param)
    dim = (
        1
        if isinstance(param.quantity, Time)
        else compute_effective_dimensionality(param.quantity, factor)
    )
    return float(
        (param.units * factor / u.s**dim).to_value(
            u.dimensionless_unscaled, equivalencies=u.dimensionless_angles()
        )
    )


def reference_internal(param: Parameter, pepoch_mjd) -> float:
    """The parameter's reference value in internal units.

    ``MJDParameter``s become seconds *relative to PEPOCH* — pyvela's
    convention, and the one that keeps every epoch difference in the trace a
    small float64 (SPEC §4.2).
    """
    if isinstance(param.quantity, Time):
        return float((param.value - pepoch_mjd) * DAY_S)
    return float((param.quantity * scale_factor(param)).si.value)


def unit_string(param: Parameter) -> str:
    """PINT's unit string, with dimensionless spelled ``"1"`` as pyvela does."""
    text = str(param.units).strip()
    return text or "1"
