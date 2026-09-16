"""Parameter layout: PINT-unit deltas in, internal-unit values out.

The engine never sees an absolute parameter value in the trace. It stores the
reference ``theta*`` as exact decimal strings plus frozen float64 constants in
internal units, and the traced function takes ``delta`` in *PINT* units
(``RAJ`` in hourangle, ``T0`` in days, ...) in ``param_names`` order. There is
no ``(theta + delta) - theta`` anywhere, so ``residual_delta(0)`` is exactly
zero by construction.

Epoch parameters are stored the way pyvela stores them: seconds relative to
``PEPOCH``. Prefix families (``F1..``, ``DM..``, ``FB..``, ``DMX_..``,
``JUMP..``, ``FD..``, ``FDJUMP..``, ``NE_SW..``) are packed into tuples in
Vela's order, so a component can Horner over them.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Mapping

import numpy as np
from astropy.time import Time
from pint.models.parameter import MJDParameter, funcParameter

from .constants import DAY_S
from .units import reference_internal, unit_conversion_factor, unit_string

#: Parameters a stage may read that a par file is allowed to omit entirely.
#: Vela's `fix_params` zeroes most of these; the rest are simply absent
#: components (no PX, no proper motion, no binary).
# fmt: off
ZERO_DEFAULTS = (
    # astrometry
    "RAJ", "DECJ", "ELONG", "ELAT", "PMRA", "PMDEC", "PMELONG", "PMELAT", "PX",
    # epochs and the phase gauge
    "POSEPOCH", "DMEPOCH", "SWEPOCH", "PHOFF",
    # orbit
    "PB", "FB0", "PBDOT", "XPBDOT", "T0", "TASC", "A1", "A1DOT",
    "ECC", "EDOT", "OM", "OMDOT", "GAMMA", "DR", "DTH",
    "EPS1", "EPS2", "EPS1DOT", "EPS2DOT", "LNEDOT",
    # Shapiro, in its three parametrisations
    "M2", "SINI", "H3", "STIGMA", "SHAPMAX", "KIN", "KOM",
    # DDR. One layout serves both mode pairs; validate_ddr_model requires COSI
    # always, GGAMMA under DDRPK N, TGEO for geo/kine.
    "COSI", "GGAMMA", "TGEO",
)
# fmt: on


class Params:
    """Internal-unit parameter values for one evaluation of the chain.

    Attribute access only; a missing name is a build error, not a silent zero.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]):
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(
                f"no parameter {name!r} in this layout "
                f"(have {len(self._values)} entries)"
            ) from None

    def __contains__(self, name) -> bool:
        return name in self._values


def _stores_long_double(param) -> bool:
    return (
        isinstance(param, MJDParameter)
        or isinstance(getattr(param, "quantity", None), Time)
        or bool(getattr(param, "long_double", False))
    )


def exact_numeric_string(value, *, long_double: bool) -> str:
    """Decimal text of a stored numeric, without a float64 round trip.

    ``str`` on a numpy longdouble prints every digit numpy holds. ``repr`` of
    a float64 is the round-trip of that type. Casting longdouble through
    ``float`` first throws away the digits that make F0-class axes
    precision-critical (SPEC §4.7, psrdata R-3.3.1).
    """
    if long_double:
        return str(np.longdouble(value))
    return repr(float(value))


def exact_string(param) -> str:
    """The parameter value as a decimal string, at full stored precision."""
    return exact_numeric_string(param.value, long_double=_stores_long_double(param))


def exact_uncertainty_string(param) -> str | None:
    """The parameter uncertainty as a decimal string, or ``None``.

    Same path as :func:`exact_string`: a longdouble uncertainty is not
    formatted via ``float`` (psrdata R-3.3.3 / producer contract §9.7).
    """
    unc = getattr(param, "uncertainty_value", None)
    if unc is None:
        return None
    try:
        stored = np.longdouble(unc) if _stores_long_double(param) else float(unc)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(stored):
        return None
    return exact_numeric_string(unc, long_double=_stores_long_double(param))


@dataclass(frozen=True)
class ParamLayout:
    names: tuple[str, ...]
    units: Mapping[str, str]
    scale: np.ndarray  # PINT unit -> internal, per free parameter
    theta_exact: Mapping[str, str]
    ref_internal: Mapping[str, float]
    live_index: Mapping[str, int]
    sequences: Mapping[str, tuple[str, ...]]
    delta_sequences: Mapping[str, tuple[str, ...]]
    f_ld: tuple  # reference spin coefficients (F0, F1, ...) in longdouble

    @property
    def f0_ref(self) -> float:
        """The float64 F0 a phase JUMP is multiplied by (Vela's ``F_ + F[1]``)."""
        return float(self.f_ld[0])

    def build(self, delta, *, wrap=None) -> Params:
        """``delta`` in PINT units, ``names`` order, to internal-unit values.

        ``wrap(name, ref, step) -> (value, delta_value)`` lets a caller
        substitute another numeric type for the live axes; the perturbative
        engine uses it to inject :class:`~vela_jax.perturbative.dual.Pert`.
        """
        values: dict[str, object] = {"F0_ref": self.f0_ref}
        for name, ref in self.ref_internal.items():
            index = self.live_index.get(name)
            if index is None:
                values[name], values["d" + name] = ref, 0.0
            else:
                step = delta[index] * self.scale[index]
                if wrap is None:
                    values[name], values["d" + name] = ref + step, step
                else:
                    values[name], values["d" + name] = wrap(name, ref, step)
        for family, members in self.sequences.items():
            values[family] = tuple(values[m] for m in members)
        for family, members in self.delta_sequences.items():
            values[family] = tuple(values["d" + m] for m in members)
        return Params(values)

    def reference_theta(self) -> np.ndarray:
        with localcontext() as ctx:
            ctx.prec = 60
            return np.array(
                [float(Decimal(self.theta_exact[n])) for n in self.names], dtype=float
            )

    def delta_from_theta(self, theta) -> np.ndarray:
        """``theta - theta*`` in Decimal, so no precision is lost on epochs."""
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.shape != (len(self.names),):
            raise ValueError(
                f"theta has shape {theta.shape}, expected ({len(self.names)},)"
            )
        with localcontext() as ctx:
            ctx.prec = 60
            return np.array(
                [
                    float(Decimal(float(value)) - Decimal(self.theta_exact[name]))
                    for name, value in zip(self.names, theta)
                ],
                dtype=float,
            )


def build_layout(model, *, free_names, sequences, delta_sequences) -> ParamLayout:
    """Assemble the layout from a prepared PINT model."""
    pepoch = model["PEPOCH"].value

    ref: dict[str, float] = {name: 0.0 for name in ZERO_DEFAULTS}
    units: dict[str, str] = {}
    needed = set(ZERO_DEFAULTS) | set(free_names)
    for members in list(sequences.values()) + list(delta_sequences.values()):
        needed |= set(members)

    for name in needed:
        if name not in model or model[name].quantity is None:
            continue
        param = model[name]
        # A funcParameter (ELL1's derived ECC/OM/T0, for instance) is a view of
        # other parameters, never an independent axis; pyvela skips them too.
        if isinstance(param, funcParameter):
            continue
        ref[name] = reference_internal(param, pepoch)
        units[name] = unit_string(param)

    missing = [n for n in free_names if n not in ref]
    if missing:
        raise KeyError(f"free parameters without a reference value: {missing}")

    scale = np.array(
        [
            (
                DAY_S
                if isinstance(model[name], MJDParameter)
                else unit_conversion_factor(model[name])
            )
            for name in free_names
        ],
        dtype=float,
    )
    return ParamLayout(
        names=tuple(free_names),
        units={name: units.get(name, "1") for name in free_names},
        scale=scale,
        theta_exact={name: exact_string(model[name]) for name in free_names},
        ref_internal=ref,
        live_index={name: i for i, name in enumerate(free_names)},
        sequences=dict(sequences),
        delta_sequences=dict(delta_sequences),
        # SI is already the internal unit for every F_k; PINT hands F0 back as
        # a longdouble, which is exactly the precision the build-time reduction needs.
        f_ld=tuple(
            np.longdouble(model[name].quantity.si.value)
            for name in delta_sequences["dF"]
        ),
    )
