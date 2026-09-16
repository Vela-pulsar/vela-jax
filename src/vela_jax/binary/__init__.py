"""Binary component resolution: PINT component name to Vela stage.

The dispatch mirrors ``pyvela.model.pint_components_to_vela``; anything else
(BT, BTX, DDGR, unresolved T2, ...) is refused rather than approximated.
"""

from __future__ import annotations

from ..constants import OBL
from ..errors import UnsupportedModelError
from .dd import binary_dd
from .ddr import DDRConfig, binary_ddr
from .ell1 import binary_ell1
from .facts import BinaryFacts

SUPPORTED = ("ELL1", "ELL1H", "ELL1k", "DD", "DDH", "DDS", "DDK", "DDR")

_PINT_COMPONENT = {
    "BinaryELL1": "ELL1",
    "BinaryELL1H": "ELL1H",
    "BinaryELL1k": "ELL1k",
    "BinaryDD": "DD",
    "BinaryDDH": "DDH",
    "BinaryDDS": "DDS",
    "BinaryDDK": "DDK",
    "BinaryDDR": "DDR",
}

#: Inventory on :attr:`BinaryFacts.shapiro` only. ``binary_chart_capability``
#: forwards ``kepler_convention``, ``epoch_shift_exact``, ``secular_terms``,
#: ``origin_certified`` and ``supports_domain`` -- never this. Nothing
#: validates the token.
_SHAPIRO = {
    "ELL1": "m2_sini",
    "ELL1k": "m2_sini",
    "DD": "m2_sini",
    "ELL1H": "h3_stig",
    "DDH": "h3_stig",
    "DDS": "shapmax",
    "DDK": "kin",
    "DDR": "m2_cosi",
}

#: The native astrometry names DDR's geometry/kinematics reads, by frame.
#: The same two tuples :mod:`vela_jax.pipeline` uses for the solar-system stage.
_EQUATORIAL = ("RAJ", "DECJ", "PMRA", "PMDEC")
_ECLIPTIC = ("ELONG", "ELAT", "PMELONG", "PMELAT")

#: Epoch-coupled rates; any of them active makes (OM+360, T0+PB) inexact.
SECULAR_PARAMS = (
    "OMDOT",
    "PBDOT",
    "EDOT",
    "A1DOT",
    "EPS1DOT",
    "EPS2DOT",
    "LNEDOT",
    "XPBDOT",
)


def resolve_family(model) -> str:
    """The Vela binary family name for a PINT model, or raise."""
    for pint_name, family in _PINT_COMPONENT.items():
        if pint_name in model.components:
            return family
    raise UnsupportedModelError(f"BINARY {model.BINARY.value}", SUPPORTED)


def uses_fbx(model) -> bool:
    """Which orbital chart the par uses, *after* PINT's setup.

    Read ``FB0`` first and treat ``PB`` only as the fallback. The obvious
    spelling -- "exactly one of PB and FB0 is set" -- stopped being true:
    ``PulsarBinary._canonicalize_fbx_views`` installs ``PB`` as a
    ``funcParameter`` view of ``FB0`` whenever any ``FBn`` is set, so both
    names carry a quantity and the XOR refused *every* FBX par, on all
    families. pyvela's ``pint_components_to_vela`` reads it the same way.

    The dropped half of the XOR was an engine-side backup only: PINT itself
    refuses a par that sets both as ordinary parameters, before the view is
    installed (``_setup_fbx_parameterization``).
    """
    from ..freeze import has_value

    use_fbx = has_value(model, "FB0")
    if not use_fbx and not has_value(model, "PB"):
        raise UnsupportedModelError("a binary par with neither PB nor FB0")
    return use_fbx


def resolve_ddr_config(model, *, use_fbx: bool, ecliptic: bool, obliquity: float):
    """DDR's six static mode flags, validated once after PINT's ``setup()``.

    Build errors only. Sampled-domain failures (``COSI``, inferred mass,
    evolved ``A1``, ``B_S``) return NaN from the traced stage.
    """
    from ..freeze import has_value

    if not (
        "AstrometryEquatorial" in model.components
        or "AstrometryEcliptic" in model.components
    ):
        raise UnsupportedModelError("DDR without an astrometry component")

    use_pk = bool(model["DDRPK"].value)
    pbdot_mode = str(model["DDRPBDOT"].value).strip().lower()
    use_geo = bool(model["DDRGEO"].value)
    use_kine = bool(model["DDRKINE"].value)

    if pbdot_mode not in ("kinematic", "absorb_gw"):
        raise UnsupportedModelError(
            f"DDRPBDOT {pbdot_mode!r}", ["kinematic", "absorb_gw"]
        )
    if use_fbx and (pbdot_mode != "absorb_gw" or use_kine):
        raise UnsupportedModelError("DDR FBX requires DDRPBDOT absorb_gw and DDRKINE N")
    if not use_fbx and use_geo and not use_kine:
        raise UnsupportedModelError("DDRGEO Y implies DDRKINE Y in the PB chart")
    if use_geo or use_kine:
        if not has_value(model, "PX") or float(model["PX"].value) <= 0:
            raise UnsupportedModelError("DDR geometry/kinematics requires PX > 0")
    if use_geo and not has_value(model, "KOM"):
        raise UnsupportedModelError("DDRGEO Y requires KOM")

    config = DDRConfig(
        use_fbx,
        ecliptic,
        use_pk,
        pbdot_mode == "kinematic",
        use_geo,
        use_kine,
        obliquity,
    )
    validate_ddr_model(model, config)
    return config


def validate_ddr_model(model, config: DDRConfig) -> None:
    """The selected branch's required inputs, after ``prepare_model``.

    Names with a zero-layout or PINT materialisation are omitted; their
    physical validity is checked in the traced stage.
    """
    from ..freeze import has_value

    required = {"A1", "TASC", "COSI"}
    required.add("FB0" if config.use_fbx else "PB")
    if not config.use_pk:
        required.add("GGAMMA")
    if config.use_geo or config.use_kine:
        required |= {"PX", "TGEO", "POSEPOCH"}
    if config.use_geo:
        required.add("KOM")
    missing = sorted(name for name in required if not has_value(model, name))
    if missing:
        raise UnsupportedModelError(f"DDR is missing required parameters {missing}")


def ddr_consumed(config: DDRConfig) -> set[str]:
    """The names DDR's *selected* modes actually read.

    Mode-dependent: a static union would claim an inactive field reaches the
    trace. ``FB0`` stands in for the live ``FB`` family.
    """
    names = {"A1", "A1DOT", "TASC", "EPS1", "EPS2", "M2", "COSI"}
    if config.use_fbx:
        names.add("FB0")
    else:
        names.add("PB")
        names.add("XPBDOT" if config.pbdot_kinematic else "PBDOT")
    if not config.use_pk:
        names |= {"GGAMMA", "OMDOT"}
    if config.use_geo or config.use_kine:
        names |= {"PX", "TGEO", "POSEPOCH"}
        names |= set(_ECLIPTIC) if config.ecliptic_coordinates else set(_EQUATORIAL)
    if config.use_geo:
        names.add("KOM")
    return names


def binary_stage(
    family: str,
    *,
    use_fbx: bool,
    ecliptic: bool,
    ell1_t2: bool = False,
    obliquity: float | None = None,
    ddr_config: DDRConfig | None = None,
):
    """The stage callable for one family, with its static choices bound.

    ``ell1_t2`` is resolved here, at build time, into one of two closures --
    never a traced predicate, so the unused polynomial is not even evaluated.
    ``obliquity`` is the same build-time value ``solar_system`` rotates the
    line of sight with; DDK needs it to put Kopeikin ``I0``/``J0`` in the
    model's sky frame. Unused for ELL1.
    """
    if family == "DDR":
        if ddr_config is None:
            raise ValueError("DDR requires a resolved DDRConfig")

        def stage(frozen, corr, p):
            return binary_ddr(frozen, corr, p, config=ddr_config)

    elif family.startswith("ELL1"):

        def stage(frozen, corr, p):
            return binary_ell1(
                frozen, corr, p, family=family, use_fbx=use_fbx, ell1_t2=ell1_t2
            )

    else:
        resolved = OBL if obliquity is None else obliquity

        def stage(frozen, corr, p):
            return binary_dd(
                frozen,
                corr,
                p,
                family=family,
                use_fbx=use_fbx,
                ecliptic=ecliptic,
                obliquity=resolved,
            )

    stage.__name__ = f"binary_{family}"
    return stage


def binary_facts(
    model, family: str, use_fbx: bool, *, conventions: str = "pint"
) -> BinaryFacts:
    secular = tuple(
        name
        for name in SECULAR_PARAMS
        if name in model
        and model[name].quantity is not None
        and (float(model[name].value) != 0.0 or not model[name].frozen)
    )
    tempo2 = conventions == "tempo2"
    return BinaryFacts(
        family=family,
        kepler_convention=(
            "ddr" if family == "DDR" else "ell1" if family.startswith("ELL1") else "dd"
        ),
        use_fbx=use_fbx,
        shapiro=_SHAPIRO[family],
        # DDR's epoch/precession convention is its own: TASC is the zero of the
        # mean longitude and periapsis advances through a regular q = nu - M,
        # so (OM+360, T0+PB) is not even the right pair of knobs to shift.
        epoch_shift_exact=(
            False if family == "DDR" else not secular and family != "DDK"
        ),
        secular_terms=secular,
        ell1_t2=tempo2 and family.startswith("ELL1"),
        # A valid box prior on DDR's independent inputs does not guarantee a
        # positive inferred pulsar mass or a positive Shapiro B_S, so the
        # sampled domain is not total and nltiming must not assume it is.
        supports_domain=family != "DDR",
    )
