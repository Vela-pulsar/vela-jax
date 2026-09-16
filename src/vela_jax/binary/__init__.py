"""Binary component resolution: PINT component name to Vela stage.

The dispatch mirrors ``pyvela.model.pint_components_to_vela``; anything else
(BT, BTX, DDGR, unresolved T2, ...) is refused rather than approximated.
"""

from __future__ import annotations

from ..errors import UnsupportedModelError
from .dd import binary_dd
from .ell1 import binary_ell1
from .facts import BinaryFacts

SUPPORTED = ("ELL1", "ELL1H", "ELL1k", "DD", "DDH", "DDS", "DDK")

_PINT_COMPONENT = {
    "BinaryELL1": "ELL1",
    "BinaryELL1H": "ELL1H",
    "BinaryELL1k": "ELL1k",
    "BinaryDD": "DD",
    "BinaryDDH": "DDH",
    "BinaryDDS": "DDS",
    "BinaryDDK": "DDK",
}

_SHAPIRO = {
    "ELL1": "m2_sini",
    "ELL1k": "m2_sini",
    "DD": "m2_sini",
    "ELL1H": "h3_stig",
    "DDH": "h3_stig",
    "DDS": "shapmax",
    "DDK": "kin",
}

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


def binary_stage(family: str, *, use_fbx: bool, ecliptic: bool, ell1_t2: bool = False):
    """The stage callable for one family, with its static choices bound.

    ``ell1_t2`` is resolved here, at build time, into one of two closures --
    never a traced predicate, so the unused polynomial is not even evaluated.
    """
    if family.startswith("ELL1"):

        def stage(frozen, corr, p):
            return binary_ell1(
                frozen, corr, p, family=family, use_fbx=use_fbx, ell1_t2=ell1_t2
            )

    else:

        def stage(frozen, corr, p):
            return binary_dd(
                frozen, corr, p, family=family, use_fbx=use_fbx, ecliptic=ecliptic
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
        kepler_convention="ell1" if family.startswith("ELL1") else "dd",
        use_fbx=use_fbx,
        shapiro=_SHAPIRO[family],
        epoch_shift_exact=not secular and family != "DDK",
        secular_terms=secular,
        ell1_t2=tempo2 and family.startswith("ELL1"),
    )
