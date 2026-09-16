"""Binary-model facts the chart layer in nltiming consumes.

A plain dataclass, so that nothing here imports nltiming; the consumer turns
it into a ``BinaryChartCapability`` and adds its own policy fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryFacts:
    family: str
    #: ``"dd"`` (polar) | ``"ell1"`` | ``"ddr"``. DDR's native coordinates are
    #: ``(EPS1, EPS2, TASC)`` with a regular ``q = nu - M`` precession, so it
    #: does **not** claim nltiming's DD polar-to-Laplace chart.
    kepler_convention: str
    use_fbx: bool
    #: Inventory only: ``"m2_sini"`` | ``"h3_stig"`` | ``"shapmax"`` | ``"kin"``
    #: | ``"m2_cosi"``. The chart capability does not forward it.
    shapiro: str
    #: Is ``(OM+360, T0+PB)`` an exact reparametrisation? False for DDR, whose
    #: epoch and precession conventions are its own.
    epoch_shift_exact: bool
    secular_terms: tuple[str, ...]
    #: tempo2's ``ELL1model.C`` Roemer truncation is in force.
    ell1_t2: bool = False
    #: Is the family's sampled domain total -- does a valid box prior on its
    #: independent inputs guarantee a physical state? Defaulted True so the
    #: seven pre-DDR families are unchanged; DDR declares False, because a
    #: sampled ``(A1, PB, M2, COSI)`` can imply a negative pulsar mass or a
    #: non-positive Shapiro ``B_S``.
    supports_domain: bool = True
