"""Binary-model facts the chart layer in nltiming consumes.

A plain dataclass, so that nothing here imports nltiming; the consumer turns
it into a ``BinaryChartCapability`` and adds its own policy fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryFacts:
    family: str
    kepler_convention: str  # "dd" or "ell1"
    use_fbx: bool
    shapiro: str  # "m2_sini" | "h3_stig" | "shapmax" | "kin"
    epoch_shift_exact: bool
    secular_terms: tuple[str, ...]
    #: tempo2's ``ELL1model.C`` Roemer truncation is in force.
    ell1_t2: bool = False
