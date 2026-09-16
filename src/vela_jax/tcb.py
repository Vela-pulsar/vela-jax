"""Par-file timescale normalisation, at the tempo2 read boundary.

A tempo2 par file is TCB unless it says otherwise: an absent ``UNITS`` line
means TCB, and ``UNITS SI`` is tempo2's own spelling of TCB. PINT's model is
TDB. The conversion is a text-to-text transform run by tempo2 itself
(``tempo2 -gr transform … tdb``), never an IFTE rescaling inside this package
and never inside the JAX graph.

The *rules* -- what ``UNITS`` means, and collapsing the doubled ``NE_SW``
line old tempo2 builds emit -- are :mod:`psrdata.partext`, shared with
MetaPulsar. What stays here is the half that is not pure text: running tempo2
itself. This package owns tempo2; psrdata owns no subprocess.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from psrdata.partext import dedupe_nonrepeatable, effective_units

from .errors import Tempo2Error

#: Timescales this package understands on a par file.
TIMESCALES = ("TDB", "TCB")


def convert_to_tdb(text: str) -> str:
    """Run ``tempo2 -gr transform … tdb`` on par text and return the result."""
    with tempfile.TemporaryDirectory(prefix="vela_jax_tcb_") as tmp:
        source = Path(tmp) / "in.par"
        target = Path(tmp) / "out.par"
        source.write_text(text)
        try:
            subprocess.run(
                ["tempo2", "-gr", "transform", str(source), str(target), "tdb"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:  # pragma: no cover - environment
            raise Tempo2Error(
                "tempo2 is not on PATH; it is required to convert a TCB par "
                "file to TDB"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise Tempo2Error(f"tempo2 transform failed: {exc.stderr}") from exc
        if not target.exists():  # pragma: no cover - tempo2 quirk
            raise Tempo2Error("tempo2 transform produced no output file")
        return dedupe_nonrepeatable(target.read_text())


def normalize_to_tdb(text: str) -> tuple[str, str]:
    """Return ``(tdb_text, source_units)`` for a tempo2 par file.

    A TCB file is converted; a TDB file comes back **byte-identical**. There
    is no third case: a par with no ``UNITS`` line is TCB by
    :func:`psrdata.partext.effective_units`, so it takes the conversion path
    and tempo2's transform writes the explicit ``UNITS TDB`` itself. An
    earlier version of this function had a branch that stamped ``UNITS TDB``
    onto an implicit par; it was unreachable, and a branch that cannot run is
    a claim about the rules that nothing checks.
    """
    units = effective_units(text)
    if units == "TDB":
        return text, units
    converted = convert_to_tdb(text)
    if effective_units(converted) != "TDB":
        raise Tempo2Error("tempo2 transform did not produce an explicit UNITS TDB par")
    return converted, units
