"""The TCB->TDB conversion: the half of the timescale rules that runs tempo2.

The *rules* -- what ``UNITS`` means, which lines are comments, collapsing the
doubled ``NE_SW`` an old tempo2 build writes -- are pure text and live in
:mod:`psrdata.partext`, tested there. What is left here is the part that is
this package's: shelling out to ``tempo2 -gr transform``.
"""

import pytest
from psrdata.partext import active_lines, effective_units


@pytest.mark.tempo2
def test_tcb_conversion_rescales_the_spin_frequency(examples, tmp_path):
    """IFTE is ~1.55e-8 in fractional rate; the conversion must show it."""
    from vela_jax.tcb import convert_to_tdb

    source = (examples / "NGC6440E.par").read_text().replace("UNITS", "C UNITS")
    converted = convert_to_tdb(source + "UNITS TCB\n")
    assert effective_units(converted) == "TDB"

    def f0(text):
        for line in active_lines(text):
            if line.split()[0].upper() == "F0":
                return float(line.split()[1])
        raise AssertionError("no F0")

    ratio = f0(converted) / f0(source)
    assert 1e-9 < abs(ratio - 1.0) < 1e-7
