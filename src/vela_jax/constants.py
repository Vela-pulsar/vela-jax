"""Physical constants, copied from Vela.jl so the two agree bit for bit.

Every value below is the literal from ``Vela.jl/src/model/solarsystem.jl``
(masses are in geometrised units of seconds) or from
``src/model/frequency_dependent.jl``.
"""

from __future__ import annotations

#: Obliquity of the ecliptic, radians. Vela `solarsystem.jl`: `const OBL`.
#:
#: This is 84381.406", the IERS2010 value, and Vela hard-codes it. **We do
#: not**: a par's ``ECL`` keyword selects which realisation its ELONG/ELAT
#: were defined in, PINT honours it, and the difference is not decorative.
#: ``ECL IERS2003`` (84381.4059") is 0.1 mas away, which is ~100 ns RMS of
#: Roemer delay on a real MSP -- 65x the PINT–tempo2 clock floor -- and is
#: the value EPTA and IPTA release pars typically set. Every Vela fixture
#: that sets ``ECL`` sets ``IERS2010``, which is why the fixture set cannot
#: see this. Kept as the default for a par that says nothing; the resolved
#: value is :func:`obliquity_radians` and it is recorded on the engine.
OBL = 0.4090926006005829


#: PINT's own table, read rather than copied so the two cannot drift
#: (``pint/data/runtime/ecliptic.dat``, exposed as ``pint.models.astrometry.OBL``).
def obliquity_radians(name: str | None) -> float:
    """The obliquity a par's ``ECL`` keyword selects, in radians."""
    import numpy as np
    from pint.models.astrometry import OBL as PINT_OBLIQUITY

    if name is None:
        return OBL
    key = str(name).strip()
    if key not in PINT_OBLIQUITY:
        from .errors import UnsupportedModelError

        raise UnsupportedModelError(
            f"ECL {key}", sorted(k for k in PINT_OBLIQUITY if k != "DEFAULT")
        )
    return float(np.deg2rad(PINT_OBLIQUITY[key].to_value("arcsec") / 3600.0))


#: One astronomical unit in light-seconds. Vela `solarsystem.jl`: `const AU`.
AU_LS = 499.00478383615643

#: Solar mass in seconds (GM/c^3). Vela `solarsystem.jl`: `const M_SUN`.
M_SUN = 4.92549094830932e-06

#: Planet masses in seconds, in Vela's Shapiro-delay iteration order.
#: Vela `solarsystem.jl`: `(M_JUPITER, M_SATURN, M_VENUS, M_URANUS, M_NEPTUNE)`.
PLANET_MASSES = {
    "jupiter": 4.702819050227708e-09,
    "saturn": 1.408128810019423e-09,
    "venus": 1.205680558494223e-11,
    "uranus": 2.1505895513637613e-10,
    "neptune": 2.5373119991867603e-10,
}

#: FD reference frequency, Hz. Vela `frequency_dependent.jl`: `νref = frequency(1e9)`.
NU_REF = 1.0e9

#: Seconds per day. Vela `toa.jl`: `const day_to_s`.
DAY_S = 86400.0

#: Mean obliquity of the ecliptic, arcseconds, as tempo2 defines it
#: (`tempo2.h`: `ECLIPTIC_OBLIQUITY_VAL`; `preProcess.C` resets the global to
#: this on every run, and only `-tempo1` overrides it). It is *not* Vela's
#: `OBL`: this one rotates tempo2's own ephemeris vectors back to ICRS, while
#: `OBL` rotates a par's ELONG/ELAT line of sight the way Vela does.
TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC = 84381.4059
