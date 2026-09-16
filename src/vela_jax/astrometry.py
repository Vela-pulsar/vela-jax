"""Solar-system Roemer, parallax and Shapiro delays, and the Doppler factor.

Vela: ``src/model/solarsystem.jl`` (`SolarSystem`). PINT splits this across
``AstrometryEquatorial``/``AstrometryEcliptic`` and ``SolarSystemShapiro``;
Vela — and therefore this module — keeps it as one component.
"""

from __future__ import annotations

from . import numerics as vm
from .constants import AU_LS, M_SUN, PLANET_MASSES
from .correction import Correction, corrected_time


def evaluate_proper_motion(long0, lat0, pm_long, pm_lat, dt) -> vm.Vec3:
    """Unit vector to the pulsar under uniform linear space motion.

    Vela's ``iszero(pm)`` short-circuit is dropped: the formula already
    reduces to ``x0`` exactly when both proper motions vanish, and a traced
    branch would need both sides evaluated anyway.
    """
    sin_a, cos_a = vm.sin(long0), vm.cos(long0)
    sin_d, cos_d = vm.sin(lat0), vm.cos(lat0)
    x0 = (cos_a * cos_d, sin_a * cos_d, sin_d)
    xdot = (
        -sin_a * pm_long - cos_a * sin_d * pm_lat,
        cos_a * pm_long - sin_a * sin_d * pm_lat,
        cos_d * pm_lat,
    )
    x1 = tuple(x0i + dt * xdi for x0i, xdi in zip(x0, xdot))
    return vm.scale3(1.0 / vm.norm3(x1), x1)


def ecliptic_to_equatorial(lhat: vm.Vec3, obliquity: float) -> vm.Vec3:
    """Rotate an ecliptic line of sight into ICRS.

    Vela ``ecliptic_to_icrs``. ``obliquity`` is resolved at build time from
    the par's ``ECL`` keyword (:func:`vela_jax.constants.obliquity_radians`),
    or from the timing package's own constant where it has already rotated
    its vectors with one. Vela hard-codes IERS2010 here; see the note on
    ``constants.OBL``.
    """
    sin_e, cos_e = vm.sin(obliquity), vm.cos(obliquity)
    x, y, z = lhat
    return (x, cos_e * y - sin_e * z, sin_e * y + cos_e * z)


def equatorial_to_ecliptic(vec: vm.Vec3, obliquity: float) -> vm.Vec3:
    """Rotate an ICRS vector into the ecliptic frame of ``ecliptic_to_equatorial``.

    Vela ``icrs_to_ecliptic``. DDK annual-parallax ``I0``/``J0`` must live in
    the same sky frame as ``KOM``; for an ecliptic model that is this frame,
    not ICRS.
    """
    sin_e, cos_e = vm.sin(obliquity), vm.cos(obliquity)
    x, y, z = vec
    return (x, cos_e * y + sin_e * z, -sin_e * y + cos_e * z)


def shapiro_delay(mass, obs_obj_pos: vm.Vec3, lhat: vm.Vec3):
    """Vela ``solar_system_shapiro_delay``."""
    r = vm.norm3(obs_obj_pos)
    return -2.0 * mass * vm.log((r - vm.dot3(lhat, obs_obj_pos)) / AU_LS)


def solar_system(
    frozen,
    corr: Correction,
    p,
    *,
    ecliptic: bool,
    planet_shapiro: bool,
    obliquity: float,
):
    long0, lat0 = (p.ELONG, p.ELAT) if ecliptic else (p.RAJ, p.DECJ)
    pm_long, pm_lat = (p.PMELONG, p.PMELAT) if ecliptic else (p.PMRA, p.PMDEC)

    dt = corrected_time(frozen, corr) - p.POSEPOCH
    lhat = evaluate_proper_motion(long0, lat0, pm_long, pm_lat, dt)
    if ecliptic:
        lhat = ecliptic_to_equatorial(lhat, obliquity)

    rvec = frozen.ssb_obs_pos
    lhat_dot_r = vm.dot3(lhat, rvec)

    delay = -lhat_dot_r
    r_perp_sqr = vm.dot3(rvec, rvec) - lhat_dot_r * lhat_dot_r
    delay = delay + 0.5 * p.PX * r_perp_sqr
    delay = delay + shapiro_delay(M_SUN, frozen.obs_sun_pos, lhat)
    if planet_shapiro:
        for name, mass in PLANET_MASSES.items():
            delay = delay + shapiro_delay(mass, frozen.planet_pos[name], lhat)

    doppler = vm.dot3(lhat, frozen.ssb_obs_vel)

    # A barycentred row (TZRSITE '@') keeps ssb_psr_pos zero, exactly as Vela's
    # early return does; downstream components read that as "unavailable".
    bary = frozen.is_bary
    return corr.add_delay(
        vm.where(bary, 0.0, delay),
        vm.where(bary, 0.0, doppler),
        vm.where3(bary, corr.ssb_psr_pos, lhat),
    )
