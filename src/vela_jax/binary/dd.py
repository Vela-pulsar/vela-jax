"""The Damour & Deruelle family: DD, DDH, DDS, DDK.

Vela: ``src/model/binary/binary_dd_base.jl`` plus ``binary_dd.jl``,
``binary_ddh.jl``, ``binary_dds.jl``, ``binary_ddk.jl``. The Shapiro
parametrisation is the only difference between DD/DDH/DDS; DDK additionally
applies the Kopeikin corrections, which need the astrometry component's
``ssb_psr_pos`` and therefore always run after it.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .. import numerics as vm
from ..correction import Correction, corrected_time
from .orbit import TWO_PI, eccentric_anomaly, mean_anomaly, mean_motion


class DDState(NamedTuple):
    """Vela ``DDState``: everything the DD delays need, computed once."""

    alpha: Any
    beta: Any
    gamma: Any
    sinu: Any
    cosu: Any
    et: Any
    er: Any
    a1: Any
    n: Any
    m2: Any
    sini: Any


def shapiro_params(family: str, p):
    """Vela ``shapiro_delay_params``, dispatched statically at build time."""
    if family in ("DD", "DDK"):
        return p.M2, p.SINI
    if family == "DDH":
        stigma = p.STIGMA
        return p.H3 / stigma**3, 2.0 * stigma / (1.0 + stigma * stigma)
    if family == "DDS":
        return p.M2, 1.0 - vm.exp(-p.SHAPMAX)
    raise ValueError(f"unknown DD family {family!r}")


def kopeikin_i0_j0(ssb_psr_pos: vm.Vec3):
    """Vela ``kopeikin_I0_J0``."""
    sin_d = ssb_psr_pos[2]
    cos_d = vm.sqrt(1.0 - sin_d * sin_d)
    cos_a = ssb_psr_pos[0] / cos_d
    sin_a = ssb_psr_pos[1] / cos_d
    zero = 0.0 * sin_d
    i0 = (-sin_a, cos_a, zero)
    j0 = (-cos_a * sin_d, -sin_a * sin_d, cos_d)
    return i0, j0


def kopeikin_corrections(frozen, corr, p, dt, a1, *, ecliptic: bool):
    """Vela ``kopeikin_corrections``: apparent dx, domega, dinc.

    PINT honours ``K96 N`` by dropping the proper-motion terms; Vela always
    applies them, and so do we (``K96 N`` is refused at build).
    """
    mu_a, mu_d = (p.PMELONG, p.PMELAT) if ecliptic else (p.PMRA, p.PMDEC)
    sin_i, cos_i = vm.sincos(p.KIN)
    sin_om, cos_om = vm.sincos(p.KOM)
    cot_i, csc_i = cos_i / sin_i, 1.0 / sin_i

    dinc_pm = (-mu_a * sin_om + mu_d * cos_om) * dt
    dx_pm = a1 * cot_i * dinc_pm
    dom_pm = csc_i * (mu_a * cos_om + mu_d * sin_om) * dt

    i0, j0 = kopeikin_i0_j0(corr.ssb_psr_pos)
    di = vm.dot3(frozen.ssb_obs_pos, i0)
    dj = vm.dot3(frozen.ssb_obs_pos, j0)
    dx_px = a1 * cot_i * p.PX * (di * sin_om - dj * cos_om)
    dom_px = -csc_i * p.PX * (di * cos_om + dj * sin_om)

    return dx_pm + dx_px, dom_pm + dom_px, dinc_pm


def dd_state(frozen, corr: Correction, p, *, family, use_fbx, ecliptic=False):
    fb = frozen.binary
    dt = corrected_time(frozen, corr) - p.T0
    dt_red = fb.dt_red - corr.delay - p.dT0

    n = mean_motion(dt, p, use_fbx)
    l = mean_anomaly(fb, dt, dt_red, p, use_fbx)

    et = p.ECC + dt * p.EDOT
    er = et * (1.0 + p.DR)
    ephi = et * (1.0 + p.DTH)

    u = eccentric_anomaly(l, et)
    sinu, cosu = vm.sincos(u)

    eta_phi = vm.sqrt(1.0 - ephi * ephi)
    beta_phi = (1.0 - eta_phi) / ephi
    # The true anomaly is *unwrapped* in Vela, because its mean anomaly is:
    # `omega = OM + (OMDOT/n) v` is how periapsis advances secularly, so the
    # whole orbits the build-time reduction divided out have to come back here (and
    # only here -- every other use of u and v is periodic).
    v = 2.0 * vm.arctan2(beta_phi * sinu, 1.0 - beta_phi * cosu) + u
    v = v + TWO_PI * fb.n_orb

    a1 = p.A1 + dt * p.A1DOT
    omega = p.OM + (p.OMDOT / n) * v
    m2, sini = shapiro_params(family, p)

    if family == "DDK":
        dx, dom, dinc = kopeikin_corrections(frozen, corr, p, dt, a1, ecliptic=ecliptic)
        a1 = a1 + dx
        omega = omega + dom
        sini = vm.sin(p.KIN + dinc)

    sin_om, cos_om = vm.sincos(omega)
    return DDState(
        alpha=a1 * sin_om,
        beta=a1 * eta_phi * cos_om,
        gamma=p.GAMMA,
        sinu=sinu,
        cosu=cosu,
        et=et,
        er=er,
        a1=a1,
        n=n,
        m2=m2,
        sini=sini,
    )


def romer_einstein_delay(s: DDState):
    return s.alpha * (s.cosu - s.er) + (s.beta + s.gamma) * s.sinu


def d_romer_einstein_d_u(s: DDState):
    return -s.alpha * s.sinu + (s.beta + s.gamma) * s.cosu


def d2_romer_einstein_d_u2(s: DDState):
    return -s.alpha * s.cosu - (s.beta + s.gamma) * s.sinu


def dd_shapiro_delay(s: DDState):
    inner = (
        1.0
        - s.et * s.cosu
        - (s.sini / s.a1) * (s.alpha * (s.cosu - s.er) + s.beta * s.sinu)
    )
    return -2.0 * s.m2 * vm.log(inner)


def dd_delay(s: DDState):
    """Inverse timing formula (Damour & Deruelle 1986), Vela's expansion."""
    dre = romer_einstein_delay(s)
    drep = d_romer_einstein_d_u(s)
    drep2 = d2_romer_einstein_d_u2(s)
    nhat = s.n / (1.0 - s.et * s.cosu)
    dre_inv = dre * (
        1.0
        - nhat * drep
        + nhat * nhat * drep * drep
        + 0.5 * nhat * nhat * dre * drep2
        - 0.5 * s.et * s.sinu / (1.0 - s.et * s.cosu) * nhat * nhat * dre * drep
    )
    # Binary Doppler: d(Roemer+Einstein)/dt = dREp . nhat.
    return dre_inv + dd_shapiro_delay(s), drep * nhat


def binary_dd(frozen, corr: Correction, p, **static):
    delay, doppler = dd_delay(dd_state(frozen, corr, p, **static))
    return corr.add_delay(delay, doppler)
