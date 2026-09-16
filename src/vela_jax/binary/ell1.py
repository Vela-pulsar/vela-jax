"""The ELL1 family for nearly circular orbits: ELL1, ELL1H, ELL1k.

Vela: ``src/model/binary/binary_ell1_base.jl`` plus ``binary_ell1.jl``,
``binary_ell1h.jl``, ``binary_ell1k.jl``. The three Roemer polynomials are
transcribed term by term from the Julia (Lange+ 2001, cubic order in
eccentricity); the coefficients are copied, never re-derived.

``ell1_t2`` selects tempo2's ``ELL1model.C`` truncation instead: it keeps the
Roemer delay only to first order in the Laplace-Lagrange parameters and drops
their harmonics from both derivatives. That is not an approximation *of* the
Vela polynomials -- it is the convention the par's fitted values came from
when the file was fitted by tempo2, so an engine with
``binary_conventions="tempo2"`` reproduces the tempo2 residual by using it.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .. import numerics as vm
from ..correction import Correction, corrected_time
from .orbit import mean_anomaly, mean_motion


class ELL1State(NamedTuple):
    """Vela ``ELL1State``. ``trigs[k]`` is ``(sin((k+1)Phi), cos((k+1)Phi))``."""

    trigs: tuple
    n: Any
    a1: Any
    eps1: Any
    eps2: Any
    m2: Any
    sini: Any


def shapiro_params(family: str, p):
    if family in ("ELL1", "ELL1k"):
        return p.M2, p.SINI
    if family == "ELL1H":
        stigma = p.STIGMA
        return p.H3 / stigma**3, 2.0 * stigma / (1.0 + stigma * stigma)
    raise ValueError(f"unknown ELL1 family {family!r}")


def ell1_state(frozen, corr: Correction, p, *, family, use_fbx):
    fb = frozen.binary
    dt = corrected_time(frozen, corr) - p.TASC
    dt_red = fb.dt_red - corr.delay - p.dTASC

    a1 = p.A1 + dt * p.A1DOT
    if family == "ELL1k":
        sin_wt, cos_wt = vm.sincos(p.OMDOT * dt)
        growth = 1.0 + p.LNEDOT * dt
        eps1 = growth * (p.EPS1 * cos_wt + p.EPS2 * sin_wt)
        eps2 = growth * (p.EPS2 * cos_wt - p.EPS1 * sin_wt)
    else:
        eps1 = p.EPS1 + dt * p.EPS1DOT
        eps2 = p.EPS2 + dt * p.EPS2DOT

    phi = mean_anomaly(fb, dt, dt_red, p, use_fbx)
    trigs = tuple(vm.sincos(k * phi) for k in (1, 2, 3, 4))
    m2, sini = shapiro_params(family, p)
    return ELL1State(trigs, mean_motion(dt, p, use_fbx), a1, eps1, eps2, m2, sini)


def romer_delay_t2(s: ELL1State):
    """tempo2 ``ELL1model.C``: ``x (sin Phi + (e2 sin 2Phi - e1 cos 2Phi)/2)``."""
    (s1, _), (s2, c2), _, _ = s.trigs
    return s.a1 * (s1 + 0.5 * (s.eps2 * s2 - s.eps1 * c2))


def d_romer_d_phi_t2(s: ELL1State):
    """tempo2 keeps no eccentricity harmonics in the first derivative."""
    return s.a1 * s.trigs[0][1]


def d2_romer_d_phi2_t2(s: ELL1State):
    return -s.a1 * s.trigs[0][0]


def romer_delay(s: ELL1State, family: str):
    (s1, c1), (s2, c2), (s3, c3), (s4, c4) = s.trigs
    e1, e2, a1 = s.eps1, s.eps2, s.a1
    value = a1 * (
        s1
        + 0.5 * (e2 * s2 - e1 * c2)
        - (1.0 / 8.0)
        * (
            (5 * e2**2 + 3 * e1**2) * s1
            - 2 * e2 * e1 * c1
            + (-3 * e2**2 + 3 * e1**2) * s3
            + 6 * e2 * e1 * c3
        )
        - (1.0 / 12.0)
        * (
            (5 * e2**3 + 3 * e1**2 * e2) * s2
            + (-6 * e1 * e2**2 - 4 * e1**3) * c2
            + (-4 * e2**3 + 12 * e1**2 * e2) * s4
            + (12 * e1 * e2**2 - 4 * e1**3) * c4
        )
    )
    if family == "ELL1k":
        # Vela `binary_ell1k.jl`: the exact treatment of periapsis advance
        # subtracts the constant 3/2 a1 eps1 term.
        value = value - 1.5 * a1 * e1
    return value


def d_romer_d_phi(s: ELL1State):
    (s1, c1), (s2, c2), (s3, c3), (s4, c4) = s.trigs
    e1, e2, a1 = s.eps1, s.eps2, s.a1
    return a1 * (
        c1
        + e1 * s2
        + e2 * c2
        - (1.0 / 8.0)
        * (
            (5 * e2**2 + 3 * e1**2) * c1
            + 2 * e1 * e2 * s1
            + (-9 * e2**2 + 9 * e1**2) * c3
            - 18 * e1 * e2 * s3
        )
        - (1.0 / 12.0)
        * (
            (10 * e2**3 + 6 * e1**2 * e2) * c2
            + (12 * e1 * e2**2 + 8 * e1**3) * s2
            + (-16 * e2**3 + 48 * e1**2 * e2) * c4
            + (-48 * e1 * e2**2 + 16 * e1**3) * s4
        )
    )


def d2_romer_d_phi2(s: ELL1State):
    (s1, c1), (s2, c2), (s3, c3), (s4, c4) = s.trigs
    e1, e2, a1 = s.eps1, s.eps2, s.a1
    return a1 * (
        -s1
        + 2 * e1 * c2
        - 2 * e2 * s2
        - (1.0 / 8.0)
        * (
            (-5 * e2**2 - 3 * e1**2) * s1
            + 2 * e1 * e2 * c1
            + (27 * e2**2 - 27 * e1**2) * s3
            - 54 * e1 * e2 * c3
        )
        - (1.0 / 12.0)
        * (
            (-20 * e2**3 - 12 * e1**2 * e2) * s2
            + (24 * e1 * e2**2 + 16 * e1**3) * c2
            + (64 * e2**3 - 192 * e1**2 * e2) * s4
            + (-192 * e1 * e2**2 + 64 * e1**3) * c4
        )
    )


def ell1_shapiro_delay(s: ELL1State, family: str):
    sin_phi = s.trigs[0][0]
    total = -2.0 * s.m2 * vm.log(1.0 - s.sini * sin_phi)
    if family != "ELL1H":
        return total
    # Vela `binary_ell1h.jl`: drop the harmonics fully covariant with Roemer.
    cos2_phi = s.trigs[1][1]
    cbar = vm.sqrt(1.0 - s.sini * s.sini)
    stigma = s.sini / (1.0 + cbar)
    a0 = -vm.log(1.0 + stigma * stigma)
    b1 = -2.0 * stigma
    a2 = stigma * stigma
    return total + 2.0 * s.m2 * (a0 + b1 * sin_phi + a2 * cos2_phi)


def ell1_delay(s: ELL1State, family: str, *, ell1_t2: bool = False):
    if ell1_t2:
        dr, drp, drp2 = (
            romer_delay_t2(s),
            d_romer_d_phi_t2(s),
            d2_romer_d_phi2_t2(s),
        )
        if family == "ELL1k":
            # ELL1k's exact periapsis advance is a constant in Phi, so it
            # survives the truncation untouched (Vela `binary_ell1k.jl`).
            dr = dr - 1.5 * s.a1 * s.eps1
    else:
        dr = romer_delay(s, family)
        drp = d_romer_d_phi(s)
        drp2 = d2_romer_d_phi2(s)
    nhat = s.n
    dr_inv = dr * (
        1.0 - nhat * drp + nhat * nhat * drp * drp + 0.5 * nhat * nhat * dr * drp2
    )
    return dr_inv + ell1_shapiro_delay(s, family), -drp * nhat


def binary_ell1(frozen, corr: Correction, p, *, family, use_fbx, ell1_t2=False):
    state = ell1_state(frozen, corr, p, family=family, use_fbx=use_fbx)
    delay, doppler = ell1_delay(state, family, ell1_t2=ell1_t2)
    return corr.add_delay(delay, doppler)
