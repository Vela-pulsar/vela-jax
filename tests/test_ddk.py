"""DDK Kopeikin annual-parallax frame (Vela ``binary_ddk.jl`` / ``test_dd.jl``)."""

from copy import deepcopy

import numpy as np
import pytest

from vela_jax.astrometry import ecliptic_to_equatorial, equatorial_to_ecliptic
from vela_jax.binary.dd import kopeikin_i0_j0
from vela_jax.constants import OBL
from vela_jax.numerics import dot3


def _vec(v):
    return np.array([float(x) for x in v])


@pytest.mark.unit
def test_equatorial_ecliptic_round_trip():
    """Vela ``DDK ecliptic annual-parallax frame``: the two rotations invert."""
    z = np.sqrt(1.0 - 0.3**2 - 0.4**2)
    l_ecl = (0.3, 0.4, z)
    l_icrs = ecliptic_to_equatorial(l_ecl, OBL)
    l_back = equatorial_to_ecliptic(l_icrs, OBL)
    assert np.allclose(_vec(l_back), _vec(l_ecl), atol=1e-15, rtol=0)


@pytest.mark.unit
def test_kopeikin_i0_j0_follow_the_sky_frame():
    """Building I0/J0 after the inverse rotation matches building them in ecliptic."""
    z = np.sqrt(1.0 - 0.3**2 - 0.4**2)
    l_ecl = (0.3, 0.4, z)
    l_icrs = ecliptic_to_equatorial(l_ecl, OBL)
    i0_ecl, j0_ecl = kopeikin_i0_j0(l_ecl)
    i0_from_icrs, j0_from_icrs = kopeikin_i0_j0(equatorial_to_ecliptic(l_icrs, OBL))
    assert np.allclose(_vec(i0_ecl), _vec(i0_from_icrs), atol=1e-15, rtol=0)
    assert np.allclose(_vec(j0_ecl), _vec(j0_from_icrs), atol=1e-15, rtol=0)


@pytest.mark.unit
def test_mixed_frame_annual_term_is_not_the_ecliptic_one():
    """The bug: ICRS I0 dotted with ICRS R is not ecliptic I0 dotted with ecliptic R.

    Vela's gate is ``abs(ΔI_ecl - ΔI_mixed) > 1 lt-s`` on the ``test_dd.jl``
    observer vector. A correct same-frame projector must not collapse to that.
    """
    z = np.sqrt(1.0 - 0.3**2 - 0.4**2)
    l_ecl = (0.3, 0.4, z)
    l_icrs = ecliptic_to_equatorial(l_ecl, OBL)
    # Vela ``test/runtests.jl`` ``ssb_obs_pos``.
    r_icrs = (18.0354099, 450.01472245, 195.05827732)
    r_ecl = equatorial_to_ecliptic(r_icrs, OBL)
    i0_ecl, _ = kopeikin_i0_j0(l_ecl)
    i0_mixed, _ = kopeikin_i0_j0(l_icrs)
    delta_i_ecl = float(dot3(r_ecl, i0_ecl))
    delta_i_mixed = float(dot3(r_icrs, i0_mixed))
    assert abs(delta_i_ecl - delta_i_mixed) > 1.0


def _freeze(model):
    model = deepcopy(model)
    for name in list(model.free_params):
        model[name].frozen = True
    return model


def _max_abs_mean_sub(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(np.max(np.abs(d - d.mean())))


def test_ddk_icrs_ecliptic_parity(examples):
    """PINT ``as_ECL()`` rotates coordinates and ``KOM``. After the frame fix,
    residuals agree across that conversion at the nanosecond level. A mixed
    ICRS/ecliptic annual term disagrees by ~2 μs on ``sim_ddk``.
    """
    from pint.residuals import Residuals

    from vela_jax import Engine
    from vela_jax.freeze import load_pint

    par, tim = examples / "sim_ddk.par", examples / "sim_ddk.tim"
    if not (par.exists() and tim.exists()):
        pytest.skip("fixture sim_ddk not available")

    model, toas = load_pint(par, tim)
    model_ecl = model.as_ECL()

    pint_icrs = Residuals(toas, model, subtract_mean=False).time_resids.to_value("s")
    pint_ecl = Residuals(toas, model_ecl, subtract_mean=False).time_resids.to_value("s")
    assert _max_abs_mean_sub(pint_icrs, pint_ecl) < 1e-12

    ours_icrs = Engine.from_pint(_freeze(model), toas).residuals()
    ours_ecl = Engine.from_pint(_freeze(model_ecl), toas).residuals()
    assert _max_abs_mean_sub(ours_icrs, ours_ecl) < 1e-9
    assert _max_abs_mean_sub(ours_ecl, pint_ecl) < 1e-9
    assert _max_abs_mean_sub(ours_icrs, pint_icrs) < 1e-9
