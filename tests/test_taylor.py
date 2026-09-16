import numpy as np
import pytest

from vela_jax.taylor import factorial_series, taylor_horner, taylor_horner_integral

pytestmark = pytest.mark.unit

pint_utils = pytest.importorskip("pint.utils")


@pytest.mark.parametrize("coeffs", [(3.0,), (3.0, -1.5), (3.0, -1.5, 0.25, 7.0)])
def test_taylor_horner_matches_pint(coeffs):
    x = np.linspace(-4.0, 4.0, 17)
    assert np.allclose(taylor_horner(x, coeffs), pint_utils.taylor_horner(x, coeffs))


def test_integral_is_the_antiderivative():
    coeffs = (3.0, -1.5, 0.25)
    x = np.linspace(-2.0, 2.0, 9)
    step = 1e-6
    numeric = (
        taylor_horner_integral(x + step, coeffs)
        - taylor_horner_integral(x - step, coeffs)
    ) / (2 * step)
    assert np.allclose(numeric, taylor_horner(x, coeffs), rtol=1e-8)


def test_factorial_series_is_the_tail():
    coeffs = (3.0, -1.5, 0.25, 7.0)
    x = np.linspace(-2.0, 2.0, 9)
    # The phase tail from F1 on is the full integral minus its leading term.
    tail = taylor_horner_integral(x, coeffs) - coeffs[0] * x
    assert np.allclose(factorial_series(x, coeffs[1:], 2), tail)
    # The frequency tail is the full series minus its constant.
    assert np.allclose(
        factorial_series(x, coeffs[1:], 1), taylor_horner(x, coeffs) - coeffs[0]
    )
