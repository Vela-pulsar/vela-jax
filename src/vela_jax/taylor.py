"""Taylor series in Vela's (and PINT's) factorial convention.

``taylor_horner(x, c) = sum_k c[k] x^k / k!`` and ``taylor_horner_integral`` is
its antiderivative through the origin. Coefficient tuples are static, so both
are plain Python loops: the length is known at trace time and the loop keeps
the routines usable with any :mod:`vela_jax.numerics` value type.
"""

from __future__ import annotations


def taylor_horner(x, coeffs):
    """``c0 + c1 x + c2 x^2/2! + ...``"""
    if not coeffs:
        return 0.0
    result = coeffs[-1]
    for k in range(len(coeffs) - 1, 0, -1):
        result = result * (x / k) + coeffs[k - 1]
    return result


def taylor_horner_integral(x, coeffs, const=0.0):
    """``const + c0 x + c1 x^2/2! + c2 x^3/3! + ...``"""
    if not coeffs:
        return const
    result = coeffs[-1] / len(coeffs)
    for k in range(len(coeffs) - 1, 0, -1):
        result = result * (x / k) + coeffs[k - 1] / k
    return result * x + const


def factorial_series(x, coeffs, first_power: int):
    """``sum_j coeffs[j] x^(first_power+j) / (first_power+j)!``

    The tail of a Taylor series whose leading coefficients live elsewhere.
    Vela keeps ``F0`` inside the series and splits it into a Double64
    high/low pair; here the high part is a build-time constant (SPEC §4.3), so the
    trace needs the series from ``F1`` on — that is, from power 2 for the
    phase and power 1 for the frequency.
    """
    total = 0.0
    term = 1.0
    power = 0
    factorial = 1.0
    for j, c in enumerate(coeffs):
        while power < first_power + j:
            power += 1
            factorial *= power
            term = term * x
        total = total + c * term / factorial
    return total
