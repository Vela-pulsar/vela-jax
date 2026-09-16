"""Certification of a perturbative engine against its fp64 parent (SPEC §11.9).

Two questions, deliberately separated:

* Is the *delta formulation* right? Run the perturbative engine in float64 and
  demand agreement with the full engine to ~1e-12 s. This is independent of
  float32 and is the test that says the identities in
  :mod:`vela_jax.perturbative.dual` are correct.
* Is float32 good enough? Same comparison at ``dtype=float32``, against a
  relative tolerance, on posterior-scale deltas only.

The fp64 engine is always the oracle here; the perturbative engine is never
validated against PINT directly.
"""

from __future__ import annotations

import numpy as np

from . import CertifyReport


def posterior_scale_deltas(
    engine, *, n_random=32, seed=0, scales=(-3.0, -1.0, 1.0, 3.0)
) -> np.ndarray:
    """``scales`` sigma on each live axis, plus random joint draws.

    Sigma comes from the par file's frequentist uncertainties. An axis with no
    uncertainty gets a small fraction of its own value, which is crude but
    keeps the axis in the suite instead of silently dropping it.
    """
    parent = engine.parent
    n = len(parent.param_names)
    sigma = np.zeros(n)
    for i, name in enumerate(parent.param_names):
        if name not in engine.live_nonlinear:
            continue
        param = parent.pint_model[name]
        value = float(param.uncertainty_value or 0.0)
        if value == 0.0:
            value = abs(float(param.value or 0.0)) * 1e-8 or 1e-10
        sigma[i] = value

    cases = []
    for i in engine._nl_index:
        for scale in scales:
            delta = np.zeros(n)
            delta[i] = scale * sigma[i]
            cases.append(delta)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        cases.append(rng.normal(size=n) * sigma)
    return np.array(cases) if cases else np.zeros((0, n))


def certify(
    engine,
    *,
    deltas=None,
    rtol=1e-5,
    atol=1e-12,
    n_random=32,
    seed=0,
    residual_cap=None,
    scales=(-3.0, -1.0, 1.0, 3.0),
):
    """Compare ``engine`` against ``engine.parent`` and report the worst case.

    ``residual_cap`` (seconds) rescales any case whose exact residual change
    exceeds it. The assembly (SPEC R11.6) is first order in ``Delta D``; what
    it drops is second order in ``Delta D`` -- the spin-frequency drift
    across the delay change (the ``Delta doppler * Delta D`` term went with
    the §8 divisor change), second order in the residual
    change with a coefficient of ~4e-4 per second. On a par whose quoted
    uncertainties are large -- ``sim_ell1k``'s 3 sigma moves the residual by
    14 ms -- that truncation dominates, and no amount of delta-formulation
    care would remove it. Capping at the ~50 microsecond scale a posterior
    actually visits separates "is the difference algebra right" from "is a
    first-order assembly valid at this amplitude", which
    :attr:`CertifyReport.quadratic_coefficient` answers on its own.
    """
    if deltas is None:
        deltas = posterior_scale_deltas(
            engine, n_random=n_random, seed=seed, scales=scales
        )
    deltas = np.atleast_2d(np.asarray(deltas, dtype=float))

    max_abs = 0.0
    max_scale = 0.0
    for delta in deltas:
        exact = engine.parent.residual_delta(delta)
        if residual_cap is not None:
            size = float(np.max(np.abs(exact)))
            if size > residual_cap:
                delta = delta * (residual_cap / size)
                exact = engine.parent.residual_delta(delta)
        approx = engine.residual_delta(delta)
        max_abs = max(max_abs, float(np.max(np.abs(approx - exact))))
        max_scale = max(max_scale, float(np.max(np.abs(exact))))
    # Relative to the largest delta in the suite, not case by case: a case
    # whose delta happens to be near zero would otherwise dominate the ratio
    # while carrying no information.
    max_rel = max_abs / max_scale if max_scale else 0.0

    jac_rel = None
    if len(engine._nl_index):
        exact = engine.parent.residual_jacobian()[:, engine._nl_index]
        approx = engine.residual_jacobian()[:, engine._nl_index]
        norm = np.maximum(np.max(np.abs(exact), axis=0), np.finfo(float).tiny)
        jac_rel = float(np.max(np.abs(approx - exact) / norm))

    return CertifyReport(
        n_cases=len(deltas),
        max_abs=max_abs,
        max_rel=max_rel,
        max_scale=max_scale,
        rtol=rtol,
        atol=atol,
        jacobian_rel=jac_rel,
    )
