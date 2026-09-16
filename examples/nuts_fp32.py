#!/usr/bin/env python
"""Gate T16: the float32 engine inside a float32 Discovery likelihood.

Runs the *same* pulsar twice through NUTS -- once with the float64 engine in a
float64 Discovery likelihood, once with the float32 perturbative engine in a
likelihood whose linear algebra is ``working=float32`` -- and reports whether
the binary-axis posteriors agree within Monte Carlo error.

    python nuts_fp32.py --timing-package tempo2
    python nuts_fp32.py --combined                 # MetaPulsar, one leg per PTA

What is being tested is the *delta channel*: the full residual cannot run in
float32 (a 500 s Roemer delay resolves to 3e-5 s), but every delta can,
provided no absolute quantity is ever formed. ``certify()`` already gates that
against the float64 parent on posterior-scale deltas; this asks the harder
question -- whether the posterior a sampler actually explores is the same one.

x64 stays enabled throughout. Discovery's float32 mode is a *working* dtype for
the expensive factorizations, not a global switch, and the timing deltas need
float64 available (nltiming refuses otherwise).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import warnings

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nuts_j1853 import RELEASE, build_likelihood, build_pulsar  # noqa: E402

#: The axes T16 names. Whatever else moves, these are the ones the Shapiro
#: delay's strongly-cancelling logarithm reaches, so they are where a float32
#: reference channel would show up first.
WATCHED = ("SINI", "M2", "H3", "STIGMA", "SHAPMAX", "KIN", "A1", "PB", "TASC", "T0")


class _Fp32Pulsar:
    """The same pulsar, handing out a float32 perturbative backend.

    A thin proxy rather than an engine keyword: the dtype is a property of this
    *experiment*, not of the pulsar contract, and ``TimingPulsar.timing_engine``
    deliberately refuses keywords it does not know (SPEC R-B6.4).
    """

    def __init__(self, pulsar, mode="binary+astrometry"):
        if not hasattr(pulsar, "engine"):
            raise TypeError(
                "the float32 gate is about the single-pulsar engine: this proxy "
                "needs a vela_jax.TimingPulsar, not a MetaPulsar composite. "
                "A composite's legs would each need their own perturbative "
                "engine, which is MetaPulsar's dispatch to make, not this "
                "script's. Drop --combined."
            )
        self._pulsar = pulsar
        self._mode = mode

    def __getattr__(self, name):
        return getattr(self._pulsar, name)

    def can_use_engines(self, engines="vela_jax", **kwargs):
        return self._pulsar.can_use_engines(engines, **kwargs)

    def pint_model(self):
        return self._pulsar.pint_model()

    def timing_engine(self, engines="vela_jax", *, nonlinear_params=None, **kwargs):
        import jax.numpy as jnp

        from vela_jax.backend import VelaJaxTimingEngine

        engine = self._pulsar.engine.perturbative(self._mode, dtype=jnp.float32)
        backend = VelaJaxTimingEngine(engine, name=self._pulsar.name)
        backend.nonlinear_params = nonlinear_params
        return backend


def run(pulsar, noisedict, *, working, args, tag):
    """One NUTS run at one working dtype; returns the posterior samples."""
    import discovery as ds
    import jax
    import jax.numpy as jnp
    import nltiming.sampling as nlts
    from nltiming import TimingSpec
    from numpyro.infer import init_to_value

    ds.utils.config(backend="jax", working=working)

    spec = TimingSpec(engines="vela_jax", name="timing")
    timing = spec.for_pulsar(pulsar)
    likelihood, fixed, _ = build_likelihood(pulsar, noisedict, timing)
    model = nlts.numpyro.decentered_model(likelihood, timing, fixed=fixed)
    mcmc = nlts.numpyro.nuts(
        model,
        timing,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=1,
        progress_bar=False,
        init_strategy=init_to_value(
            values=nlts.numpyro.decentered_init_values(timing, model.transport)
        ),
    )
    started = time.time()
    mcmc.run(jax.random.PRNGKey(args.seed), extra_fields=nlts.numpyro.NUTS_EXTRA_FIELDS)
    elapsed = time.time() - started
    divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    print(
        f"[{tag}] working={np.dtype(working).name} NUTS "
        f"{args.warmup}+{args.samples} in {elapsed:.1f}s, "
        f"divergences={divergences}"
    )
    assert divergences == 0 or working is jnp.float32, "float64 run diverged"

    import arviz as az

    idata = nlts.numpyro.posterior(mcmc, timing)
    ess = az.ess(idata)
    return {
        name: (
            np.asarray(idata.posterior[name], dtype=float).reshape(-1),
            float(np.asarray(ess[name])),
        )
        for name in idata.posterior.data_vars
    }, divergences


def compare(reference, other, *, tolerance):
    """Do the two posteriors agree to Monte Carlo error?

    The comparison is in units of the *combined* MC error of the two means,
    which is the only scale that means anything here: two independent chains of
    the same posterior differ by O(1) of it, and the engines are what is under
    test, not the sampler's variance.
    """
    print(
        f"{'parameter':22s} {'fp64 mean':>16s} {'fp32 mean':>16s} "
        f"{'sd':>11s} {'Δ/MC err':>9s}"
    )
    worst, worst_name = 0.0, ""
    for name in sorted(reference):
        if name not in other:
            continue
        a, ess_a = reference[name]
        b, ess_b = other[name]
        sd = 0.5 * (np.std(a) + np.std(b))
        mc = np.sqrt(np.var(a) / max(ess_a, 1.0) + np.var(b) / max(ess_b, 1.0))
        z = abs(np.mean(a) - np.mean(b)) / mc if mc > 0 else float("nan")
        flag = " <-- watched" if any(name.startswith(w) for w in WATCHED) else ""
        print(
            f"{name:22s} {np.mean(a): 16.9g} {np.mean(b): 16.9g} "
            f"{sd: 11.3g} {z: 9.2f}{flag}"
        )
        if any(name.startswith(w) for w in WATCHED) and z > worst:
            worst, worst_name = z, name
    verdict = "PASS" if worst <= tolerance else "FAIL"
    print(
        f"\n[T16] worst watched axis: {worst_name} at {worst:.2f} MC errors "
        f"(tolerance {tolerance}) -> {verdict}"
    )
    return worst <= tolerance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="directory containing epta_dr1_v2_2/ and nanograv_9y/",
    )
    parser.add_argument("--pulsar", default="J1853+1303")
    parser.add_argument("--timing-package", default="tempo2", choices=sorted(RELEASE))
    parser.add_argument(
        "--combined",
        action="store_true",
        help="refused: see _Fp32Pulsar; the gate is the single-pulsar engine",
    )
    parser.add_argument("--mode", default="binary+astrometry")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=4.0)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    try:
        from loguru import logger

        logger.remove()
    except ImportError:
        pass

    import nltiming.sampling as nlts

    nlts.numpyro.ensure_x64()

    import jax.numpy as jnp

    pulsar, dictionary = build_pulsar(args)
    tag = "combined" if args.combined else args.timing_package
    print(f"[{tag}] {pulsar.name}: {len(pulsar.toas)} TOAs")
    noisedict = json.load(
        open(pathlib.Path(args.data_root) / "noisedicts" / f"{dictionary}.json")
    )

    reference, _ = run(pulsar, noisedict, working=jnp.float64, args=args, tag=tag)
    other, divergences = run(
        _Fp32Pulsar(pulsar, args.mode),
        noisedict,
        working=jnp.float32,
        args=args,
        tag=tag + "/f32",
    )
    ok = compare(reference, other, tolerance=args.tolerance)
    print(f"[T16] float32 divergences: {divergences}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
