#!/usr/bin/env python
"""Sample J1853+1303 timing parameters with NUTS, through vela-jax.

Three configurations, one script. ``--data-root`` is the directory that
contains the public EPTA DR1 v2.2 and NANOGrav 9-year trees
(``epta_dr1_v2_2/`` and ``nanograv_9y/``):

    # one release, tempo2 reads it, native vela-jax pulsar
    python nuts_j1853.py --data-root DIR --timing-package tempo2

    # one release, PINT reads it
    python nuts_j1853.py --data-root DIR --timing-package pint

    # both releases combined by MetaPulsar, one vela-jax leg each
    python nuts_j1853.py --data-root DIR --combined

The likelihood is Discovery's: per-backend EFAC/EQUAD (and ECORR where the
noise dictionary has it) plus a power-law red-noise GP, all fixed at the
dictionary values, with the timing block sampled through nltiming's dynamic
decentering. Only the timing parameters move, which is the point -- the run
exercises the delay engine, not the noise model.

Needs ``discovery``, ``numpyro`` and ``nltiming``; for a tempo2 timing package,
``libstempo`` and tempo2.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import warnings

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

RELEASE = {"tempo2": "epta_dr1_v2_2", "pint": "nanograv_9y"}


def build_pulsar(args):
    """A vela-jax ``TimingPulsar``, or a MetaPulsar with one leg per release."""
    from vela_jax import TimingPulsar

    root = pathlib.Path(args.data_root)
    if not args.combined:
        base = root / RELEASE[args.timing_package]
        return (
            TimingPulsar.from_files(
                base / "par" / f"{args.pulsar}.par",
                base / "tim" / f"{args.pulsar}.tim",
                timing_package=args.timing_package,
            ),
            f"metapulsar-{args.timing_package}",
        )

    from metapulsar import create_metapulsar

    inputs = {
        release: [
            {
                "par": root / release / "par" / f"{args.pulsar}.par",
                "tim": root / release / "tim" / f"{args.pulsar}.tim",
                "timing_package": package,
            }
        ]
        for package, release in RELEASE.items()
    }
    return (
        create_metapulsar(
            inputs,
            combination_strategy="per_pta",
            use_pulse_numbers="reuse",
            # Each leg is read once, by vela-jax, and the record it emits *is*
            # the engine's -J. Without this the legs would be read by PINT and
            # tempo2 and the engine chosen afterwards, which MetaPulsar now
            # refuses by name.
            engines="vela_jax",
        ),
        "metapulsar",
    )


def build_likelihood(pulsar, noisedict, timing):
    import discovery as ds

    noise = {k: v for k, v in noisedict.items() if k.startswith(pulsar.name + "_")}
    backends = sorted({b for b in pulsar.backend_flags if b})
    has_ecorr = all(f"{pulsar.name}_{b}_log10_ecorr" in noise for b in backends)

    signals = [pulsar.residuals, ds.makenoise_measurement(pulsar, noise)]
    if has_ecorr:
        signals.append(ds.makegp_ecorr(pulsar, noise))
    signals += [
        ds.makegp_fourier(pulsar, ds.powerlaw, 30, name="red_noise"),
        *timing.discovery_signals(),
    ]
    likelihood = ds.PulsarLikelihood(signals)
    free = set(getattr(likelihood.logL, "params", []))
    return likelihood, {k: v for k, v in noise.items() if k in free}, backends


def report(mcmc, timing, tag):
    import arviz as az
    import nltiming.sampling as nlts

    extra = mcmc.get_extra_fields()
    print(
        f"[{tag}] divergences={int(np.sum(np.asarray(extra['diverging'])))}  "
        f"mean leapfrog={np.mean(np.asarray(extra['num_steps'])):.1f}"
    )

    idata = nlts.numpyro.posterior(mcmc, timing)
    posterior, ess = idata.posterior, az.ess(idata)
    references = timing.engine.reference_theta_exact()
    header = f"{'parameter':24s} {'par value':>18s} {'posterior mean':>18s}"
    print(f"[{tag}] {header} {'sd':>10s} {'z':>7s} {'ESS':>7s}")
    for name in sorted(posterior.data_vars):
        draws = np.asarray(posterior[name], dtype=float).reshape(-1)
        reference = float(references.get(name, "nan"))
        sd = float(np.std(draws))
        z = (np.mean(draws) - reference) / sd if sd > 0 else float("nan")
        print(
            f"[{tag}] {name:24s} {reference: 18.10g} {np.mean(draws): 18.10g} "
            f"{sd: 10.3g} {z: 7.2f} {float(np.asarray(ess[name])): 7.0f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="directory containing epta_dr1_v2_2/ and nanograv_9y/",
    )
    parser.add_argument("--pulsar", default="J1853+1303")
    parser.add_argument("--timing-package", default="tempo2", choices=sorted(RELEASE))
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    try:
        from loguru import logger

        logger.remove()
    except ImportError:
        pass

    import nltiming.sampling as nlts

    nlts.numpyro.ensure_x64()  # before any JAX array exists

    import jax
    from nltiming import TimingSpec
    from numpyro.infer import init_to_value

    started = time.time()
    pulsar, dictionary = build_pulsar(args)
    tag = "combined" if args.combined else args.timing_package
    print(
        f"[{tag}] {pulsar.name}: {len(pulsar.toas)} TOAs, {len(pulsar.fitpars)} fitpars"
    )

    timing = TimingSpec(engines="vela_jax", name="timing").for_pulsar(pulsar)
    print(f"[{tag}] engine={type(timing.engine).__name__} sampled={timing.sampled}")

    noisedict = json.load(
        open(pathlib.Path(args.data_root) / "noisedicts" / f"{dictionary}.json")
    )
    likelihood, fixed, backends = build_likelihood(pulsar, noisedict, timing)
    print(f"[{tag}] {len(backends)} backends, {len(fixed)} noise parameters held fixed")

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
    sampling_started = time.time()
    mcmc.run(jax.random.PRNGKey(args.seed), extra_fields=nlts.numpyro.NUTS_EXTRA_FIELDS)
    print(
        f"[{tag}] NUTS {args.warmup}+{args.samples} in "
        f"{time.time() - sampling_started:.1f}s"
    )
    report(mcmc, timing, tag)
    print(f"[{tag}] total {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
