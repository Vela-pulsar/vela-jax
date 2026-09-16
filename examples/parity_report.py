#!/usr/bin/env python
"""Regenerate the measured-parity tables in ``docs/PARITY.md``.

Runs, for every Vela.jl example fixture this engine supports:

* the fp64 residual against Vela.jl itself, via pyvela, built from the *same*
  PINT model object so that only physics is compared;
* posterior-scale residual *deltas* against the same oracle;
* the perturbative engine, in float64 and float32, against the fp64 parent;
* the PINT–tempo2 freeze comparison (gate H7), with ``ECL`` set to
  tempo2's IERS2003 so PINT uses the same obliquity, when tempo2 is available.

    python examples/parity_report.py --fixtures /path/to/pyvela/examples

Needs the ``oracle`` extra (pyvela + a Julia runtime); the H7 table needs the
``tempo2`` extra and is skipped without it. Vela ``sim_jump`` and ``sim_sw``
are omitted from H7 (clock coverage / mixed-engine solar-wind profile);
replacements are under ``tests/data/``.

Vela's rows are PINT's table rows, and so are this engine's: nothing in this
package reorders TOAs, so the arrays are compared as they come.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import tempfile
import warnings

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

#: Every fixture in ``pyvela/examples`` this engine accepts, with what it
#: exercises. The rest are refused by name (BT, WaveX, CMWaveX, ...).
FIXTURES = [
    ("NGC6440E", "isolated, equatorial"),
    ("sim1", "isolated + parallax"),
    ("sim2", "isolated, no DM"),
    ("pure_rotator", "no astrometry at all"),
    ("J1856-3754.sim", "isolated, ecliptic"),
    ("sim_dmx", "DMX"),
    ("sim_fd", "FD"),
    ("sim_jump", "non-exclusive JUMPs"),
    ("sim_jump_ex", "exclusive JUMPs"),
    ("sim_sw", "solar wind + planet Shapiro, ecliptic"),
    ("sim_dd", "DD"),
    ("J0955-6150.sim", "DD with OMDOT"),
    ("sim_ddk", "DDK (Kopeikin)"),
    ("J0453+1559.sim", "DDH, infinite-frequency TZR"),
    ("J1208-5936.sim", "DDH"),
    ("J2302+4442.sim", "DDS, ecliptic"),
    ("J1802-2124.sim", "ELL1"),
    ("J1227-6208.sim", "ELL1H"),
    ("sim_ell1k", "ELL1k"),
]

#: Fixtures with a binary, for the perturbative gate.
PERTURBATIVE = [name for name, _ in FIXTURES[10:]]


def build_pair(examples, name):
    """An engine and a Vela session over one PINT model, so only physics differs."""
    from pyvela import SPNTA

    from vela_jax import Engine
    from vela_jax.freeze import strip_noise_lines

    par, tim = examples / f"{name}.par", examples / f"{name}.tim"
    stripped = pathlib.Path(tempfile.mkdtemp()) / par.name
    stripped.write_text(strip_noise_lines(par.read_text()))
    spnta = SPNTA(str(stripped), str(tim), center_epochs=False, check=False)
    return Engine.from_pint(spnta.model_pint_modified, spnta.toas_pint), spnta


def oracle_row(examples, name, description, *, draws=4, seed=0):
    engine, spnta = build_pair(examples, name)
    theta = np.asarray(spnta.default_params, dtype=float)
    scale = np.asarray(spnta.scale_factors, dtype=float)
    vela_names = [str(n) for n in spnta.param_names]
    permutation = np.array([vela_names.index(n) for n in engine.param_names])
    reference = np.asarray(spnta.time_residuals(theta))

    absolute = np.max(np.abs(engine.residuals() - reference))

    sigma = np.array(
        [
            float(engine.pint_model[n].uncertainty_value or 0.0)
            for n in engine.param_names
        ]
    )
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(draws):
        delta = rng.normal(size=len(sigma)) * sigma
        perturbed = theta.copy()
        perturbed[permutation] += delta * scale[permutation]
        theirs = np.asarray(spnta.time_residuals(perturbed)) - reference
        worst = max(worst, float(np.max(np.abs(engine.residual_delta(delta) - theirs))))
    return {
        "fixture": name,
        "what": description,
        "stages": " ".join(s.split(".")[-1] for s in engine.stages),
        "n": engine.toa_count,
        "rms_ns": float(np.std(engine.residuals())) * 1e9,
        "abs_ps": float(absolute) * 1e12,
        "delta_ps": worst * 1e12,
    }


#: The residual amplitude the float64 gate caps at (SPEC R11.6): above it the
#: comparison measures the assembly's dropped second-order term, not the
#: difference algebra. The float32 gate is relative and runs uncapped.
POSTERIOR_SCALE_S = 2e-5


def perturbative_row(examples, name):
    import jax.numpy as jnp

    from vela_jax import Engine

    engine = Engine.from_files(examples / f"{name}.par", examples / f"{name}.tim")
    rows = {"fixture": name}
    rows["f64"] = engine.perturbative("binary+astrometry", dtype=jnp.float64).certify(
        n_random=8, residual_cap=POSTERIOR_SCALE_S
    )
    rows["f32"] = engine.perturbative("binary+astrometry", dtype=jnp.float32).certify(
        n_random=8
    )
    rows["quadratic"] = engine.perturbative(
        "binary+astrometry", dtype=jnp.float64
    ).certify(n_random=8)
    return rows


#: tempo2's own obliquity, as an ``ECL`` name. ``readParfile.C`` has no ``ECL``
#: keyword: tempo2 rotates every ephemeris vector with
#: ``ECLIPTIC_OBLIQUITY_VAL`` = 84381.4059", which is PINT's ``IERS2003``.
#: Setting that name on the PINT copy is how both packages use one frame.
TEMPO2_ECL = "IERS2003"


def _frame_pinned_par(examples, name, directory):
    """The par with ``ECL`` set to tempo2's default so PINT uses it too.

    Equatorial pars are a no-op. Ecliptic Vela fixtures say ``IERS2010``;
    without this pin the table would report the 0.1 mas obliquity
    disagreement (~120 ns), not the clock / ephemeris floor.
    """
    source = (examples / f"{name}.par").read_text()
    keep = [line for line in source.splitlines() if line.split()[:1] != ["ECL"]]
    if "ECL" in source:
        keep.append(f"ECL {TEMPO2_ECL}")
    par = pathlib.Path(directory) / f"{name}.par"
    par.write_text("\n".join(keep) + "\n")
    return par


def two_package_row(examples, name, directory):
    """Gate H7: RMS between the PINT-read and tempo2-read residuals."""
    from vela_jax import Engine

    par = _frame_pinned_par(examples, name, directory)
    tim = examples / f"{name}.tim"
    pint_engine = Engine.from_files(par, tim)
    tempo2_engine = Engine.from_tempo2(par, tim)
    if pint_engine.toa_count != tempo2_engine.toa_count:
        return None
    difference = tempo2_engine.residuals() - pint_engine.residuals()
    difference = difference - difference.mean()
    return {
        "fixture": name,
        "n": pint_engine.toa_count,
        "rms_ns": float(np.sqrt(np.mean(difference**2))) * 1e9,
        "max_ns": float(np.max(np.abs(difference))) * 1e9,
        "pulse_numbers": tempo2_engine.pulse_number_source,
    }


def design_row(examples, name):
    """Gate M1: -Mmat against a five-point difference quotient, worst column."""
    from vela_jax import Engine

    engine = Engine.from_files(examples / f"{name}.par", examples / f"{name}.tim")
    design = engine.design_matrix()
    worst, worst_name = 0.0, ""
    for column, param in enumerate(engine.param_names):
        scale = float(np.max(np.abs(design[:, column])))
        if scale == 0.0:
            continue
        errors = []
        delta = np.zeros(len(engine.param_names))
        for target in (1e-6, 1e-7, 1e-8):
            step = target / scale

            def moved(scaling, step=step, column=column):
                delta[column] = scaling * step
                out = engine.residual_delta(delta)
                delta[column] = 0.0
                return out

            numeric = -(8 * (moved(1) - moved(-1)) - (moved(2) - moved(-2))) / (
                12 * step
            )
            error = float(np.max(np.abs(numeric - design[:, column])) / scale)
            if np.isfinite(error):
                errors.append(error)
        if errors and min(errors) > worst:
            worst, worst_name = min(errors), param
    return {"fixture": name, "worst": worst, "param": worst_name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        default=os.environ.get("VELA_JAX_FIXTURES", "../Vela.jl/pyvela/examples"),
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    try:
        from loguru import logger

        logger.remove()
    except ImportError:
        pass

    examples = pathlib.Path(args.fixtures)
    print("| fixture | exercises | TOAs | RMS | ‖r − r_Vela‖∞ | ‖Δr − Δr_Vela‖∞ |")
    print("|---|---|---:|---:|---:|---:|")
    for name, description in FIXTURES:
        if not (examples / f"{name}.par").exists():
            continue
        row = oracle_row(examples, name, description)
        print(
            f"| `{row['fixture']}` | {row['what']} | {row['n']} | "
            f"{row['rms_ns']:.0f} ns | {row['abs_ps']:.3g} ps | "
            f"{row['delta_ps']:.3g} ps |"
        )

    print()
    print(
        "| fixture | live axes | float64 ‖Δ‖∞ | float64 Jacobian | "
        "float32 ‖Δ‖∞ | float32 Jacobian | float32 gate |"
    )
    print("|---|---:|---:|---:|---:|---:|---|")
    for name in PERTURBATIVE:
        if not (examples / f"{name}.par").exists():
            continue
        rows = perturbative_row(examples, name)
        f64, f32 = rows["f64"], rows["f32"]
        print(
            f"| `{name}` | {f64.n_cases} cases | {f64.max_abs:.2g} s | "
            f"{f64.jacobian_rel:.2g} | {f32.max_abs:.2g} s | "
            f"{f32.jacobian_rel:.2g} | {'pass' if f32.passed else '**FAIL**'} |"
        )

    print()
    print("| fixture | worst column | ‖(−M) − ΔQ‖∞ / max\\|M\\| |")
    print("|---|---|---:|")
    for name, _ in FIXTURES:
        if not (examples / f"{name}.par").exists():
            continue
        row = design_row(examples, name)
        print(f"| `{row['fixture']}` | `{row['param']}` | {row['worst']:.2g} |")

    try:
        from libstempo.sandbox import tempopulsar  # noqa: F401
    except ImportError:
        return
    print()
    print("| fixture | TOAs | pulse numbers | RMS(r_tempo2 − r_pint) | max |")
    print("|---|---:|---|---:|---:|")
    directory = tempfile.mkdtemp(prefix="vela_jax_frame_")
    h7_skip = {"sim_jump", "sim_sw"}
    for name, _ in FIXTURES:
        if name in h7_skip:
            continue
        if not (examples / f"{name}.par").exists():
            continue
        try:
            row = two_package_row(examples, name, directory)
        except Exception as error:  # pragma: no cover - reporting
            print(f"| `{name}` | | | *{type(error).__name__}* | |")
            continue
        if row is None:
            print(f"| `{name}` | | | *TOA counts differ* | |")
            continue
        print(
            f"| `{row['fixture']}` | {row['n']} | {row['pulse_numbers']} | "
            f"{row['rms_ns']:.2f} ns | {row['max_ns']:.2f} ns |"
        )

    extra = pathlib.Path(__file__).resolve().parents[1] / "tests" / "data"
    for name in ("sim_jump_clk", "sim_sw_aligned"):
        par, tim = extra / name / f"{name}.par", extra / name / f"{name}.tim"
        if not (par.is_file() and tim.is_file()):
            continue
        from vela_jax import Engine

        pint_engine = Engine.from_files(par, tim, timing_package="pint")
        tempo2_engine = Engine.from_files(par, tim, timing_package="tempo2")
        difference = tempo2_engine.residuals() - pint_engine.residuals()
        difference = difference - difference.mean()
        rms_ns = float(np.sqrt(np.mean(difference**2))) * 1e9
        max_ns = float(np.max(np.abs(difference))) * 1e9
        print(
            f"| `{name}` | {pint_engine.toa_count} | "
            f"{tempo2_engine.pulse_number_source} | "
            f"{rms_ns:.2f} ns | {max_ns:.2f} ns |"
        )

    matched = (
        pathlib.Path(__file__).resolve().parents[1]
        / "tests"
        / "data"
        / "J1909-3744-sim"
    )
    par, tim = matched / "J1909-3744.par", matched / "J1909-3744.tim"
    if par.is_file() and tim.is_file():
        import astropy.units as u
        from pint.models import get_model
        from pint.residuals import Residuals
        from pint.toa import get_TOAs

        model = get_model(str(par))
        toas = get_TOAs(str(tim), model=model, include_pn=True)
        r_pint = np.asarray(Residuals(toas, model).time_resids.to_value(u.s))
        psr = tempopulsar(parfile=str(par), timfile=str(tim), dofit=False)
        r_t2 = np.asarray(
            psr.residuals(updatebats=True, formresiduals=True, removemean=True),
            dtype=float,
        )
        r_pint = r_pint - r_pint.mean()
        r_t2 = r_t2 - r_t2.mean()
        rms_ns = float(np.sqrt(np.mean((r_pint - r_t2) ** 2))) * 1e9
        max_ns = float(np.max(np.abs(r_pint - r_t2))) * 1e9
        print()
        print("| fixture | TOAs | RMS(PINT − libstempo) | max |")
        print("|---|---:|---:|---:|")
        print(
            f"| `J1909-3744-sim` (packages, barycentric) | "
            f"{len(r_pint)} | {rms_ns:.3f} ns | {max_ns:.3f} ns |"
        )


if __name__ == "__main__":
    main()
