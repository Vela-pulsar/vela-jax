"""Test fixtures.

The par/tim pairs are Vela.jl's own examples, so the two projects gate on the
same data. Point ``VELA_JAX_FIXTURES`` at a copy of ``pyvela/examples`` (the
default guesses a sibling Vela.jl checkout); tests skip cleanly without it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:  # x64 must be set before any JAX array exists
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover
    pass


def pytest_addoption(parser):
    parser.addoption(
        "--min-passed",
        type=int,
        default=0,
        help="fail the run unless at least this many tests passed. Every "
        "optional dependency in this suite skips cleanly when absent, which "
        "means a CI job whose environment quietly lost one reports success "
        "while checking nothing. A floor turns that into a red run.",
    )
    parser.addoption(
        "--certify",
        choices=("default", "full"),
        default="default",
        help="how much of the perturbative delta suite to run: 'default' is "
        "+-3 sigma per live axis plus 2 joint draws, 'full' adds +-1 sigma "
        "and 8 draws (breadth, not behaviour)",
    )


_DEFAULTS = (
    Path(__file__).resolve().parents[2] / "Vela.jl" / "pyvela" / "examples",
    Path(__file__).resolve().parents[1] / "tests" / "fixtures",
)


def fixture_dir() -> Path | None:
    env = os.environ.get("VELA_JAX_FIXTURES")
    candidates = (Path(env),) if env else _DEFAULTS
    for path in candidates:
        if path.is_dir() and any(path.glob("*.par")):
            return path
    return None


@pytest.fixture(scope="session")
def examples() -> Path:
    path = fixture_dir()
    if path is None:
        pytest.skip("no par/tim fixtures; set VELA_JAX_FIXTURES")
    return path


@pytest.fixture(scope="session")
def tempo2_available() -> bool:
    """``libstempo.sandbox`` importable, and a tempo2 binary on PATH.

    Missing ``siteVel`` / clock-correction fields still refuse at
    ``read_tempo2._require`` on the sandbox instance, with a named error.
    """
    import shutil

    try:
        from libstempo.sandbox import tempopulsar  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tempo2") is not None


@pytest.fixture(autouse=True)
def _skip_without_tempo2(request, tempo2_available):
    """Skip a tempo2 test without tempo2 -- unless it never needed it.

    A couple of tests live in the tempo2 module because that is where the code
    they check lives, while being pure string functions (the ``FDJUMPn`` and
    ``CLOCK`` respellings). ``no_tempo2`` opts them back in, so they run in the
    unit tier on a machine with no tempo2 at all.
    """
    if request.node.get_closest_marker("no_tempo2"):
        return
    if request.node.get_closest_marker("tempo2") and not tempo2_available:
        pytest.skip("needs libstempo.sandbox and tempo2")


@pytest.fixture(scope="session")
def tempo2_engine_factory(examples):
    """Cached engines with tempo2 as the timing package."""
    from vela_jax import Engine

    cache: dict[tuple[str, str], Engine] = {}

    def build(name: str, conventions: str = "pint"):
        key = (name, conventions)
        if key not in cache:
            par, tim = examples / f"{name}.par", examples / f"{name}.tim"
            if not (par.exists() and tim.exists()):
                pytest.skip(f"fixture {name} not available")
            cache[key] = Engine.from_tempo2(par, tim, binary_conventions=conventions)
        return cache[key]

    return build


@pytest.fixture(scope="session")
def engine_factory(examples):
    """Cached engine builder; PINT ingest dominates the test runtime."""
    from vela_jax import Engine

    cache: dict[str, Engine] = {}

    def build(name: str):
        if name not in cache:
            par, tim = examples / f"{name}.par", examples / f"{name}.tim"
            if not (par.exists() and tim.exists()):
                pytest.skip(f"fixture {name} not available")
            cache[name] = Engine.from_files(par, tim)
        return cache[name]

    return build


#: ``(fixture, what it exercises, PINT agreement budget in seconds)``.
#:
#: Vela.jl is the oracle (see ``test_oracle_pyvela.py``, which gates at 1 ns);
#: PINT parity is a coarse sanity check on top. The two genuinely disagree on
#: the orthometric Shapiro parametrisation -- Vela subtracts the a0/b1/a2
#: harmonics analytically where PINT truncates a harmonic series at NHARMS --
#: so the ELL1H and DDH fixtures carry a wider budget on purpose.
ENGINES = [
    ("NGC6440E", "isolated, equatorial", 1e-7),
    ("sim_dd", "DD", 1e-7),
    ("sim_ddk", "DDK", 1e-7),
    ("J1802-2124.sim", "ELL1", 1e-7),
    ("J1227-6208.sim", "ELL1H", 1e-6),
    ("sim_ell1k", "ELL1k", 1e-7),
    ("J0453+1559.sim", "DDH, infinite-frequency TZR", 1e-6),
    ("J2302+4442.sim", "DDS, ecliptic", 1e-7),
    ("sim_dmx", "DMX", 1e-7),
    ("sim_fd", "FD", 1e-7),
    ("sim_jump", "non-exclusive JUMPs", 1e-7),
    ("sim_jump_ex", "exclusive JUMPs", 1e-7),
    ("sim_sw", "solar wind, planet Shapiro", 1e-7),
    ("pure_rotator", "no astrometry", 1e-7),
]


#: The fixtures a default run uses: one per binary family, plus one of each
#: structural case (no astrometry, DMX, FD, both JUMP kinds, solar wind),
#: choosing the cheapest fixture that exercises the thing. Everything else is
#: a *breadth* check rather than a *behaviour* check, so it is marked ``slow``
#: and runs at checkpoints.
#:
#: Two fixtures are deliberately not core because they are slow for a reason
#: outside this package: ``pure_rotator`` and ``J0453+1559.sim`` name an
#: ephemeris astropy has to fetch, and pay ~9 s of ``download_file`` on a cold
#: cache. ``J1208-5936.sim`` covers DDH in their place at 500 TOAs.
CORE_FIXTURES = frozenset(
    {
        "NGC6440E",  # isolated, equatorial, 62 TOAs
        "sim_dd",  # DD
        "J1208-5936.sim",  # DDH
        "J2302+4442.sim",  # DDS, ecliptic
        "sim_ddk",  # DDK (Kopeikin)
        "J1802-2124.sim",  # ELL1
        "J1227-6208.sim",  # ELL1H
        "sim_ell1k",  # ELL1k
        "sim_dmx",  # DMX
        "sim_jump",  # non-exclusive JUMPs
    }
)


def engine_params(names=None):
    """Parametrise over fixtures, marking the non-core ones ``slow``.

    One test function, two tiers: ``-m "not slow"`` runs the behaviour set,
    a full run sweeps everything. Keeping it as marks rather than two lists
    means a gate cannot drift between the tiers.
    """
    import pytest as _pytest

    chosen = [n for n, _, _ in ENGINES] if names is None else list(names)
    return [
        _pytest.param(name, marks=() if name in CORE_FIXTURES else _pytest.mark.slow)
        for name in chosen
    ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Enforce ``--min-passed``, after the report so the count is visible."""
    floor = config.getoption("--min-passed")
    if not floor:
        return
    passed = len(terminalreporter.stats.get("passed", []))
    if passed >= floor:
        return
    terminalreporter.write_line(
        f"ERROR: {passed} tests passed, expected at least {floor}. "
        "Something the suite depends on is missing, so most of it skipped.",
        red=True,
    )
    terminalreporter.section("min-passed")
    session = terminalreporter._session
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
