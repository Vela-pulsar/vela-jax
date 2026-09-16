"""The tempo2-read evidence the design was owed (SPEC A.6; gates H8 and S4).

H7 -- the PINT–tempo2 clock floor, with ``ECL`` set so PINT uses tempo2's
obliquity -- lives with the rest of the tempo2 timing-package checks in
``test_read_tempo2.py``. The two gates here need fixtures that do not exist
in Vela.jl's example set, so each builds its own from one that does, in the
open, with the modification and the reason written down.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tempo2


def _lines(path):
    return [line for line in path.read_text().splitlines() if line.strip()]


# --- H8: TRACK -2 is required, not permitted (A.6.2) -----------------------


@pytest.fixture(scope="module")
def phase_connected_fixture(examples, tmp_path_factory):
    """``sim_dd`` with a phase connection tempo2 alone can know about.

    ``sim_dd``'s tim already carries ``-pn`` flags; the par gets ``TRACK -2``,
    which is what makes tempo2 read them (``formResiduals.C:2263``). Two tims
    are written, identical except that one shifts the last quarter of the
    ``-pn`` values by a whole turn.

    Nothing in the timing model distinguishes the two files -- same TOAs, same
    par, same delays -- so a timing package that silently dropped the
    connection would produce *the same residuals twice*. Honouring it puts a
    step of one spin period on a quarter of the data: A.6.2's requirement
    that a dropped connection be at least half a period wrong rather than
    invisible.

    Working in the difference of two connected files, rather than against
    PINT's model numbering, is deliberate: tempo2 references the flags to its
    own count at ``bbat[0]`` and the reader then re-references the result onto
    PINT's fiducial by the median offset (``resolve_pulse_numbers``); the
    difference of two runs cancels both conventions and leaves exactly the
    injected turn. The shift is applied to a minority of rows for the same
    reason -- a majority shift would be absorbed into that median. Engine row
    ``i`` is file row ``i``, so the injected mask is just an index range.
    """
    directory = tmp_path_factory.mktemp("track2")
    par_in, tim_in = examples / "sim_dd.par", examples / "sim_dd.tim"

    body, header = [], []
    for line in _lines(tim_in):
        (header if line.split()[0] in ("FORMAT", "C", "MODE") else body).append(line)
    assert all(" -pn " in line for line in body), "fixture has no phase connection"

    def shifted(line, turns):
        tokens = line.split()
        index = tokens.index("-pn")
        tokens[index + 1] = f"{float(tokens[index + 1]) + turns:.1f}"
        return " ".join(tokens)

    shift_from = int(0.75 * len(body))
    paths = {}
    for label, turns in (("plain", 0), ("shifted", 1)):
        out = list(header) + [
            shifted(line, turns) if index >= shift_from else line
            for index, line in enumerate(body)
        ]
        path = directory / f"{label}.tim"
        path.write_text("\n".join(out) + "\n")
        paths[label] = path

    par = directory / "track2.par"
    par.write_text("\n".join(_lines(par_in)) + "\nTRACK -2\n")
    return par, paths, shift_from, par_in


def test_the_phase_connection_is_taken_from_tempo2(phase_connected_fixture):
    """H8. ``pulse_number_source == "tempo2"`` is required, not merely allowed."""
    from vela_jax import Engine

    par, paths, _, _ = phase_connected_fixture
    for path in paths.values():
        engine = Engine.from_tempo2(par, path)
        assert engine.pulse_number_source == "tempo2"


def test_a_bare_pn_flag_is_not_a_phase_connection(examples):
    """A ``-pn`` flag without ``TRACK -2`` is decoration, and we say so.

    Measured: bumping a ``-pn`` value by one turn on a tim with no ``TRACK``
    changes tempo2's residuals by nothing at all. Claiming ``pulse_number_
    source == "tempo2"`` there would advertise an authority tempo2 never
    exercised -- most of Vela.jl's fixtures are exactly that case.
    """
    from vela_jax import Engine

    engine = Engine.from_tempo2(examples / "sim_dd.par", examples / "sim_dd.tim")
    assert "pn" in set(engine.tempo2_columns.flags)
    assert engine.pulse_number_source == "model"


def test_a_dropped_phase_connection_would_be_a_whole_turn(phase_connected_fixture):
    """H8's tripwire: the fixture is built so silence is not an option."""
    from vela_jax import Engine

    par, paths, shift_from, par_in = phase_connected_fixture
    plain = Engine.from_tempo2(par, paths["plain"])
    moved = Engine.from_tempo2(par, paths["shifted"])
    difference = moved.residuals() - plain.residuals()

    # One turn over the *instantaneous* spin frequency, which on this fixture
    # (F1 = -4.2e-8 Hz/s) drifts by several per cent across the data span.
    period = 1.0 / plain.reference_spin_frequency()
    shifted = np.arange(len(difference)) >= shift_from

    assert np.max(np.abs(difference[~shifted])) < 1e-9, "unshifted rows moved"
    assert np.allclose(np.abs(difference[shifted]), period[shifted], rtol=1e-6)
    assert np.abs(difference[shifted]).min() > 0.5 * period.max()


# --- S4: the conventions flag, against libstempo (A.6.3) -------------------

#: A.6.3's budget. Deliberately loose: this is the first *external* constraint
#: on the A.4 conventions, not a parity program with libstempo.
S4_RMS_S = 50e-9


@pytest.fixture(scope="module")
def discriminating_ell1(examples, tmp_path_factory):
    """``J1227-6208`` as a plain ELL1, with the ingest choices matched.

    Three edits, each for a stated reason:

    * ``BINARY ELL1H -> ELL1`` (dropping ``H3``/``NHARMS``). The Roemer
      truncation A.4 is about applies to both, but Vela and tempo2 also
      disagree about the ELL1H *Shapiro* harmonics, by ~1.7 microseconds on
      this par -- a separate finding (SPEC 16.6), recorded in ``PARITY.md``,
      which would otherwise swamp the gate.
    * ``NE_SW 0``. tempo2 defaults the solar-wind density to 4 cm^-3 and PINT
      to 0; unpinned, that is ~500 ns of solar-wind delay difference and
      nothing to do with the binary.
    * ``CORRECT_TROPOSPHERE N``. tempo2 models it, this engine does not
      (SPEC 1.2's documented relaxation); ~3 microseconds peak to peak here.

    What is *not* touched is the eccentricity: ``EPS1``/``EPS2`` give
    ``e = 1.15e-3``, which is what makes the fixture discriminating. A
    near-circular EPTA ELL1 would pass 50 ns with the flag off and prove
    nothing.
    """
    directory = tmp_path_factory.mktemp("s4")
    dropped = {"NE_SW", "CORRECT_TROPOSPHERE", "H3", "H4", "STIGMA", "NHARMS"}
    out = []
    for line in _lines(examples / "J1227-6208.sim.par"):
        key = line.split()[0].upper()
        if key in dropped:
            continue
        out.append("BINARY ELL1" if key == "BINARY" else line)
    out += ["NE_SW 0.0", "CORRECT_TROPOSPHERE N"]
    par = directory / "J1227-6208.ell1.par"
    par.write_text("\n".join(out) + "\n")
    return par, examples / "J1227-6208.sim.tim"


def _libstempo_residuals(par, tim):
    """libstempo's own residuals for the par text *tempo2* is given."""
    import tempfile
    from pathlib import Path

    from libstempo.sandbox import tempopulsar

    from vela_jax.read_tempo2 import prepare_par

    tempo2_text, _, _ = prepare_par(par)
    directory = Path(tempfile.mkdtemp(prefix="vela_jax_s4_"))
    path = directory / "s4.par"
    path.write_text(tempo2_text)
    pulsar = tempopulsar(parfile=str(path), timfile=str(tim), dofit=False)
    return np.asarray(pulsar.residuals(removemean=False), dtype=float)


def _s4_rms(par, tim, conventions):
    from vela_jax import Engine

    engine = Engine.from_tempo2(par, tim, binary_conventions=conventions)
    difference = engine.residuals() - _libstempo_residuals(par, tim)
    return float(np.std(difference - difference.mean()))


def test_the_tempo2_conventions_reproduce_libstempo(discriminating_ell1):
    """S4 (A.6.3). RMS <= 50 ns with ``binary_conventions="tempo2"``."""
    pytest.importorskip("libstempo.sandbox")
    assert _s4_rms(*discriminating_ell1, "tempo2") < S4_RMS_S


def test_the_fixture_discriminates(discriminating_ell1):
    """S4's other half: the same comparison MUST fail under ``"pint"``.

    Without it the gate would be satisfied by any near-circular ELL1 and would
    constrain nothing. The measured gap is the 22.9 microseconds A.4 predicts
    for ``e = 1.2e-3``.
    """
    pytest.importorskip("libstempo.sandbox")
    assert _s4_rms(*discriminating_ell1, "pint") > 10 * S4_RMS_S
