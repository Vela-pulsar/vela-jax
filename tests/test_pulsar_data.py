"""The pulsar product (SPEC Addendum B; gates P1, P2, P3, P5, P6).

These are the gates v1 conspicuously did not have: the identities behind
``toas`` and ``freqs`` are checked as array equalities, the one-source rule is
enforced by monkeypatch rather than by docstring, and the row order is a
tested property rather than an unstated assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from vela_jax import TimingPulsar

CASES = ["NGC6440E", "sim_dd", "sim_sw", "sim_dmx", "J1802-2124.sim"]


@pytest.fixture(scope="module")
def pulsars(engine_factory):
    cache = {}

    def build(name):
        if name not in cache:
            cache[name] = TimingPulsar(engine_factory(name))
        return cache[name]

    return build


# --- P3: one row order -----------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_the_freeze_does_not_reorder_toas(engine_factory, name):
    """P3. Row ``i`` is the timing package's row ``i``, in every frozen array.

    Not "sorted", not "sorted stably", not "sorted by a documented key" --
    *unchanged*. A timing package that permutes its own rows publishes one
    order to whoever holds the engine and another to whoever holds the
    timing-package object, and the two are only ever reconciled by a
    permutation that a consumer has to remember to apply.
    """
    engine = engine_factory(name)
    table = engine.pint_toas.table
    n = engine.toa_count

    tau = np.asarray(engine.frozen.tau)[:-1]
    table_tau = (
        np.asarray(table["tdbld"].value, dtype=np.longdouble)
        - np.longdouble(engine.pint_model["PEPOCH"].value)
    ) * np.longdouble(86400.0)
    assert np.array_equal(tau, np.asarray(table_tau, dtype=float))

    assert np.array_equal(
        np.asarray(engine.frozen.freq_hz)[:-1],
        np.asarray(engine.pint_toas.get_freqs().to_value("Hz")),
    )
    assert np.array_equal(
        np.asarray(engine.toa_columns.telescope),
        np.asarray(engine.pint_toas.get_obss()),
    )
    assert len(engine.toa_columns.sat_seconds) == n


def test_no_row_permutation_is_published_anywhere(engine_factory, pulsars):
    """The rule, as a shape: there is no permutation to get wrong.

    ``data_order`` existed briefly, as "provenance only". A consumer promptly
    applied it (MetaPulsar's leg engine), which is how a package boundary
    grows a permutation protocol. The absence is the contract.
    """
    engine = engine_factory("sim_dd")
    psr = pulsars("sim_dd")
    for holder in (engine, psr, psr.data, engine.toa_columns):
        for attribute in ("data_order", "_isort", "_iisort", "sort_data", "toa_index"):
            assert not hasattr(holder, attribute), (type(holder).__name__, attribute)


# --- P1: the one-source identities -----------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_toas_and_freqs_are_the_chains_own_barycentric_snapshot(engine_factory, name):
    """P1. Array equality, not "close": these are definitions, not estimates."""
    engine = engine_factory(name)
    data = engine.pulsar_data()
    delay, doppler = engine.reference_barycentric()

    expected_toas = np.asarray(engine.toa_columns.tdb_seconds - delay, dtype=float)
    assert np.array_equal(data.toas, expected_toas)

    freq_hz = np.asarray(engine.frozen.freq_hz, dtype=float)[: engine.toa_count]
    assert np.array_equal(data.freqs, freq_hz * (1.0 - doppler) / 1.0e6)
    assert np.array_equal(
        data.stoas, np.asarray(engine.toa_columns.sat_seconds, dtype=float)
    )


@pytest.mark.parametrize("name", ["sim_dd", "J1802-2124.sim", "NGC6440E", "sim_fd"])
def test_the_barycentric_cutoff_is_pints(engine_factory, name):
    """P1. The cross-check that makes a wrong cutoff impossible to miss.

    Vela's ``corrected_toa_value`` subtracts *every* delay; PINT's barycentric
    arrival stops before the binary. On an ELL1/DD pulsar the two differ by the
    binary Roemer delay -- **seconds** -- so this comparison fails loudly if
    the snapshot is taken at the wrong stage. The difference is formed in
    longdouble because a float64 MJD-second carries ~1e-6 s of its own.
    """
    engine = engine_factory(name)
    ld = np.longdouble
    theirs = np.asarray(
        engine.pint_model.get_barycentric_toas(engine.pint_toas).value, dtype=ld
    ) * ld(86400.0)
    ours = engine.toa_columns.tdb_seconds - engine.reference_barycentric()[0]
    assert np.max(np.abs(np.asarray(theirs - ours, dtype=float))) < 1e-7


def test_an_infinite_frequency_toa_is_published_as_infinite(examples, tmp_path):
    """The graph's finite placeholder must not reach the record.

    ``freeze`` substitutes a finite frequency for an infinite one so the
    dispersive derivative stays defined (R3.7) -- a TOA that carries no
    dispersive delay at all. Publishing that placeholder as ``1000 MHz`` would
    be a made-up measurement; PINT's ``barycentric_radio_freq`` reports
    ``inf``, and so does the record.
    """
    from vela_jax import Engine

    source = examples / "NGC6440E.tim"
    lines = source.read_text().splitlines()
    body = [
        i
        for i, line in enumerate(lines)
        if line.split()[:1] not in ([], ["FORMAT"], ["C"], ["MODE"])
    ]
    target = body[3]
    # This fixture is in the fixed-width Princeton format, so the frequency is
    # overwritten in place rather than re-joined: re-spacing the line changes
    # which field is which. PINT reads frequency 0 as infinite.
    original = lines[target]
    column = original.index("1724.609")
    lines[target] = (
        original[:column] + "0.000000".ljust(len("1724.609")) + original[column + 8 :]
    )
    tim = tmp_path / "inf.tim"
    tim.write_text("\n".join(lines) + "\n")

    engine = Engine.from_files(examples / "NGC6440E.par", tim)
    freqs = engine.pulsar_data().freqs
    row = body.index(target)
    assert np.isinf(freqs[row])
    assert np.all(np.isfinite(np.delete(freqs, row)))
    assert np.array_equal(freqs, engine.toa_rows().freqs, equal_nan=True)


@pytest.mark.parametrize("name", CASES)
def test_freqs_are_the_solar_system_doppler_not_the_full_chains(engine_factory, name):
    """P1. PINT's ``barycentric_radio_freq`` cuts at the same place we do."""
    engine = engine_factory(name)
    theirs = np.asarray(
        engine.pint_model.barycentric_radio_freq(engine.pint_toas), dtype=float
    )
    ours = engine.pulsar_data().freqs
    assert np.max(np.abs(ours - theirs) / theirs) < 1e-12


# --- P2: the one-source rule, enforced -------------------------------------


def test_no_pint_physics_pass_populates_the_record(engine_factory, monkeypatch):
    """P2. The v1 violation, made impossible.

    v1 filled ``toas`` and ``freqs`` with a second PINT pass over the frozen
    table -- when tempo2 read the files, PINT physics over injected columns. Every
    method that could do that raises here while the record is built.
    """
    import pint.models.astrometry as astrometry
    import pint.models.timing_model as timing_model
    import pint.residuals as residuals

    engine = engine_factory("sim_dd")
    engine.residual_jacobian()  # R9.6: paid once, before the guard goes up

    def forbidden(name):
        def raiser(*args, **kwargs):
            raise AssertionError(f"the pulsar record called PINT's {name}")

        return raiser

    monkeypatch.setattr(
        timing_model.TimingModel,
        "get_barycentric_toas",
        forbidden("get_barycentric_toas"),
    )
    monkeypatch.setattr(
        astrometry.Astrometry,
        "barycentric_radio_freq",
        forbidden("barycentric_radio_freq"),
    )
    monkeypatch.setattr(
        timing_model.TimingModel, "designmatrix", forbidden("designmatrix")
    )
    monkeypatch.setattr(
        astrometry.Astrometry, "ssb_to_psb_xyz_ICRS", forbidden("ssb_to_psb_xyz_ICRS")
    )
    monkeypatch.setattr(residuals, "Residuals", forbidden("Residuals"))

    from vela_jax.pulsar_data import build_pulsar_data

    data = build_pulsar_data(engine)
    assert len(data.toas) == engine.toa_count


# --- B.2 / B.6: the record and its forwarding ------------------------------


@pytest.mark.parametrize("name", CASES)
def test_every_row_array_has_the_same_length(pulsars, name):
    psr = pulsars(name)
    n = len(psr.toas)
    for field in ("residuals", "toaerrs", "freqs", "stoas", "backend_flags"):
        assert len(getattr(psr, field)) == n, field
    assert psr.Mmat.shape == (n, len(psr.fitpars))
    assert psr.pos_t.shape == (n, 3)
    assert psr.sunssb.shape == (n, 6)
    assert psr.planetssb.shape == (n, 9, 6)
    for values in psr.flags.values():
        assert len(values) == n


def test_the_pulsar_forwards_the_record_without_copying(pulsars):
    """R-B6.1. Composition: every forwarded array *is* the record's array."""
    psr = pulsars("sim_dd")
    for name in TimingPulsar.FORWARDED:
        assert hasattr(psr, name), name
        value = getattr(psr, name)
        if isinstance(value, np.ndarray):
            assert value is getattr(psr.data, name), name
    assert psr.producer == psr.data.producer


def test_the_record_is_read_only(pulsars):
    """Attribute-frozen: arrays are views of producer storage (psrdata R-1.3)."""
    psr = pulsars("sim_dd")
    with pytest.raises(Exception):
        psr.data.name = "other"


def test_the_matrix_is_the_engines_own(pulsars):
    psr = pulsars("sim_dd")
    engine = psr.timing_engine("vela_jax")
    assert tuple(psr.fitpars) == tuple(engine.fitpars)
    assert np.array_equal(psr.Mmat, engine.design_matrix())
    assert np.array_equal(psr.Mmat, -psr.engine.residual_jacobian())
    assert np.array_equal(psr.residuals, psr.engine.residuals())


def test_uncertainties_keep_longdouble_digits(pulsars):
    """Producer contract §9.7 / R-3.3.3: no float64 round trip.

    PINT stores F0's uncertainty as longdouble. ``float`` then ``.16g``
    drops digits the par token and the timing package still hold.
    """
    psr = pulsars("sim_dd")
    param = psr.engine.pint_model["F0"]
    stored = str(np.longdouble(param.uncertainty_value))
    truncated = format(float(param.uncertainty_value), ".16g")
    assert stored != truncated
    assert psr.parameters["F0"].uncertainty == stored


@pytest.mark.parametrize("name", CASES)
def test_the_line_of_sight_is_the_one_the_chain_used(pulsars, name):
    psr = pulsars(name)
    if not np.any(psr.pos_t):
        pytest.skip("no astrometry component (bare rotator)")
    assert np.allclose(np.linalg.norm(psr.pos_t, axis=1), 1.0)
    assert np.max(np.abs(psr.pos_t - psr.pos)) < 1e-6


# --- P5: dmx ---------------------------------------------------------------


def test_dmx_is_the_real_table_and_partitions_its_toas(pulsars):
    """P5. ``{}`` on a DMX pulsar is a refused state (B.3.2)."""
    psr = pulsars("sim_dmx")
    model = psr.engine.pint_model
    assert psr.dmx, "a DMX pulsar with an empty dmx table"

    days = np.asarray(psr.stoas) / 86400.0
    covered = np.zeros(len(days), dtype=int)
    for key, entry in psr.dmx.items():
        assert entry["DMX"] == pytest.approx(float(model[key].value))
        assert entry["fit"] is (not model[key].frozen)
        assert entry["DMXR1"] < entry["DMXR2"]
        covered += (days >= entry["DMXR1"]) & (days <= entry["DMXR2"])
    # Exclusive windows, and every TOA in exactly one of them.
    assert np.all(covered == 1)


def test_dmx_is_none_without_the_component(pulsars):
    assert pulsars("sim_dd").dmx is None


def test_dm_never_raises(pulsars):
    assert pulsars("sim_dd").dm > 0.0


# --- P6: planetssb ---------------------------------------------------------


@pytest.mark.parametrize("name", ["NGC6440E", "sim_sw"])
def test_planet_slots_carry_positions_and_nan_velocities(pulsars, name):
    """P6, the half that needs no Enterprise: shape, slots and NaN policy."""
    psr = pulsars(name)
    for slot in (2, 4, 5, 6, 7):
        assert np.all(np.isfinite(psr.planetssb[:, slot, :3])), slot
    # Venus is frozen by both timing packages, so it is filled -- a superset of
    # Enterprise's PINT path, which leaves it NaN (B.3.3).
    assert np.all(np.isfinite(psr.planetssb[:, 1, :3]))
    assert np.all(np.isnan(psr.planetssb[:, :, 3:])), "velocities have no reader"
    for slot in (0, 3, 8):
        assert np.all(np.isnan(psr.planetssb[:, slot, :3])), slot
    assert np.all(np.isfinite(psr.sunssb[:, :3]))
    assert np.array_equal(psr.sunssb[:, 3:], np.zeros((len(psr.toas), 3)))


def test_planet_positions_match_enterprises_pint_path(examples, pulsars):
    """P6. Against Enterprise itself, slot by slot, Venus excluded.

    Enterprise's PINT path leaves slot 1 NaN because PINT once had no Venus
    column; it does now, we freeze it, and filling it is a superset. That is
    why the gate excludes slot 1 rather than NaN-ing ours to match.
    """
    enterprise = pytest.importorskip("enterprise.pulsar")

    psr = pulsars("NGC6440E")
    theirs = enterprise.Pulsar(
        str(examples / "NGC6440E.par"),
        str(examples / "NGC6440E.tim"),
        planets=True,
        timing_package="pint",
    )
    ours_order = np.argsort(psr.stoas, kind="stable")
    theirs_order = np.argsort(np.asarray(theirs.stoas), kind="stable")
    assert np.allclose(
        np.asarray(psr.stoas)[ours_order],
        np.asarray(theirs.stoas)[theirs_order],
        atol=1e-6,
    )
    for slot in (2, 4, 5, 6, 7):
        assert (
            np.max(
                np.abs(
                    psr.planetssb[ours_order, slot, :3]
                    - np.asarray(theirs.planetssb)[theirs_order, slot, :3]
                )
            )
            < 1e-10
        ), slot
    assert (
        np.max(
            np.abs(
                psr.sunssb[ours_order, :3] - np.asarray(theirs.sunssb)[theirs_order, :3]
            )
        )
        < 1e-10
    )
