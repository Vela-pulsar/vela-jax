"""What vela-jax owes the feather contract: its own record round-trips.

The schema itself -- lossless round trip, both consumers' stock readers, the
metadata block, a refused foreign schema -- is :mod:`psrdata`'s and is tested
there against a synthetic record. What is this package's is narrower and not
testable there: that the record *an engine produces* is a valid one, and says
it came from here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from psrdata import SCHEMA, PulsarData

from vela_jax import TimingPulsar


@pytest.fixture(scope="module")
def written(engine_factory, tmp_path_factory):
    psr = TimingPulsar(engine_factory("sim_dd"))
    return psr, psr.to_feather(tmp_path_factory.mktemp("feather") / "sim_dd.feather")


def test_an_engine_built_record_round_trips(written):
    psr, path = written
    back = PulsarData.from_feather(path)
    for field in ("toas", "stoas", "toaerrs", "residuals", "freqs", "Mmat", "pos_t"):
        assert np.array_equal(
            np.asarray(getattr(psr, field)), np.asarray(getattr(back, field))
        ), field
    assert np.array_equal(psr.planetssb, back.planetssb, equal_nan=True)
    assert tuple(back.fitpars) == tuple(psr.fitpars)


def test_the_file_says_where_it_came_from(written):
    """``producer`` and the data-set mappings let a reader tell a vela-jax
    record from a MetaPulsar combination without opening the arrays."""
    import pyarrow.feather
    from psrdata import SINGLE_KEY

    psr, path = written
    meta = json.loads(pyarrow.feather.read_table(str(path)).schema.metadata[b"json"])
    assert meta["schema"] == SCHEMA
    assert meta["producer"] == "vela_jax"
    assert meta["timing_package"] == {SINGLE_KEY: "vela_jax"}
    assert meta["partim_compatibility"] == {SINGLE_KEY: psr.engine.timing_package}
    assert set(meta["parameters"]) >= set(psr.fitpars)


def test_the_metadata_block_rebuilds_a_linear_analysis(written):
    """The claim the metadata exists for, checked against the engine that wrote it.

    nltiming's ``LinearTimingEngine.from_feather`` reads the file alone -- no
    par, no tim, no timing package -- and must reproduce the producing
    engine's own residual on a linear axis.
    """
    pytest.importorskip("nltiming")
    from nltiming.engine_support import LinearTimingEngine

    psr, path = written
    engine = LinearTimingEngine.from_feather(path)
    assert tuple(engine.fitpars) == tuple(psr.fitpars)

    delta = np.zeros(len(engine.fitpars))
    delta[engine.fitpars.index("PHOFF")] = 1e-6
    assert np.allclose(
        engine.residual_delta(delta), psr.engine.residual_delta(delta), atol=1e-15
    )
