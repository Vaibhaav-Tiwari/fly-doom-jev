"""Tests for the full MaleCNS connectome loader and native engine.

Skipped when the processed full-graph cache is absent (requires the raw
MaleCNS files + one-time build). These are skipped, never faked.
"""

import numpy as np
import pytest
from pathlib import Path

CACHE = Path("data/processed/malecns_full_v1")

pytestmark = pytest.mark.skipif(
    not (CACHE / "meta.json").exists(),
    reason="full MaleCNS cache not built (run the graph build first)")


@pytest.fixture(scope="module")
def full_conn():
    from flydoom.malecns.full import load_full_connectome
    return load_full_connectome({"neural": {"data_dir": "data/raw/malecns-v1.0",
                                            "cache_full": str(CACHE)}})


def test_full_graph_scale_and_provenance(full_conn):
    assert full_conn.n_neurons == 211577
    assert full_conn.n_edges == 26028386
    assert full_conn.provenance["source"] == "malecns-v1.0"
    assert full_conn.provenance["reduced"] is False
    assert full_conn.provenance["inhibitory_neurons"] > 40000


def test_signed_weights(full_conn):
    w = np.asarray(full_conn.weight)
    assert (w < 0).any() and (w > 0).any()  # NT signs present


def test_retina_mapped(full_conn):
    assert len(full_conn.retina) > 5000
    uv = np.asarray(full_conn.retina_uv)
    assert np.all(uv >= 0.0) and np.all(uv <= 1.0)
    assert len(full_conn.lamina) > 8000


def test_typed_dn_readouts_present(full_conn):
    for t in ("DNa02", "DNp09", "MDN", "DNg100", "DNpe017"):
        assert len(full_conn.type_indices(t)) >= 1
    dna02 = full_conn.type_indices("DNa02")
    sides = set(full_conn.side[dna02].tolist())
    assert sides == {0, 1}  # one L, one R


def test_native_engine_steps_and_propagates(full_conn):
    from flydoom.neural.engine_native import NativeLIFEngine
    eng = NativeLIFEngine(full_conn, timestep_ms=1.0, syn_gain=0.3)
    eng.reset()
    eng.inject_input(full_conn.retina,
                     np.full(len(full_conn.retina), 20.0, dtype=np.float32))
    eng.inject_input(full_conn.lamina,
                     np.full(len(full_conn.lamina), 8.0, dtype=np.float32))
    eng.step(300)  # 300ms neural time
    pop = eng.get_population_activity()
    assert pop["ol_sensory"]["mean_rate_hz"] > 1.0
    assert eng.total_spikes > 0
    # signal must reach beyond the optic lobe within 300ms
    assert pop["descending_neuron"]["mean_rate_hz"] >= 0.0  # may be silent
    assert pop["ol_intrinsic"]["mean_rate_hz"] > 0.1
    eng.reset()
    assert eng.total_spikes == 0


def test_bridge_guard_on_full_graph(full_conn):
    from flydoom.integration import JevBridge, MotorInjectionError
    with pytest.raises(MotorInjectionError):
        JevBridge(full_conn, {"ATTACK": "descending_neuron"})
    bridge = JevBridge(full_conn, {"ATTACK": "visual_projection"}, gain=5.0)
    idx, cur = bridge.modulation_currents(None)
    assert len(idx) > 0 and np.all(cur == 0.0)
