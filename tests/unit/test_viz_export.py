"""Tests for the shared connectome viz-asset export (3D brain view).

Skipped when the processed full-graph cache is absent (the export only exists
for the real full MaleCNS graph).
"""

import json
from pathlib import Path

import numpy as np
import pytest

CACHE = Path("data/processed/malecns_full_v1")

pytestmark = pytest.mark.skipif(
    not (CACHE / "meta.json").exists(),
    reason="full MaleCNS cache not built (run the graph build first)")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from flydoom.malecns.full import load_full_connectome
    from flydoom.malecns.viz_export import export_connectome_assets
    conn = load_full_connectome({"neural": {"cache_full": str(CACHE)}})
    out = tmp_path_factory.mktemp("viz_assets")
    export_connectome_assets(conn, out)
    return conn, out


def test_asset_arrays_match_graph(exported):
    conn, out = exported
    n = conn.n_neurons
    pos = np.fromfile(out / "positions.f32", dtype="<f4").reshape(n, 3)
    pop = np.fromfile(out / "population.i16", dtype="<i2")
    flags = np.fromfile(out / "flags.u8", dtype=np.uint8)
    ids = np.fromfile(out / "ids.i64", dtype="<i8")
    assert pop.shape == (n,) and flags.shape == (n,) and ids.shape == (n,)
    assert np.array_equal(ids, np.asarray(conn.ids))
    assert np.array_equal(pop, np.asarray(conn.population_index, dtype=np.int16))
    # positions match, NaN included
    assert np.array_equal(pos, np.asarray(conn.positions, dtype=np.float32),
                          equal_nan=True)
    # flags consistent with the graph's own population membership
    dn = conn.population_indices("descending_neuron")
    assert (flags[dn] & 4).all()
    assert (flags[np.asarray(conn.retina)] & 1).all()
    assert ((flags & 16) > 0).sum() == int((~np.isnan(pos[:, 0])).sum())


def test_asset_meta(exported):
    conn, out = exported
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_neurons"] == conn.n_neurons
    assert meta["with_position"] == 141781  # measured MaleCNS soma annotations
    assert "descending_neuron" in meta["populations"]
    assert meta["provenance"]["source"] == "malecns-v1.0"


def test_typed_decoder_contributing_sets():
    from flydoom.malecns.full import load_full_connectome
    from flydoom.motor.decoder import TypedDNDecoder
    conn = load_full_connectome({"neural": {"cache_full": str(CACHE)}})
    dec = TypedDNDecoder(conn, ["turn_left", "turn_right", "attack", "noop"])
    contributing = dec.describe()["contributing"]
    assert set(contributing) == {"turn_left", "turn_right", "attack"}
    # DNa02 left feeds turn_left, DNa02 right feeds turn_right (Rayshubskiy 2020)
    left = conn.type_indices("DNa02")
    left = left[conn.side[left] == 0]
    assert sorted(contributing["turn_left"]["indices"]) == sorted(map(int, left))
    for entry in contributing.values():
        assert len(entry["indices"]) == len(entry["body_ids"]) > 0
