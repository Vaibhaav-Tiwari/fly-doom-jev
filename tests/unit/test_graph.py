import numpy as np
import pytest

from flydoom.malecns.graph import (MOTOR_POPULATION, POPULATIONS,
                                   fixture_connectome, load_connectome)


def test_fixture_graph_labeled_not_real(fixture_connectome):
    assert not fixture_connectome.is_real_malecns
    assert fixture_connectome.provenance["source"] == "fixture"
    assert fixture_connectome.provenance["synthetic"] is True


def test_fixture_graph_structure(fixture_connectome):
    sizes = fixture_connectome.population_sizes()
    assert set(sizes) == set(POPULATIONS)
    assert sizes[MOTOR_POPULATION] > 0
    W = fixture_connectome.weights
    assert W.shape == (fixture_connectome.n_neurons,) * 2
    # normalized: each postsynaptic row sums to <= 1 (+fp tolerance)
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    assert rowsum.max() <= 1.0 + 1e-5


def test_fixture_graph_deterministic():
    a, b = fixture_connectome(seed=3), fixture_connectome(seed=3)
    assert (a.weights != b.weights).nnz == 0
    np.testing.assert_array_equal(a.body_ids, b.body_ids)


def test_reduced_mode_refuses_missing_data(tmp_path):
    cfg = {"neural": {"mode": "reduced", "data_dir": str(tmp_path / "nope")}}
    with pytest.raises(FileNotFoundError):
        load_connectome(cfg)


def test_auto_mode_falls_back_to_labeled_fixture(tmp_path):
    cfg = {"neural": {"mode": "auto", "data_dir": str(tmp_path / "nope"),
                      "reduced": {"seed": 42}}}
    conn = load_connectome(cfg)
    assert conn.provenance["source"] == "fixture"  # explicit, not silent
