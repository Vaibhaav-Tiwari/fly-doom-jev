"""Reduced connectome loading: real MaleCNS v1.0 subset or labeled fixture.

The reduced graph keeps the population structure the loop needs:

- ``visual_projection``  — sensory input population (retinal drive enters here)
- ``ol_intrinsic``       — optic-lobe intrinsic interneurons (intermediate)
- ``cx_intrinsic``       — central-complex (class == 'CX') neurons (intermediate)
- ``descending_neuron``  — motor/readout population (all of them)

Synapse-count weights are normalized per postsynaptic neuron (each column sums
to <= 1). That normalization is an engineering assumption, not biology; see
SCIENCE.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)

POPULATIONS = ("visual_projection", "ol_intrinsic", "cx_intrinsic", "descending_neuron")
MOTOR_POPULATION = "descending_neuron"

FIXTURE_SIZES = {
    "visual_projection": 96,
    "ol_intrinsic": 192,
    "cx_intrinsic": 96,
    "descending_neuron": 48,
}


@dataclass
class ReducedConnectome:
    """Sparse directed weighted graph plus per-neuron population labels."""

    weights: sp.csr_matrix          # [post, pre] normalized synaptic weight, float32
    body_ids: np.ndarray            # (N,) int64 MaleCNS bodyIds (fixture: synthetic ids)
    population: np.ndarray          # (N,) int8 index into POPULATIONS
    positions: np.ndarray           # (N, 3) float32 soma positions, NaN when unknown
    provenance: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return self.weights.shape[0]

    @property
    def is_real_malecns(self) -> bool:
        return self.provenance.get("source") == "malecns-v1.0"

    def population_mask(self, name: str) -> np.ndarray:
        return self.population == POPULATIONS.index(name)

    def population_indices(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.population_mask(name))

    def population_sizes(self) -> dict[str, int]:
        return {p: int(self.population_mask(p).sum()) for p in POPULATIONS}


def load_connectome(cfg: dict) -> ReducedConnectome:
    """Load the connectome per config.

    mode=auto    -> real reduced MaleCNS subset if raw data present, else fixture
    mode=reduced -> real subset required; raise if raw data missing
    mode=fixture -> deterministic fixture graph (tests only; clearly labeled)
    """
    neural = cfg["neural"]
    mode = neural.get("mode", "auto")
    data_dir = Path(neural.get("data_dir", "data/raw/malecns-v1.0"))
    cache = Path(neural.get("cache", "data/processed/malecns_reduced_v1.npz"))

    if mode == "fixture":
        log.warning("connectome mode=fixture: using deterministic synthetic test graph "
                    "(NOT MaleCNS data — permitted for tests/dev fixtures only)")
        return fixture_connectome(seed=int(neural.get("reduced", {}).get("seed", 42)))

    annotations = data_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    weights_file = data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    if annotations.exists() and weights_file.exists():
        return load_reduced_malecns(data_dir, cache, neural.get("reduced", {}))

    if mode == "reduced":
        raise FileNotFoundError(
            f"neural.mode=reduced requires MaleCNS v1.0 feather files in {data_dir}; "
            "run the data download (see README) or use mode: auto / fixture."
        )
    log.warning("MaleCNS raw data not found in %s — falling back to the deterministic "
                "fixture graph. This fallback is recorded in provenance and surfaced "
                "in telemetry and the dashboard.", data_dir)
    return fixture_connectome(seed=int(neural.get("reduced", {}).get("seed", 42)))


def load_reduced_malecns(data_dir: Path, cache: Path, sizes: dict) -> ReducedConnectome:
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return ReducedConnectome(
            weights=sp.csr_matrix((z["w_data"], z["w_indices"], z["w_indptr"]),
                                  shape=tuple(z["w_shape"])),
            body_ids=z["body_ids"], population=z["population"],
            positions=z["positions"],
            provenance=dict(source="malecns-v1.0", reduced=True, mode="roi-subset",
                            cache=str(cache), note="REDUCED subset of MaleCNS v1.0"),
        )

    import pyarrow.feather as feather

    ann = feather.read_table(
        data_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather").to_pandas()
    seed = int(sizes.get("seed", 42))
    rng = np.random.default_rng(seed)

    selected: dict[str, np.ndarray] = {}
    for pop, col, val in (("visual_projection", "superclass", "visual_projection"),
                          ("ol_intrinsic", "superclass", "ol_intrinsic"),
                          ("descending_neuron", "superclass", "descending_neuron")):
        ids = ann.loc[ann[col] == val, "bodyId"].to_numpy()
        cap = int(sizes.get(pop, len(ids)))
        if cap and len(ids) > cap:
            ids = np.sort(rng.choice(ids, size=cap, replace=False))
        selected[pop] = ids
    cx = ann.loc[ann["class"] == "CX", "bodyId"].to_numpy()
    cap = int(sizes.get("cx_intrinsic", len(cx)))
    if cap and len(cx) > cap:
        cx = np.sort(rng.choice(cx, size=cap, replace=False))
    selected["cx_intrinsic"] = cx

    body_ids = np.concatenate([selected[p] for p in POPULATIONS]).astype(np.int64)
    population = np.concatenate(
        [np.full(len(selected[p]), i, dtype=np.int8) for i, p in enumerate(POPULATIONS)])

    # soma positions (voxel units, 8nm); NaN when unknown
    pos_map: dict[int, tuple[float, float, float]] = {}
    sub = ann[ann["bodyId"].isin(set(body_ids.tolist()))]
    for bid, loc in zip(sub["bodyId"], sub["somaLocation"]):
        pos_map[int(bid)] = tuple(float(v) for v in loc) if loc is not None else (np.nan,) * 3
    positions = np.array([pos_map.get(int(b), (np.nan,) * 3) for b in body_ids],
                         dtype=np.float32)

    index = {int(b): i for i, b in enumerate(body_ids)}
    edges = feather.read_table(
        data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather").to_pandas()
    keep = edges["body_pre"].isin(index.keys()) & edges["body_post"].isin(index.keys())
    edges = edges.loc[keep]
    pre = edges["body_pre"].map(index).to_numpy()
    post = edges["body_post"].map(index).to_numpy()
    w = edges["weight"].to_numpy(dtype=np.float32)

    W = sp.csr_matrix((w, (post, pre)), shape=(len(body_ids), len(body_ids)))
    # normalize per postsynaptic neuron so total inbound drive is O(1)
    colsum = np.asarray(W.sum(axis=0)).ravel()  # note: axis=0 sums over post for each pre
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    scale = np.divide(1.0, rowsum, out=np.zeros_like(rowsum), where=rowsum > 0)
    W = sp.diags(scale) @ W
    W = W.astype(np.float32).tocsr()
    del colsum

    conn = ReducedConnectome(
        weights=W, body_ids=body_ids, population=population, positions=positions,
        provenance=dict(
            source="malecns-v1.0", reduced=True, mode="roi-subset",
            note="REDUCED subset of MaleCNS v1.0 (population-capped induced subgraph; "
                 "weights = synapse counts normalized per postsynaptic neuron)",
            sizes={p: int(len(selected[p])) for p in POPULATIONS},
            edges=int(W.nnz), seed=seed,
        ),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, w_data=W.data, w_indices=W.indices, w_indptr=W.indptr,
        w_shape=np.array(W.shape), body_ids=body_ids, population=population,
        positions=positions)
    log.info("reduced MaleCNS graph: %d neurons, %d edges (cached to %s)",
             conn.n_neurons, W.nnz, cache)
    return conn


def fixture_connectome(seed: int = 42, sizes: dict | None = None) -> ReducedConnectome:
    """Deterministic small graph with the same population structure.

    Test/dev fixture ONLY. Provenance is labeled 'fixture' so nothing downstream
    can mistake it for connectome data.
    """
    sizes = sizes or FIXTURE_SIZES
    rng = np.random.default_rng(seed)
    pops = list(POPULATIONS)
    counts = [int(sizes[p]) for p in pops]
    n = sum(counts)
    population = np.concatenate([np.full(c, i, dtype=np.int8) for i, c in enumerate(counts)])
    body_ids = np.arange(1, n + 1, dtype=np.int64)  # synthetic ids, not MaleCNS bodyIds

    # feedforward-biased random sparse graph: sensory -> intrinsic -> descending
    edges_pre, edges_post, edges_w = [], [], []
    bounds = np.concatenate([[0], np.cumsum(counts)])
    for pi in range(len(pops) - 1):
        src = np.arange(bounds[pi], bounds[pi + 1])
        for pj in range(pi + 1, len(pops)):
            dst = np.arange(bounds[pj], bounds[pj + 1])
            density = 0.15 if pj == pi + 1 else 0.03
            mask = rng.random((len(dst), len(src))) < density
            r, c = np.nonzero(mask)
            edges_pre.append(src[c]); edges_post.append(dst[r])
            edges_w.append(rng.random(len(r), dtype=np.float32) + 0.1)
    pre = np.concatenate(edges_pre); post = np.concatenate(edges_post)
    w = np.concatenate(edges_w).astype(np.float32)
    W = sp.csr_matrix((w, (post, pre)), shape=(n, n))
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    scale = np.divide(1.0, rowsum, out=np.zeros_like(rowsum), where=rowsum > 0)
    W = (sp.diags(scale) @ W).astype(np.float32).tocsr()

    positions = rng.random((n, 3), dtype=np.float32) * 100.0
    return ReducedConnectome(
        weights=W, body_ids=body_ids, population=population, positions=positions,
        provenance=dict(source="fixture", reduced=True, synthetic=True,
                        note="DETERMINISTIC SYNTHETIC FIXTURE — not connectome data; "
                             "permitted for tests/dev only",
                        sizes=dict(zip(pops, counts)), edges=int(W.nnz), seed=seed),
    )
