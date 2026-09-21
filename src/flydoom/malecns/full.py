"""Full MaleCNS v1.0 connectome: builder and loader.

Builds a signed, inbound-normalized CSR graph (by presynaptic neuron, for spike
propagation) from the official flat-connectome feather files, plus annotation
arrays (types, superclasses, sides, soma positions, ommatidia hexes) and the
retinotopic receptor arrays the visual pathway needs.

The build is cached to a directory of .npy files (memory-mapped on load, so
startup does not require loading the full graph into RAM eagerly). Raw data is
never modified and never committed.

Signs come from per-neuron consensus neurotransmitter predictions:
acetylcholine +1, GABA -1, glutamate -1 (ionotropic inhibitory in insects),
everything else +1 (documented as an engineering default in SCIENCE.md).
Magnitude: synapse counts normalized per postsynaptic neuron (sum |w_in| = 1).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"

NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}
DEFAULT_SIGN = 1.0

# Coarse populations used for telemetry/dashboard summaries in full mode.
# Each entry: label -> predicate on annotation columns.
COARSE_POPULATIONS = [
    "ol_sensory", "visual_projection", "visual_centrifugal", "ol_intrinsic",
    "cx_intrinsic", "cb_intrinsic", "cb_sensory", "ascending_neuron",
    "descending_neuron", "vnc_sensory", "vnc_intrinsic", "vnc_motor", "other",
]

PHOTORECEPTOR_PREFIXES = ("R1-R6", "R7", "R8")
LAMINA_TYPES = ("L1", "L2", "L3", "L4", "L5")


@dataclass
class FullConnectome:
    """Full-scale graph. Arrays may be memory-mapped (read-only)."""

    ids: np.ndarray            # (N,) int64 bodyIds
    ptr: np.ndarray            # (N+1,) int64 CSR outgoing-edge pointers (by pre)
    post: np.ndarray           # (E,) int32 postsynaptic indices
    weight: np.ndarray         # (E,) float32 signed, inbound-normalized
    superclass: np.ndarray     # (N,) int16 code into SUPERCLASS_CODES
    cell_type: np.ndarray      # (N,) unicode array ('' when unknown)
    side: np.ndarray           # (N,) int8: -1 unknown, 0 L, 1 R, 2 midline
    positions: np.ndarray      # (N, 3) float32 soma positions, NaN unknown
    retina: np.ndarray         # (R,) int32 photoreceptor indices
    retina_uv: np.ndarray      # (R, 2) float32 viewport coordinates in [0,1]
    retina_channel: np.ndarray # (R,) int8 0=luma 1=green 2=blue (R8 spectral proxy)
    lamina: np.ndarray         # (L,) int32 lamina interneuron indices
    population_index: np.ndarray  # (N,) int16 code into COARSE_POPULATIONS
    provenance: dict

    SUPERCLASS_CODES: tuple = ()
    n_neurons: int = 0
    n_edges: int = 0

    def population_mask(self, name: str) -> np.ndarray:
        return self.population_index == COARSE_POPULATIONS.index(name)

    def population_indices(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.population_mask(name))

    def population_sizes(self) -> dict[str, int]:
        return {p: int(self.population_mask(p).sum()) for p in COARSE_POPULATIONS}

    def type_indices(self, cell_type: str) -> np.ndarray:
        return np.flatnonzero(self.cell_type == cell_type)


def load_full_connectome(cfg: dict) -> FullConnectome:
    neural = cfg["neural"]
    data_dir = Path(neural.get("data_dir", "data/raw/malecns-v1.0"))
    cache = Path(neural.get("cache_full", "data/processed/malecns_full_v1"))
    if (cache / "meta.json").exists():
        return _load_cache(cache)
    return build_full_graph(data_dir, cache)


def _load_cache(cache: Path) -> FullConnectome:
    meta = json.loads((cache / "meta.json").read_text())
    def arr(name, mmap=True):
        return np.load(cache / f"{name}.npy", mmap_mode="r" if mmap else None,
                       allow_pickle=False)
    conn = FullConnectome(
        ids=arr("ids"), ptr=arr("ptr"), post=arr("post"), weight=arr("weight"),
        superclass=arr("superclass"), cell_type=arr("cell_type", mmap=False),
        side=arr("side"), positions=arr("positions"),
        retina=arr("retina"), retina_uv=arr("retina_uv"),
        retina_channel=arr("retina_channel"), lamina=arr("lamina"),
        population_index=arr("population_index"),
        SUPERCLASS_CODES=tuple(meta["superclass_codes"]),
        n_neurons=meta["n_neurons"], n_edges=meta["n_edges"],
        provenance=meta["provenance"])
    log.info("full MaleCNS graph mapped: %d neurons, %d edges (cache %s)",
             conn.n_neurons, conn.n_edges, cache)
    return conn


def build_full_graph(data_dir: Path, cache: Path, cfg_norm: str = "raw") -> FullConnectome:
    import pyarrow.feather as feather
    t0 = time.time()

    ann = feather.read_table(data_dir / ANNOTATIONS).to_pandas()
    edges = feather.read_table(data_dir / WEIGHTS).to_pandas()
    log.info("loaded %d annotations, %d edges (%.0fs)", len(ann), len(edges),
             time.time() - t0)

    ann_ids = np.sort(ann["bodyId"].to_numpy(np.int64))
    pre_b = edges["body_pre"].to_numpy(np.int64)
    post_b = edges["body_post"].to_numpy(np.int64)
    w_all = edges["weight"].to_numpy(np.float32)
    # Neuron-level graph: restrict to annotated bodies. connectome-weights is
    # segment-to-segment and includes unannotated fragments; dropping edges with
    # an unannotated endpoint is the standard neuron-level projection.
    keep = np.isin(pre_b, ann_ids) & np.isin(post_b, ann_ids)
    ids = ann_ids
    n = len(ids)
    pre_b = pre_b[keep]
    post_b = post_b[keep]
    w_raw = w_all[keep]
    del edges, w_all, keep
    log.info("neuron-level projection: %d edges between %d annotated bodies",
             len(w_raw), n)

    pre = np.searchsorted(ids, pre_b)
    post = np.searchsorted(ids, post_b).astype(np.int32)
    del pre_b, post_b

    # CSR by presynaptic neuron
    order = np.argsort(pre, kind="stable")
    pre_sorted = pre[order]
    post_sorted = post[order]
    w_sorted = w_raw[order]
    del pre, post, w_raw, order
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ptr, pre_sorted + 1, 1)
    np.cumsum(ptr, out=ptr)

    # neurotransmitter sign per presynaptic neuron
    sign = np.full(n, DEFAULT_SIGN, dtype=np.float32)
    try:
        nt = feather.read_table(data_dir / NEUROTRANSMITTERS).to_pandas()
        nt_ids = np.searchsorted(ids, nt["body"].to_numpy(np.int64))
        in_graph = (nt_ids < n) & (ids[np.minimum(nt_ids, n - 1)]
                                   == nt["body"].to_numpy(np.int64))
        nt_sign = nt["consensus_nt"].map(NT_SIGN).fillna(DEFAULT_SIGN).to_numpy(np.float32)
        sign[nt_ids[in_graph]] = nt_sign[in_graph]
        log.info("neurotransmitter signs: %d neurons inhibitory",
                 int((sign < 0).sum()))
    except FileNotFoundError:
        log.warning("%s missing; all synapses excitatory (fallback, recorded in "
                    "provenance)", NEUROTRANSMITTERS)
    weight = (w_sorted * sign[pre_sorted]).astype(np.float32)
    del w_sorted, pre_sorted

    # Synapse counts kept raw (signed). The engine applies a global synaptic
    # efficacy; inbound normalization is available but NOT the default because
    # it kills propagation (each edge's weight ~ 1/indegree). See SCIENCE.md.
    norm = str(cfg_norm or "raw")
    if norm == "inbound":
        absum = np.bincount(post_sorted, weights=np.abs(weight), minlength=n)
        weight *= (1.0 / np.maximum(absum, 1e-9))[post_sorted]
        weight = weight.astype(np.float32)

    # annotation arrays aligned to ids
    ann = ann.set_index("bodyId").reindex(ids)
    superclass_names = sorted(ann["superclass"].dropna().unique().tolist())
    sc_code = {s: i for i, s in enumerate(superclass_names)}
    superclass = ann["superclass"].map(sc_code).fillna(-1).to_numpy(np.int16)
    cell_type = ann["type"].fillna("").to_numpy()
    side_map = {"L": 0, "R": 1, "M": 2}
    side = ann["somaSide"].map(side_map).fillna(-1).to_numpy(np.int8)
    positions = np.full((n, 3), np.nan, dtype=np.float32)
    has_pos = ann["somaLocation"].notna().to_numpy()
    positions[has_pos] = np.array(
        [tuple(map(float, v)) for v in ann.loc[has_pos, "somaLocation"]],
        dtype=np.float32)

    population_index = _coarse_population_index(ann, superclass, superclass_names)

    # --- retinotopic receptor arrays -------------------------------------
    hex1 = ann["assignedOlHex1"].to_numpy()
    hex2 = ann["assignedOlHex2"].to_numpy()
    retina, retina_uv, retina_channel, retina_stats = _build_retina(
        ann, ids, ptr, post_sorted, weight, hex1, hex2, side)
    lamina_mask = np.isin(cell_type, LAMINA_TYPES)
    lamina = np.flatnonzero(lamina_mask).astype(np.int32)

    provenance = dict(
        source="malecns-v1.0", reduced=False,
        note="FULL MaleCNS v1.0 connectome; signed NT weights (raw synapse counts)",
        n_neurons=int(n), n_edges=int(len(weight)),
        inhibitory_neurons=int((sign < 0).sum()),
        retina=retina_stats,
        build_seconds=round(time.time() - t0, 1),
    )
    conn = FullConnectome(
        ids=ids, ptr=ptr, post=post_sorted, weight=weight,
        superclass=superclass, cell_type=cell_type.astype(str), side=side,
        positions=positions, retina=retina.astype(np.int32),
        retina_uv=retina_uv.astype(np.float32),
        retina_channel=retina_channel.astype(np.int8), lamina=lamina,
        population_index=population_index,
        SUPERCLASS_CODES=tuple(superclass_names),
        n_neurons=int(n), n_edges=int(len(weight)), provenance=provenance)

    cache.mkdir(parents=True, exist_ok=True)
    for name in ("ids", "ptr", "post", "weight", "superclass", "cell_type",
                 "side", "positions", "retina", "retina_uv", "retina_channel",
                 "lamina", "population_index"):
        np.save(cache / f"{name}.npy", getattr(conn, name))
    (cache / "meta.json").write_text(json.dumps({
        "n_neurons": conn.n_neurons, "n_edges": conn.n_edges,
        "superclass_codes": list(conn.SUPERCLASS_CODES),
        "provenance": provenance}, indent=2))
    log.info("full graph built in %.0fs -> %s", time.time() - t0, cache)
    return conn


def _coarse_population_index(ann, superclass, superclass_names) -> np.ndarray:
    sc = ann["superclass"].to_numpy()
    cls = ann["class"].to_numpy()
    idx = np.full(len(ann), COARSE_POPULATIONS.index("other"), dtype=np.int16)
    # superclass-based groups first, then class-based CX last so it is not
    # overwritten by its containing superclass (cb_intrinsic)
    ordered = [p for p in COARSE_POPULATIONS if p not in ("cx_intrinsic", "other")]
    ordered.append("cx_intrinsic")
    for name in ordered:
        if name == "cx_intrinsic":
            mask = cls == "CX"
        else:
            mask = sc == name
        idx[mask] = COARSE_POPULATIONS.index(name)
    return idx


def _hex_to_xy(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    return np.column_stack([h1 - 0.5 * h2, (np.sqrt(3.0) / 2.0) * h2])


def _build_retina(ann, ids, ptr, post, weight, hex1, hex2, side):
    """Map photoreceptors to viewport UVs via ommatidia columns.

    Photoreceptors carry no hex annotation in MaleCNS v1.0, so each receptor
    inherits the modal hex column of its annotated outgoing targets (lamina
    neurons), weighted by synapse count — the column-inheritance trick used by
    both DOOMFLY and FlyBrain (see PROVENANCE.md).
    """
    types = ann["type"].fillna("").to_numpy()
    is_pr = np.array([t.startswith(PHOTORECEPTOR_PREFIXES) for t in types])
    pr_idx = np.flatnonzero(is_pr)
    has_hex = ~(np.isnan(hex1) | np.isnan(hex2))

    chosen_h1 = np.full(len(ids), np.nan)
    chosen_h2 = np.full(len(ids), np.nan)
    conf = np.zeros(len(ids), dtype=np.float32)
    for i in pr_idx:
        e0, e1 = int(ptr[i]), int(ptr[i + 1])
        if e0 == e1:
            continue
        targets = post[e0:e1]
        w = np.abs(weight[e0:e1])
        valid = has_hex[targets]
        if not valid.any():
            continue
        votes: dict[tuple[float, float], float] = {}
        for t, wt in zip(targets[valid], w[valid]):
            h = (float(hex1[t]), float(hex2[t]))
            votes[h] = votes.get(h, 0.0) + float(wt)
        h = max(votes, key=votes.get)
        chosen_h1[i], chosen_h2[i] = h
        conf[i] = votes[h] / sum(votes.values())

    mapped = pr_idx[~np.isnan(chosen_h1[pr_idx])]
    xy = _hex_to_xy(chosen_h1[mapped], chosen_h2[mapped])
    uv = np.zeros((len(mapped), 2), dtype=np.float64)
    pr_side = side[mapped]
    for s, xlo in ((0, 0.0), (1, 0.4)):  # L eye -> [0,.6] mirrored; R -> [.4,1]
        m = pr_side == s
        if not m.any():
            continue
        lo = xy[m].min(axis=0)
        span = np.ptp(xy[m], axis=0)
        span[span == 0] = 1.0
        z = (xy[m] - lo) / span
        if s == 0:
            uv[m, 0] = 0.6 * z[:, 0]
        else:
            uv[m, 0] = 0.4 + 0.6 * (1.0 - z[:, 0])
        uv[m, 1] = 1.0 - z[:, 1]
    uv = np.clip(uv, 0.0, 1.0)

    t = types[mapped]
    channel = np.where(np.char.startswith(t.astype(str), "R8p"), 2,
                       np.where(np.char.startswith(t.astype(str), "R8y"), 1, 0))
    stats = dict(photoreceptors=int(len(pr_idx)), mapped=int(len(mapped)),
                 unmapped=int(len(pr_idx) - len(mapped)),
                 median_vote_confidence=float(np.median(conf[mapped])) if len(mapped) else 0.0)
    log.info("retina: %s", stats)
    return mapped, uv, channel.astype(np.int8), stats
