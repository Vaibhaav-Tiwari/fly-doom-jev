"""MaleCNS data access.

Default is the FULL MaleCNS v1.0 neuron-level connectome (211,577 neurons,
26.0M edges) via full.py. A reduced subset (graph.py) and a labeled
deterministic fixture graph remain available for dev/tests only.
"""

from .full import FullConnectome, load_full_connectome
from .graph import ReducedConnectome, fixture_connectome


def load_connectome(cfg: dict):
    """Dispatch on neural.mode: full (default) | reduced | fixture | auto."""
    from pathlib import Path
    from .graph import load_connectome as _load_reduced

    neural = cfg["neural"]
    mode = neural.get("mode", "full")
    if mode in ("reduced", "fixture"):
        return _load_reduced(cfg)
    if mode == "auto":
        import logging
        data_dir = Path(neural.get("data_dir", "data/raw/malecns-v1.0"))
        if (data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather").exists():
            return load_full_connectome(cfg)
        logging.getLogger(__name__).warning(
            "auto mode: MaleCNS raw data missing, using labeled fixture graph")
        return _load_reduced({**cfg, "neural": {**neural, "mode": "fixture"}})
    if mode == "full":
        return load_full_connectome(cfg)
    raise ValueError(f"unknown neural.mode {mode!r}")


__all__ = ["FullConnectome", "ReducedConnectome", "fixture_connectome",
           "load_connectome", "load_full_connectome"]
