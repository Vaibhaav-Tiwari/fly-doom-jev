"""FastAPI replay server (v1).

Serves recorded experiments and the replay dashboard. No Jev API, no ViZDoom,
no neural simulation involved — the browser reconstructs episodes purely from
recording files. There are no secrets on this server.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flydoom.api.replay import list_recordings, load_recording
from flydoom.config import load_config

log = logging.getLogger(__name__)

WEB_STATIC = Path(__file__).resolve().parents[3] / "web" / "static"

CONNECTOME_ASSET_NAMES = frozenset(
    {"meta.json", "positions.f32", "population.i16", "flags.u8", "ids.i64"})


def _ensure_connectome_assets(cfg: dict) -> Path | None:
    """Export the shared viz bundle if the full-graph cache exists but the
    assets don't. Returns the assets dir, or None when no full graph is
    available (replay of reduced/fixture recordings works without it)."""
    cache = Path(cfg.get("neural", {}).get("cache_full",
                                           "data/processed/malecns_full_v1"))
    out_dir = Path(cfg.get("recording", {}).get("connectome_assets_dir",
                                                "outputs/connectome_assets"))
    if (out_dir / "meta.json").exists():
        return out_dir
    if not (cache / "meta.json").exists():
        return None
    from flydoom.malecns.full import load_full_connectome
    from flydoom.malecns.viz_export import export_connectome_assets
    conn = load_full_connectome(cfg)
    export_connectome_assets(conn, out_dir)
    return out_dir


def create_app(cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config("configs/demo.yaml")
    recordings_root = cfg.get("recording", {}).get("directory", "outputs/recordings")

    app = FastAPI(title="jev-doom-fly replay", version="0.1.0")
    app.state.recordings_root = recordings_root
    app.state.connectome_assets = _ensure_connectome_assets(cfg)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "mode": "replay"}

    @app.get("/api/recordings")
    def recordings() -> list[dict]:
        return list_recordings(app.state.recordings_root)

    @app.get("/api/recordings/{run_id}")
    def recording(run_id: str) -> dict:
        try:
            return load_recording(app.state.recordings_root, run_id)
        except FileNotFoundError:
            raise HTTPException(404, f"no recording {run_id!r}")

    @app.get("/api/recordings/{run_id}/frames/{frame_name}")
    def frame(run_id: str, frame_name: str) -> FileResponse:
        # frames_meta.pattern is frames/{:06d}.jpg; only allow plain filenames
        if "/" in frame_name or ".." in frame_name:
            raise HTTPException(400, "bad frame name")
        path = Path(app.state.recordings_root) / run_id / "frames" / frame_name
        if not path.exists():
            raise HTTPException(404, "no such frame")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/connectome/{filename}")
    def connectome_asset(filename: str) -> FileResponse:
        """Shared connectome viz bundle (positions/ids/flags), loaded once by
        the frontend. Binary arrays keyed by neuron index; see meta.json."""
        if filename not in CONNECTOME_ASSET_NAMES:
            raise HTTPException(400, "bad asset name")
        assets = app.state.connectome_assets
        if assets is None:
            raise HTTPException(404, "no full connectome assets on this server")
        path = assets / filename
        if not path.exists():
            raise HTTPException(404, "asset not exported")
        media = ("application/json" if filename.endswith(".json")
                 else "application/octet-stream")
        return FileResponse(path, media_type=media)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_STATIC), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay server + dashboard")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    host = args.host or cfg.get("api", {}).get("host", "127.0.0.1")
    port = args.port or int(cfg.get("api", {}).get("port", 8420))
    app = create_app(cfg)

    import uvicorn
    print(f"replay dashboard: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
