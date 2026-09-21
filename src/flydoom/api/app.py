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


def create_app(cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config("configs/demo.yaml")
    recordings_root = cfg.get("recording", {}).get("directory", "outputs/recordings")

    app = FastAPI(title="jev-doom-fly replay", version="0.1.0")
    app.state.recordings_root = recordings_root

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

    @app.get("/api/recordings/{run_id}/frames")
    def frames(run_id: str) -> FileResponse:
        path = Path(app.state.recordings_root) / run_id / "frames.u8"
        if not path.exists():
            raise HTTPException(404, "no frames for this recording")
        return FileResponse(path, media_type="application/octet-stream")

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
