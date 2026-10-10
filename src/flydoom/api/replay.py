"""Recording loading for replay (no ViZDoom / no neural sim / no Jev needed)."""

from __future__ import annotations

import json
from pathlib import Path


def list_recordings(root: str | Path) -> list[dict]:
    root = Path(root)
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        rec = d / "recording.jsonl"
        if not d.is_dir() or not rec.exists():
            continue
        header = None
        with open(rec) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("kind") == "header":
                    header = obj
                    break
        env_info = header.get("environment", {}) if header else {}
        out.append({
            "run_id": d.name,
            "created_at": header.get("created_at") if header else None,
            "format_version": header.get("recording_format_version") if header else None,
            "environment_backend": env_info.get("backend"),
            "scenario": env_info.get("scenario"),
            "connectome_source": (header.get("connectome", {}).get("provenance", {})
                                  .get("source") if header else None),
            "connectome_reduced": (header.get("connectome", {}).get("provenance", {})
                                   .get("reduced") if header else None),
            "n_neurons": header.get("connectome", {}).get("n_neurons") if header else None,
            "jev": header.get("jev") if header else None,
            "warnings": header.get("warnings", []) if header else [],
        })
    return out


def load_recording(root: str | Path, run_id: str) -> dict:
    rec = Path(root) / run_id / "recording.jsonl"
    if not rec.exists():
        raise FileNotFoundError(run_id)
    header, steps, frames_meta, summary = None, [], None, None
    with open(rec) as f:
        for line_no, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"recording {run_id} has malformed JSON on line {line_no}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"recording {run_id} line {line_no} must be a JSON object"
                )
            kind = obj.pop("kind", None)
            if kind == "header":
                header = obj
            elif kind == "frames_meta":
                frames_meta = obj
            elif kind in ("summary", "episode_end"):
                summary = obj
            elif kind == "step":
                steps.append(obj)
    if header is None:
        raise ValueError(f"recording {run_id} has no header")
    if frames_meta is not None:
        if not frames_meta.get("pattern"):
            raise ValueError(f"recording {run_id} frames metadata has no pattern")
        frames_meta["url_pattern"] = f"/api/recordings/{run_id}/frames/" + "{:06d}.jpg"
    return {"run_id": run_id, "header": header, "steps": steps,
            "frames": frames_meta, "summary": summary}
