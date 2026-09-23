"""Package a structural recording as a slim web bundle for the static site.

Live recordings carry the top-10,000 active neurons per step (~220 MB JSONL);
the site renders beautifully with far fewer. This trims activity.top to
--top (default 1500) entries per step and copies frames as-is.

Usage:
  python scripts/package_recording.py <recording_dir> <bundle_id> \
      [--title T] [--subtitle S] [--featured] [--top 1500]

Writes web/public/recordings/<bundle_id>/ and registers it in
web/public/recordings/index.json (idempotent per id).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "public" / "recordings"


def _recompress_frames(frames: Path, quality: int = 65) -> None:
    from PIL import Image
    import io
    for f in frames.glob("*.jpg"):
        im = Image.open(f)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) < f.stat().st_size:
            f.write_bytes(data)


def package(src: Path, bundle_id: str, title: str, subtitle: str,
            featured: bool, top: int) -> None:
    dest = WEB / bundle_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(src / "frames", dest / "frames")
    _recompress_frames(dest / "frames")

    with (src / "recording.jsonl").open() as fh, \
         (dest / "recording.jsonl").open("w") as out:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") == "step":
                act = r.get("activity")
                if isinstance(act, dict) and act.get("top"):
                    act = dict(act)
                    act["top"] = act["top"][:top]
                    r["activity"] = act
                ret = r.get("retina")
                if isinstance(ret, dict) and ret.get("indices"):
                    # top 300 photoreceptors by drive — visually identical layer
                    pairs = sorted(zip(ret["indices"], ret["drive"]),
                                   key=lambda p: -p[1])[:300]
                    r["retina"] = {**ret,
                                   "indices": [int(p[0]) for p in pairs],
                                   "drive": [round(float(p[1]), 4) for p in pairs]}
                pops = r.get("populations")
                if isinstance(pops, dict) and pops and isinstance(
                        next(iter(pops.values())), dict):
                    r["populations"] = {k: {"size": v.get("size"),
                                            "mean_rate_hz": round(float(v.get("mean_rate_hz", 0.0)), 3),
                                            "sampled": (v.get("sampled") or [])[:8]}
                                        for k, v in pops.items()}
            out.write(json.dumps(r) + "\n")

    idx_path = WEB / "index.json"
    idx = json.loads(idx_path.read_text())
    recs = [e for e in idx.get("recordings", []) if e["id"] != bundle_id]
    if featured:
        for e in recs:
            e["featured"] = False
    entry = {"id": bundle_id, "title": title, "subtitle": subtitle,
             "path": f"recordings/{bundle_id}/recording.jsonl",
             "assetsPath": "connectome-assets"}
    if featured:
        entry["featured"] = True
    recs.insert(0, entry)
    idx_path.write_text(json.dumps({"recordings": recs}))
    print(f"{bundle_id}: {(sum(f.stat().st_size for f in dest.rglob('*'))) // (1 << 20)} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("bundle_id")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", required=True)
    ap.add_argument("--featured", action="store_true")
    ap.add_argument("--top", type=int, default=1500)
    a = ap.parse_args()
    package(Path(a.src), a.bundle_id, a.title, a.subtitle, a.featured, a.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
