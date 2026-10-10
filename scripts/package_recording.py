"""Package a structural recording as a slim web bundle for the static site.

Live recordings carry the top-10,000 active neurons per step (~220 MB JSONL);
the site renders beautifully with far fewer. This trims activity.top to
--top (default 1500) entries per step and copies frames as-is.

Usage:
  python scripts/package_recording.py <recording_dir> <bundle_id> \
      [--title T] [--subtitle S] [--featured] [--top 1500]

Writes web/public/recordings/<bundle_id>/ and registers it in
web/public/recordings/index.json (idempotent per id).  A bundle is not
considered complete unless every frame referenced by its recording is
present.  This is deliberately checked before removing an existing bundle,
so a typo in a curated source cannot leave a half-written showcase.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "public" / "recordings"


def _frame_refs(recording: Path) -> list[str]:
    """Return the frame paths referenced by step records in *recording*."""
    refs = []
    with recording.open() as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{recording}:{line_no}: invalid JSON") from exc
            if record.get("kind") == "step" and record.get("frame_ref"):
                refs.append(record["frame_ref"])
    return refs


def validate_recording(src: Path) -> list[str]:
    """Validate a source recording and return its referenced frame paths.

    Paths are kept relative to the recording directory.  Rejecting absolute
    and parent-traversing paths also prevents a malformed recording from
    making the packager inspect files outside its bundle.
    """
    recording = src / "recording.jsonl"
    frames = src / "frames"
    if not recording.is_file():
        raise FileNotFoundError(f"curated bundle is missing {recording}")
    if not frames.is_dir():
        raise FileNotFoundError(f"curated bundle is missing {frames}")

    missing = []
    refs = _frame_refs(recording)
    for ref in refs:
        path = Path(ref)
        if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("frames",):
            missing.append(f"{ref} (invalid frame_ref)")
            continue
        target = src.joinpath(*path.parts)
        if not target.is_file():
            missing.append(ref)
    if missing:
        raise FileNotFoundError(
            f"curated bundle {src} is incomplete; missing frame(s): "
            + ", ".join(missing)
        )
    return refs


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
    validate_recording(src)
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
    if featured:
        # Keep the showcase manifest derived from the catalog rather than
        # from directory enumeration.  The catalog is the curated source of
        # truth and write_featured_json sorts it before serializing.
        write_featured_json(recs)
    print(f"{bundle_id}: {(sum(f.stat().st_size for f in dest.rglob('*'))) // (1 << 20)} MB")


def write_featured_json(entries: list[dict], output: Path = WEB / "featured.json") -> None:
    """Write the static showcase manifest in deterministic order.

    Only the small, URL-facing fields are emitted.  Sorting by id makes the
    generated file independent of filesystem and manifest enumeration order.
    Every entry is checked against the packaged output before it is written.
    """
    normalized = []
    for entry in entries:
        bundle_id = entry["id"]
        bundle = output.parent / bundle_id
        validate_recording(bundle)
        normalized.append({
            "id": bundle_id,
            "title": entry["title"],
            "subtitle": entry["subtitle"],
            "path": f"recordings/{bundle_id}/recording.jsonl",
            "framesPath": f"recordings/{bundle_id}/frames",
        })
    normalized.sort(key=lambda item: item["id"])
    output.write_text(json.dumps({"format": 1, "recordings": normalized},
                                 indent=2, sort_keys=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("bundle_id")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", required=True)
    ap.add_argument("--featured", action="store_true")
    ap.add_argument("--top", type=int, default=1500)
    ap.add_argument("--featured-json", action="store_true",
                    help="also write featured.json for the packaged bundle")
    a = ap.parse_args()
    package(Path(a.src), a.bundle_id, a.title, a.subtitle, a.featured, a.top)
    if a.featured_json:
        write_featured_json([{"id": a.bundle_id, "title": a.title,
                              "subtitle": a.subtitle}])
    return 0


if __name__ == "__main__":
    sys.exit(main())
