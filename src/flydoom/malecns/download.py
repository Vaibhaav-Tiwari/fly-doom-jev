"""MaleCNS data download + checksum verification against the pinned manifest.

    python -m flydoom.malecns.download --manifest data/manifests/malecns-v1.0.manifest.json
    python -m flydoom.malecns.download --manifest ... --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify(manifest_path: Path, data_dir: Path) -> bool:
    manifest = json.loads(manifest_path.read_text())
    ok = True
    for name, spec in manifest["files"].items():
        path = data_dir / spec["filename"]
        if not path.exists():
            status = "MISSING"
            ok = False if spec.get("required") else ok
        else:
            digest = sha256_file(path)
            expected = spec.get("sha256", "")
            if expected.startswith("PLACEHOLDER"):
                status = "UNPINNED (no checksum in manifest)"
            elif digest == expected:
                status = "OK"
            else:
                status = f"CHECKSUM MISMATCH (got {digest[:12]}...)"
                ok = False
        print(f"  {name:24s} {status}")
    return ok


def download(manifest_path: Path, data_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    base = manifest["base_url"]
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest["files"].items():
        path = data_dir / spec["filename"]
        if path.exists():
            continue
        print(f"downloading {name}: {spec['filename']} ...")
        with urllib.request.urlopen(f"{base}/{spec['filename']}") as r, \
                open(path.with_suffix(".partial"), "wb") as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
        path.with_suffix(".partial").replace(path)
        print(f"  {path} ({path.stat().st_size / 1e6:.0f} MB)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", default=None,
                   help="default: data/raw/<dataset>-<version> next to repo layout")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    data_dir = Path(args.data_dir) if args.data_dir else \
        Path(manifest.get("directory",
                          f"data/raw/{manifest['dataset']}-{manifest['version']}"))
    if not args.verify_only:
        download(manifest_path, data_dir)
    print(f"verifying against {manifest_path}:")
    ok = verify(manifest_path, data_dir)
    print("data verification: PASS" if ok else "data verification: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
