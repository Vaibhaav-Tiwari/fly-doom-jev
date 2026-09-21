# jev-doom-fly v1 — closed-loop Jev(mock) -> reduced MaleCNS -> ViZDoom demo.
# Requires: uv (https://docs.astral.sh/uv/). No npm build needed for the v1 dashboard.

UV ?= uv
PY := .venv/bin/python

.PHONY: setup run-demo replay test

setup:
	$(UV) venv --python 3.14 .venv || $(UV) venv .venv
	$(UV) pip install --python $(PY) -e ".[dev]"

# Record a closed-loop episode -> outputs/recordings/<run_id>/
run-demo:
	$(PY) -m flydoom.experiments.runner --config configs/demo.yaml

# Verify MaleCNS raw files against the pinned sha256 manifest
verify-data:
	$(PY) -m flydoom.malecns.download --manifest data/manifests/malecns-v1.0.manifest.json --verify-only

download-data:
	$(PY) -m flydoom.malecns.download --manifest data/manifests/malecns-v1.0.manifest.json

# Serve the latest recording + replay dashboard at http://127.0.0.1:8420
replay:
	$(PY) -m flydoom.api.app

test:
	$(PY) -m pytest tests -q
