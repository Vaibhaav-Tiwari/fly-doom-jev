# jev-doom-fly — closed-loop Jev -> MaleCNS -> ViZDoom research system.
# Requires: uv (https://docs.astral.sh/uv/), Apple clang (kernel build).

UV ?= uv
PY := .venv/bin/python

.PHONY: setup run-demo replay live test verify-data download-data benchmark

setup:
	$(UV) venv --python 3.14 .venv || $(UV) venv .venv
	$(UV) pip install --python $(PY) -e ".[dev]"

# Record a closed-loop episode (full MaleCNS + ViZDoom) -> outputs/recordings/<id>/
run-demo:
	$(PY) -m flydoom.experiments.runner --config configs/demo.yaml

# Serve the replay API + dashboard at http://127.0.0.1:8420
replay:
	$(PY) -m flydoom.api.app

# Live mode: continuous ViZDoom + MaleCNS + Jev loop at http://127.0.0.1:8420
# (GET /state, POST /new, GET /health). Needs JEV_API_KEY/JEV_BASE_URL sourced
# (set -a; source ../.env; set +a); degrades to clearly-marked mock otherwise.
live:
	$(PY) -m flydoom.api.live

test:
	$(PY) -m pytest tests -q

# Checksum-verify the raw MaleCNS files against the pinned manifest
verify-data:
	$(PY) -m flydoom.malecns.download --manifest data/manifests/malecns-v1.0.manifest.json --verify-only

download-data:
	$(PY) -m flydoom.malecns.download --manifest data/manifests/malecns-v1.0.manifest.json

benchmark:
	$(PY) scripts/benchmark.py --config configs/demo.yaml
