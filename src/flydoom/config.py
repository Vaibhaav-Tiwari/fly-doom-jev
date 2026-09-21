"""Minimal YAML config loader with `extends` and ${ENV_VAR:-default} substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


def _substitute(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) or "")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    return value


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if "extends" in raw:
        base = load_config(path.parent / raw["extends"])
        raw = _merge(base, raw)
    return _substitute(raw)
