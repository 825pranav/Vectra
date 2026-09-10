"""Collection configuration: YAML defaults + dotted-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "index" / "default.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_dotted(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply {"hnsw.M": "32"} style overrides. String values are parsed as YAML."""
    out = copy.deepcopy(cfg)
    for path, value in overrides.items():
        if isinstance(value, str):
            value = yaml.safe_load(value)
        node = out
        *parents, leaf = path.split(".")
        for p in parents:
            if not isinstance(node.get(p), dict):
                raise KeyError(f"unknown config section {path!r}")
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key {path!r}")
        node[leaf] = value
    return out


def default_config(**overrides: Any) -> dict[str, Any]:
    return apply_dotted(load_yaml(DEFAULT_CONFIG), overrides)
