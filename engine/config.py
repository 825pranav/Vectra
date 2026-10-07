"""Collection configuration: YAML defaults + dotted-path overrides."""

# Imports: deepcopy so callers never share nested dicts, and PyYAML for config files.
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Repo root, and the YAML file holding every default index/collection setting.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "index" / "default.yaml"


# Read a YAML file into a dict (an empty file gives {}).
def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Recursively merge `over` on top of `base` and return a new dict; nested sections merge key by key.
def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# Apply dotted-path overrides (from CreateCollection options) to a copy of the config.
def apply_dotted(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply {"hnsw.M": "32"} style overrides. String values are parsed as YAML."""
    out = copy.deepcopy(cfg)
    # Walk each dotted path down to its leaf; unknown sections or keys raise KeyError (bad option).
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


# Default config with keyword overrides applied; used when building a new collection's config.
def default_config(**overrides: Any) -> dict[str, Any]:
    return apply_dotted(load_yaml(DEFAULT_CONFIG), overrides)
