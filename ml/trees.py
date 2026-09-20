"""LightGBM tree ensembles as flat arrays, evaluated inside Numba kernels.

Calling ``booster.predict`` from Python costs tens of microseconds per row,
which is the same order as an entire HNSW query, so a per-query (let alone
per-checkpoint) model call from Python would eat the latency it is meant to
save. Instead the trained booster is dumped once into five arrays and walked by
a ~10-line Numba function that runs *inside* the search loop:

    feature[i], threshold[i], left[i], right[i]   internal node i
    value[j]                                     leaf j (child ids < 0 encode ~j)
    roots[t]                                     root node of tree t (or ~leaf)

Only numerical ``<=`` splits are produced for our dense float features; the
flattener rejects anything else rather than silently mis-evaluating it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, nogil=True)
def _njit_predict(x, feature, threshold, left, right, value, roots):
    """Raw ensemble score for one feature vector (sum of tree outputs)."""
    s = 0.0
    for t in range(roots.shape[0]):
        node = roots[t]
        while node >= 0:
            if x[feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        s += value[~node]
    return s


def ref_predict(x: np.ndarray, forest: dict[str, np.ndarray]) -> float:
    """Pure-Python twin of ``_njit_predict``."""
    s = 0.0
    for root in forest["roots"]:
        node = int(root)
        while node >= 0:
            f = int(forest["feature"][node])
            node = int(forest["left"][node] if x[f] <= forest["threshold"][node]
                       else forest["right"][node])  # fmt: skip
        s += float(forest["value"][~node])
    return s


def flatten(dump: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten ``booster.dump_model()`` into the array form above."""
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []
    roots: list[int] = []

    def walk(node: dict[str, Any]) -> int:
        if "leaf_value" in node:
            value.append(float(node["leaf_value"]))
            return ~(len(value) - 1)
        if node.get("decision_type", "<=") != "<=":
            raise ValueError(f"unsupported split {node.get('decision_type')}")
        i = len(feature)
        feature.append(int(node["split_feature"]))
        threshold.append(float(node["threshold"]))
        left.append(0)
        right.append(0)
        left[i] = walk(node["left_child"])
        right[i] = walk(node["right_child"])
        return i

    for tree in dump["tree_info"]:
        roots.append(walk(tree["tree_structure"]))
    return {
        "feature": np.asarray(feature, dtype=np.int32),
        "threshold": np.asarray(threshold, dtype=np.float64),
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "value": np.asarray(value, dtype=np.float64),
        "roots": np.asarray(roots, dtype=np.int32),
    }


def save_forest(path: Path, forest: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    payload = {k: v.tolist() for k, v in forest.items()}
    payload["meta"] = meta
    Path(path).write_text(json.dumps(payload))


def load_forest(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    meta = raw.pop("meta")
    dtypes = {"threshold": np.float64, "value": np.float64}
    forest = {k: np.asarray(v, dtype=dtypes.get(k, np.int32)) for k, v in raw.items()}
    return forest, meta


def predict(x: np.ndarray, forest: dict[str, np.ndarray]) -> float:
    return float(
        _njit_predict(
            np.ascontiguousarray(x, dtype=np.float64),
            forest["feature"],
            forest["threshold"],
            forest["left"],
            forest["right"],
            forest["value"],
            forest["roots"],
        )
    )
