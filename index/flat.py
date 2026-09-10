"""Exact brute-force search.

Used directly as the ``flat`` index type, by the planner's low-selectivity
strategy (scan only the ids that match a filter), and as ground truth in tests.
"""

from __future__ import annotations

import numpy as np

from index.distance import _njit_gather_l2sq, _njit_scan_l2sq


def topk_from_dists(dists: np.ndarray, k: int) -> np.ndarray:
    """Positions of the ``k`` smallest finite distances, sorted ascending."""
    if dists.shape[0] == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, dists.shape[0])
    if k < dists.shape[0]:
        part = np.argpartition(dists, k - 1)[:k]
    else:
        part = np.arange(dists.shape[0])
    order = part[np.argsort(dists[part], kind="stable")]
    return order[np.isfinite(dists[order])]


def search_flat(
    vecs: np.ndarray,
    n: int,
    q: np.ndarray,
    k: int,
    deleted: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k over rows ``0..n-1``, skipping tombstoned rows."""
    if n == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    dists = _njit_scan_l2sq(q, vecs, n)
    if deleted is not None:
        dists[deleted[:n].view(np.bool_)] = np.inf
    pos = topk_from_dists(dists, k)
    return pos.astype(np.int32), dists[pos]


def search_subset(
    vecs: np.ndarray, ids: np.ndarray, q: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k restricted to ``ids`` (already filtered and tombstone-free)."""
    if ids.shape[0] == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    dists = _njit_gather_l2sq(q, vecs, ids)
    pos = topk_from_dists(dists, k)
    return ids[pos], dists[pos]
