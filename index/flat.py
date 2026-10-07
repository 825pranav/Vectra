"""Exact brute-force search.

Used directly as the ``flat`` index type, by the planner's low-selectivity
strategy (scan only the ids that match a filter), and as ground truth in tests.
"""

# Imports: the two distance kernels this exact search is built on.
from __future__ import annotations

import numpy as np

from index.distance import _njit_gather_l2sq, _njit_scan_l2sq


# Turns an array of distances into the positions of the k smallest, nearest first.
def topk_from_dists(dists: np.ndarray, k: int) -> np.ndarray:
    """Positions of the ``k`` smallest finite distances, sorted ascending."""
    # Nothing to rank: return an empty result.
    if dists.shape[0] == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, dists.shape[0])
    # argpartition finds the k smallest in linear time; then only those k get sorted.
    if k < dists.shape[0]:
        part = np.argpartition(dists, k - 1)[:k]
    else:
        part = np.arange(dists.shape[0])
    order = part[np.argsort(dists[part], kind="stable")]
    # Drop infinite distances (tombstoned rows) so they never show up as hits.
    return order[np.isfinite(dists[order])]


# Exact search over every stored row: the "flat" index type. Returns (row ids, squared distances).
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
    # Score every row, then push tombstoned rows to infinity so top-k skips them.
    dists = _njit_scan_l2sq(q, vecs, n)
    if deleted is not None:
        dists[deleted[:n].view(np.bool_)] = np.inf
    pos = topk_from_dists(dists, k)
    return pos.astype(np.int32), dists[pos]


# Exact search over only the given row ids; the planner's brute_force path for rare filters.
def search_subset(
    vecs: np.ndarray, ids: np.ndarray, q: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k restricted to ``ids`` (already filtered and tombstone-free)."""
    if ids.shape[0] == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    # Numba kernel wants a contiguous int32 array of ids.
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    dists = _njit_gather_l2sq(q, vecs, ids)
    # pos indexes into ids, so map it back to real row ids for the reply.
    pos = topk_from_dists(dists, k)
    return ids[pos], dists[pos]
