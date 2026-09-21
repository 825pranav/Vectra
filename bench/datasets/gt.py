"""Exact ground truth by brute force (BLAS, chunked), cached on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm


def brute_force_topk(base: np.ndarray, queries: np.ndarray, k: int, chunk: int = 128) -> np.ndarray:
    """Exact top-k by squared L2, via ||x||^2 - 2 x.q (||q||^2 is constant per row)."""
    base = np.ascontiguousarray(base, dtype=np.float32)
    norms = (base.astype(np.float64) ** 2).sum(1).astype(np.float32)
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in tqdm(range(0, len(queries), chunk), desc="ground truth", leave=False):
        q = np.ascontiguousarray(queries[s : s + chunk], dtype=np.float32)
        d = norms[None, :] - 2.0 * (q @ base.T)
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, 1), axis=1, kind="stable")
        out[s : s + chunk] = np.take_along_axis(part, order, 1)
    return out


def cached_topk(path: Path, base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    if path.exists():
        gt = np.load(path)
        if gt.shape == (len(queries), k):
            return gt
    gt = brute_force_topk(base, queries, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, gt)
    return gt
