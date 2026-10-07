"""Distance kernels shared by every index.

All distances are squared L2 in float32. Cosine collections normalise vectors on
the way in, so squared L2 on unit vectors ranks identically to cosine similarity
(||a - b||^2 = 2 - 2 cos(a, b)).

Each ``_njit_`` kernel has a pure-NumPy twin (``ref_*``) used by the tests.
"""

# Imports: NumPy for arrays, Numba njit to compile the hot distance loops to machine code.
from __future__ import annotations

import numpy as np
from numba import njit


# Squared L2 between two vectors, inlined into every other kernel; the innermost loop of search.
@njit(cache=True, fastmath=True, nogil=True, inline="always")
def _njit_l2sq(a: np.ndarray, b: np.ndarray) -> np.float32:
    acc = np.float32(0.0)
    for j in range(a.shape[0]):
        t = a[j] - b[j]
        acc += t * t
    return acc


# Query-path kernels are deliberately serial: concurrency comes from serving many
# queries at once (the gRPC pool), not from nesting OpenMP regions inside one.


# Distances from the query to the first n stored rows; used by the flat (exact) scan.
@njit(cache=True, fastmath=True, nogil=True)
def _njit_scan_l2sq(q: np.ndarray, vecs: np.ndarray, n: int) -> np.ndarray:
    """Distances from ``q`` to rows ``0..n-1`` of ``vecs``."""
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = _njit_l2sq(q, vecs[i])
    return out


# Distances from the query to only the listed row ids; used by brute-force filtered search.
@njit(cache=True, fastmath=True, nogil=True)
def _njit_gather_l2sq(q: np.ndarray, vecs: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Distances from ``q`` to the rows listed in ``ids``."""
    out = np.empty(ids.shape[0], dtype=np.float32)
    for i in range(ids.shape[0]):
        out[i] = _njit_l2sq(q, vecs[ids[i]])
    return out


# Plain NumPy version of the scan kernel, kept so tests can check the Numba result.
def ref_scan_l2sq(q: np.ndarray, vecs: np.ndarray, n: int) -> np.ndarray:
    diff = vecs[:n].astype(np.float64) - q.astype(np.float64)
    return np.einsum("ij,ij->i", diff, diff).astype(np.float32)


# Plain NumPy version of the gather kernel, also only for tests.
def ref_gather_l2sq(q: np.ndarray, vecs: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return ref_scan_l2sq(q, vecs[ids], len(ids))


# Scale each vector to length 1 so cosine collections can reuse squared L2 for ranking.
def normalize(x: np.ndarray) -> np.ndarray:
    """Row-normalise to unit length (zero rows stay zero)."""
    x = np.asarray(x, dtype=np.float32)
    # Avoid divide-by-zero: an all-zero row keeps norm 1 and stays zero.
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)
