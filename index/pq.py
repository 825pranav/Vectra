"""Product quantization (Jégou et al., 2011).

A ``dim``-vector is cut into ``m`` subvectors of ``dsub = dim / m`` dims; each
subspace gets its own 256-entry k-means codebook, so a vector becomes ``m``
bytes (128-d float32 at m=16: 512 B -> 16 B).

Distances are asymmetric (ADC): the query stays in float32. Per query we build
an (m, 256) table of squared distances from each query subvector to each
centroid once; the distance to any code is then ``m`` table lookups summed.

Training is NumPy k-means (BLAS assignment) or, optionally, the same algorithm
in PyTorch on the GPU (``device="cuda"``) — the GPU is only ever used here,
offline, never at query time. Encoding, table building and scanning are Numba.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange

from index.flat import topk_from_dists

KSUB = 256


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _njit_encode(x, codebooks):
    """Nearest centroid per subspace. x: (n, dim), codebooks: (m, ksub, dsub)."""
    m, ksub, dsub = codebooks.shape
    n = x.shape[0]
    codes = np.empty((n, m), dtype=np.uint8)
    for i in prange(n):
        for j in range(m):
            best = np.inf
            arg = 0
            off = j * dsub
            for c in range(ksub):
                acc = np.float32(0.0)
                for t in range(dsub):
                    d = x[i, off + t] - codebooks[j, c, t]
                    acc += d * d
                if acc < best:
                    best = acc
                    arg = c
            codes[i, j] = arg
    return codes


@njit(cache=True, fastmath=True, nogil=True)
def _njit_lut(q, codebooks):
    """ADC table: lut[j, c] = ||q_j - codebooks[j, c]||^2."""
    m, ksub, dsub = codebooks.shape
    lut = np.empty((m, ksub), dtype=np.float32)
    for j in range(m):
        off = j * dsub
        for c in range(ksub):
            acc = np.float32(0.0)
            for t in range(dsub):
                d = q[off + t] - codebooks[j, c, t]
                acc += d * d
            lut[j, c] = acc
    return lut


@njit(cache=True, fastmath=True, nogil=True, inline="always")
def _njit_adc(lut, codes, i):
    acc = np.float32(0.0)
    for j in range(codes.shape[1]):
        acc += lut[j, codes[i, j]]
    return acc


@njit(cache=True, fastmath=True, nogil=True)
def _njit_adc_scan(codes, lut, n):
    """ADC distance from the query (via its table) to codes ``0..n-1``."""
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = _njit_adc(lut, codes, i)
    return out


# ---------------------------------------------------------------------------
# NumPy twins
# ---------------------------------------------------------------------------


def ref_encode(x: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    m, _, dsub = codebooks.shape
    out = np.empty((x.shape[0], m), dtype=np.uint8)
    for j in range(m):
        sub = x[:, j * dsub : (j + 1) * dsub].astype(np.float64)
        d = ((sub[:, None, :] - codebooks[j][None].astype(np.float64)) ** 2).sum(-1)
        out[:, j] = d.argmin(1)
    return out


def ref_lut(q: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    m, _, dsub = codebooks.shape
    sub = q.reshape(m, 1, dsub).astype(np.float64)
    return ((sub - codebooks.astype(np.float64)) ** 2).sum(-1).astype(np.float32)


def ref_adc_scan(codes: np.ndarray, lut: np.ndarray, n: int) -> np.ndarray:
    m = codes.shape[1]
    return lut[np.arange(m)[None, :], codes[:n].astype(np.int64)].sum(1).astype(np.float32)


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------


def _assign(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = (x * x).sum(1)[:, None] - 2.0 * (x @ c.T) + (c * c).sum(1)[None, :]
    return d.argmin(1)


def kmeans(x: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    """Lloyd's k-means; empty clusters are re-seeded from random points."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[0]
    cent = x[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        labels = _assign(x, cent)
        counts = np.bincount(labels, minlength=k)
        sums = np.stack(
            [np.bincount(labels, weights=x[:, t], minlength=k) for t in range(x.shape[1])], 1
        )
        live = counts > 0
        cent[live] = (sums[live] / counts[live, None]).astype(np.float32)
        if not live.all():
            cent[~live] = x[rng.choice(n, int((~live).sum()), replace=False)]
    return cent


def kmeans_torch(x: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    """Same algorithm on the GPU (offline training only)."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).cuda()
    n = xt.shape[0]
    cent = xt[torch.randperm(n, generator=g)[:k].cuda()].clone()
    for _ in range(iters):
        d = torch.cdist(xt, cent)
        labels = d.argmin(1)
        counts = torch.bincount(labels, minlength=k)
        sums = torch.zeros_like(cent).index_add_(0, labels, xt)
        live = counts > 0
        cent[live] = sums[live] / counts[live, None].float()
        dead = int((~live).sum())
        if dead:
            cent[~live] = xt[torch.randperm(n, generator=g)[:dead].cuda()]
    return cent.cpu().numpy()


# ---------------------------------------------------------------------------
# quantizer
# ---------------------------------------------------------------------------


class ProductQuantizer:
    def __init__(
        self,
        dim: int,
        m: int = 16,
        iters: int = 20,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if dim % m:
            raise ValueError(f"dim {dim} is not divisible by m={m}")
        self.dim, self.m, self.dsub = dim, m, dim // m
        self.iters, self.seed, self.device = iters, seed, device
        self.codebooks: np.ndarray | None = None  # (m, 256, dsub) float32

    @classmethod
    def from_config(cls, dim: int, cfg: dict[str, Any]) -> ProductQuantizer:
        p = cfg["pq"]
        return cls(dim, m=p["m"], iters=p["kmeans_iters"], seed=p["seed"], device=p["device"])

    @property
    def trained(self) -> bool:
        return self.codebooks is not None

    def train(self, x: np.ndarray) -> None:
        if x.shape[0] < KSUB:
            raise ValueError(f"need at least {KSUB} training vectors, got {x.shape[0]}")
        books = np.empty((self.m, KSUB, self.dsub), dtype=np.float32)
        for j in range(self.m):
            sub = x[:, j * self.dsub : (j + 1) * self.dsub]
            if self.device == "cuda":
                books[j] = kmeans_torch(sub, KSUB, self.iters, self.seed + j)
            else:
                books[j] = kmeans(sub, KSUB, self.iters, np.random.default_rng(self.seed + j))
        self.codebooks = books

    def encode(self, x: np.ndarray) -> np.ndarray:
        return _njit_encode(np.ascontiguousarray(x, dtype=np.float32), self.codebooks)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        parts = [self.codebooks[j][codes[:, j]] for j in range(self.m)]
        return np.concatenate(parts, axis=1)

    def lut(self, q: np.ndarray) -> np.ndarray:
        return _njit_lut(np.ascontiguousarray(q, dtype=np.float32), self.codebooks)

    def nbytes(self) -> int:
        return 0 if self.codebooks is None else int(self.codebooks.nbytes)


def search_pq_scan(
    codes: np.ndarray, n: int, lut: np.ndarray, k: int, deleted: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Standalone mode: ADC-scan every code, return the approximate top-k."""
    d = _njit_adc_scan(codes, lut, n)
    if deleted is not None:
        d[deleted[:n].view(np.bool_)] = np.inf
    pos = topk_from_dists(d, k)
    return pos.astype(np.int32), d[pos]
