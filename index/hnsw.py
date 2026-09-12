"""HNSW (Malkov & Yashunin, 2016) over flat NumPy arrays, compiled with Numba.

Graph state
-----------
* ``nbr0``      (cap, 2M) int32  layer-0 neighbour lists, -1 = empty slot
* ``upper``     (ucap, M) int32  neighbour lists for every (node, layer >= 1)
* ``upper_row`` (cap,)    int32  first row of a node in ``upper`` (-1 if level 0);
                                 node ``x`` on layer ``l`` lives at row
                                 ``upper_row[x] + l - 1``
* ``levels``    (cap,)    int8   top layer of each node
* ``entry``, ``max_level``       entry point and its level

Layer 0 is an (N, 2M) table; the upper layers share one compact (rows, M) table
instead of one (N, M) table per layer: only ~N/M nodes exist above layer 0, so
per-layer full tables would cost ~5 x N x M x 4 bytes of -1s at 1M points. It is
the same "preallocated int32, -1 = empty" layout, indexed through ``upper_row``.

Build
-----
Node levels are derived from a hash of the node id (not from RNG state), so a
rebuild or WAL replay reproduces the same graph. Insertion runs in batches:

1. *Search phase* (``prange`` over the batch): every new node searches the
   graph as it was before the batch, picks neighbours with Malkov's heuristic
   and writes its own rows. Nothing else is written, so there are no races.
2. *Link phase* (``prange`` over target nodes): reverse edges are grouped by
   (layer, target); each target merges its new in-edges and re-prunes with the
   heuristic if it overflows. Every row has exactly one writer.

Batch members cannot see each other, so batches are kept to a small fraction of
the current graph (``batch_fraction``) after a serial warm-up. Because neither
phase depends on thread scheduling, the build is deterministic for any thread
count.

Concurrency
-----------
All arrays live in one ``HNSWGraph`` object. Growth copies every array into a
new object and swaps ``self.g``; old objects are never written again. Readers
capture ``g`` once (before capturing the vector array, see engine.collection),
so every id they can reach is in bounds for every array they hold.
"""

from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from numba import njit, prange

from index.distance import _njit_l2sq

EMPTY = -1
_MAX_THREADS = numba.config.NUMBA_NUM_THREADS

# ---------------------------------------------------------------------------
# heaps on parallel (dist, id) arrays
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True, inline="always")
def _njit_minheap_push(hd, hi, size, d, i):
    j = size
    while j > 0:
        p = (j - 1) >> 1
        if hd[p] <= d:
            break
        hd[j] = hd[p]
        hi[j] = hi[p]
        j = p
    hd[j] = d
    hi[j] = i
    return size + 1


@njit(cache=True, nogil=True, inline="always")
def _njit_minheap_pop(hd, hi, size):
    """Drop the root (read ``hd[0]``/``hi[0]`` first). Returns the new size."""
    size -= 1
    if size <= 0:
        return 0
    d = hd[size]
    i = hi[size]
    j = 0
    while True:
        c = 2 * j + 1
        if c >= size:
            break
        if c + 1 < size and hd[c + 1] < hd[c]:
            c += 1
        if hd[c] >= d:
            break
        hd[j] = hd[c]
        hi[j] = hi[c]
        j = c
    hd[j] = d
    hi[j] = i
    return size


@njit(cache=True, nogil=True, inline="always")
def _njit_maxheap_push(hd, hi, size, d, i):
    j = size
    while j > 0:
        p = (j - 1) >> 1
        if hd[p] >= d:
            break
        hd[j] = hd[p]
        hi[j] = hi[p]
        j = p
    hd[j] = d
    hi[j] = i
    return size + 1


@njit(cache=True, nogil=True, inline="always")
def _njit_maxheap_pop(hd, hi, size):
    size -= 1
    if size <= 0:
        return 0
    d = hd[size]
    i = hi[size]
    j = 0
    while True:
        c = 2 * j + 1
        if c >= size:
            break
        if c + 1 < size and hd[c + 1] > hd[c]:
            c += 1
        if hd[c] <= d:
            break
        hd[j] = hd[c]
        hi[j] = hi[c]
        j = c
    hd[j] = d
    hi[j] = i
    return size


@njit(cache=True, nogil=True)
def _njit_grow(hd, hi):
    nd = np.empty(hd.shape[0] * 2, dtype=hd.dtype)
    ni = np.empty(hi.shape[0] * 2, dtype=hi.dtype)
    nd[: hd.shape[0]] = hd
    ni[: hi.shape[0]] = hi
    return nd, ni


@njit(cache=True, nogil=True)
def _njit_drain_maxheap(wd, wi, nw):
    """Empty a max-heap into ascending (ids, dists) arrays."""
    out_i = np.empty(nw, dtype=np.int32)
    out_d = np.empty(nw, dtype=np.float32)
    for t in range(nw - 1, -1, -1):
        out_d[t] = wd[0]
        out_i[t] = wi[0]
        nw = _njit_maxheap_pop(wd, wi, nw)
    return out_i, out_d


# ---------------------------------------------------------------------------
# graph kernels
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, nogil=True)
def _njit_greedy(q, vecs, g, rowbase, offset, cur, dcur):
    """Greedy walk on one layer (ef = 1). ``offset < 0`` means row = node id."""
    ndist = 0
    changed = True
    while changed:
        changed = False
        row = cur if offset < 0 else rowbase[cur] + offset
        for j in range(g.shape[1]):
            e = g[row, j]
            if e < 0:
                break
            d = _njit_l2sq(q, vecs[e])
            ndist += 1
            if d < dcur:
                dcur = d
                cur = e
                changed = True
    return cur, dcur, ndist


@njit(cache=True, fastmath=True, nogil=True)
def _njit_search_layer(q, vecs, g, rowbase, offset, ep_i, ep_d, ef, visited, tag):
    """Best-first search on one layer from several entry points (construction)."""
    cap = max(64, 4 * ef)
    cd = np.empty(cap, dtype=np.float32)
    ci = np.empty(cap, dtype=np.int32)
    wd = np.empty(ef + 1, dtype=np.float32)
    wi = np.empty(ef + 1, dtype=np.int32)
    nc = 0
    nw = 0
    for a in range(ep_i.shape[0]):
        e = ep_i[a]
        if visited[e] == tag:
            continue
        visited[e] = tag
        if nc == cd.shape[0]:
            cd, ci = _njit_grow(cd, ci)
        nc = _njit_minheap_push(cd, ci, nc, ep_d[a], e)
        nw = _njit_maxheap_push(wd, wi, nw, ep_d[a], e)
        if nw > ef:
            nw = _njit_maxheap_pop(wd, wi, nw)
    width = g.shape[1]
    while nc > 0:
        dc = cd[0]
        c = ci[0]
        if nw >= ef and dc > wd[0]:
            break
        nc = _njit_minheap_pop(cd, ci, nc)
        row = c if offset < 0 else rowbase[c] + offset
        for j in range(width):
            e = g[row, j]
            if e < 0:
                break
            if visited[e] == tag:
                continue
            visited[e] = tag
            de = _njit_l2sq(q, vecs[e])
            if nw < ef or de < wd[0]:
                if nc == cd.shape[0]:
                    cd, ci = _njit_grow(cd, ci)
                nc = _njit_minheap_push(cd, ci, nc, de, e)
                nw = _njit_maxheap_push(wd, wi, nw, de, e)
                if nw > ef:
                    nw = _njit_maxheap_pop(wd, wi, nw)
    return _njit_drain_maxheap(wd, wi, nw)


@njit(cache=True, fastmath=True, nogil=True)
def _njit_select(vecs, cand_i, cand_d, n_cand, m, out):
    """Malkov's neighbour-selection heuristic (Alg. 4, no candidate extension).

    ``cand_*`` must be sorted by distance to the base point. A candidate is kept
    only if it is closer to the base than to every already-kept neighbour, which
    spreads links across directions instead of piling them into one cluster.
    Like hnswlib, fewer than ``m`` candidates are all kept.
    """
    if n_cand < m:
        for a in range(n_cand):
            out[a] = cand_i[a]
        return n_cand
    cnt = 0
    for a in range(n_cand):
        if cnt >= m:
            break
        e = cand_i[a]
        de = cand_d[a]
        ve = vecs[e]
        good = True
        for b in range(cnt):
            if _njit_l2sq(ve, vecs[out[b]]) < de:
                good = False
                break
        if good:
            out[cnt] = e
            cnt += 1
    return cnt


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _njit_search_batch_phase(
    batch, vecs, levels, nbr0, upper, upper_row, entry, max_level, m, efc, visited, tags
):
    """Phase 1 of a batch insert: each new node finds and writes its own links."""
    for b in prange(batch.shape[0]):
        tid = numba.get_thread_id()
        vis = visited[tid]
        x = batch[b]
        q = vecs[x]
        lx = levels[x]
        cur = entry
        dcur = _njit_l2sq(q, vecs[cur])
        for l in range(max_level, lx, -1):
            cur, dcur, _ = _njit_greedy(q, vecs, upper, upper_row, l - 1, cur, dcur)
        ep_i = np.empty(1, dtype=np.int32)
        ep_d = np.empty(1, dtype=np.float32)
        ep_i[0] = cur
        ep_d[0] = dcur
        out = np.empty(m, dtype=np.int32)
        for l in range(min(lx, max_level), -1, -1):
            tags[tid] += 1
            if l == 0:
                w_i, w_d = _njit_search_layer(
                    q, vecs, nbr0, upper_row, -1, ep_i, ep_d, efc, vis, tags[tid]
                )
                cnt = _njit_select(vecs, w_i, w_d, w_i.shape[0], m, out)
                for j in range(cnt):
                    nbr0[x, j] = out[j]
            else:
                w_i, w_d = _njit_search_layer(
                    q, vecs, upper, upper_row, l - 1, ep_i, ep_d, efc, vis, tags[tid]
                )
                cnt = _njit_select(vecs, w_i, w_d, w_i.shape[0], m, out)
                row = upper_row[x] + l - 1
                for j in range(cnt):
                    upper[row, j] = out[j]
            ep_i = w_i
            ep_d = w_d


@njit(cache=True, nogil=True)
def _njit_collect_reverse(batch, levels, max_level, nbr0, upper, upper_row):
    """List (layer, target, source) for every edge written in the search phase."""
    total = 0
    for b in range(batch.shape[0]):
        x = batch[b]
        for l in range(min(levels[x], max_level) + 1):
            g = nbr0 if l == 0 else upper
            row = x if l == 0 else upper_row[x] + l - 1
            for j in range(g.shape[1]):
                if g[row, j] < 0:
                    break
                total += 1
    lay = np.empty(total, dtype=np.int32)
    tgt = np.empty(total, dtype=np.int32)
    src = np.empty(total, dtype=np.int32)
    t = 0
    for b in range(batch.shape[0]):
        x = batch[b]
        for l in range(min(levels[x], max_level) + 1):
            g = nbr0 if l == 0 else upper
            row = x if l == 0 else upper_row[x] + l - 1
            for j in range(g.shape[1]):
                e = g[row, j]
                if e < 0:
                    break
                lay[t] = l
                tgt[t] = e
                src[t] = x
                t += 1
    return lay, tgt, src


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _njit_link_phase(seg, lay, tgt, src, vecs, nbr0, upper, upper_row):
    """Phase 2: add reverse edges; one thread owns each (layer, target) row."""
    for s in prange(seg.shape[0] - 1):
        a = seg[s]
        b = seg[s + 1]
        t = tgt[a]
        l = lay[a]
        g = nbr0 if l == 0 else upper
        row = t if l == 0 else upper_row[t] + l - 1
        mmax = g.shape[1]
        cnt = 0
        while cnt < mmax and g[row, cnt] >= 0:
            cnt += 1
        total = cnt + (b - a)
        if total <= mmax:
            for j in range(a, b):
                g[row, cnt] = src[j]
                cnt += 1
            continue
        vt = vecs[t]
        ci = np.empty(total, dtype=np.int32)
        cd = np.empty(total, dtype=np.float32)
        for j in range(cnt):
            ci[j] = g[row, j]
            cd[j] = _njit_l2sq(vt, vecs[ci[j]])
        for j in range(a, b):
            ci[cnt + j - a] = src[j]
            cd[cnt + j - a] = _njit_l2sq(vt, vecs[src[j]])
        order = np.argsort(cd, kind="mergesort")
        si = ci[order]
        sd = cd[order]
        out = np.empty(mmax, dtype=np.int32)
        kept = _njit_select(vecs, si, sd, total, mmax, out)
        for j in range(kept):
            g[row, j] = out[j]
        for j in range(kept, mmax):
            g[row, j] = EMPTY


# ---------------------------------------------------------------------------
# query kernels
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, nogil=True)
def _njit_descend(q, vecs, upper, upper_row, entry, max_level):
    """Greedy descent through the upper layers to a layer-0 entry point."""
    cur = entry
    dcur = _njit_l2sq(q, vecs[cur])
    nd = 1
    for l in range(max_level, 0, -1):
        cur, dcur, k = _njit_greedy(q, vecs, upper, upper_row, l - 1, cur, dcur)
        nd += k
    return cur, dcur, nd


@njit(cache=True, fastmath=True, nogil=True)
def _njit_search_l0(q, vecs, nbr0, ep, dep, ef, deleted, mask, use_mask, visited, tag, max_hops):
    """Layer-0 best-first search with tombstones and an optional filter bitmap.

    Traversal walks through every node; only *accepted* nodes (alive and, if
    ``use_mask``, allowed by the filter) enter the result set. While fewer than
    ``ef`` accepted nodes are known, every neighbour stays a candidate, which is
    what makes very selective filters expensive (the planner avoids that case).
    ``max_hops >= 0`` caps the number of node expansions.
    """
    cap = max(64, 4 * ef)
    cd = np.empty(cap, dtype=np.float32)
    ci = np.empty(cap, dtype=np.int32)
    wd = np.empty(ef + 1, dtype=np.float32)
    wi = np.empty(ef + 1, dtype=np.int32)
    visited[ep] = tag
    nc = _njit_minheap_push(cd, ci, 0, dep, ep)
    nw = 0
    if deleted[ep] == 0 and (not use_mask or mask[ep]):
        nw = _njit_maxheap_push(wd, wi, 0, dep, ep)
    hops = 0
    ndist = 0
    width = nbr0.shape[1]
    while nc > 0:
        dc = cd[0]
        c = ci[0]
        if nw >= ef and dc > wd[0]:
            break
        if max_hops >= 0 and hops >= max_hops:
            break
        nc = _njit_minheap_pop(cd, ci, nc)
        hops += 1
        for j in range(width):
            e = nbr0[c, j]
            if e < 0:
                break
            if visited[e] == tag:
                continue
            visited[e] = tag
            de = _njit_l2sq(q, vecs[e])
            ndist += 1
            if nw < ef or de < wd[0]:
                if nc == cd.shape[0]:
                    cd, ci = _njit_grow(cd, ci)
                nc = _njit_minheap_push(cd, ci, nc, de, e)
                if deleted[e] == 0 and (not use_mask or mask[e]):
                    nw = _njit_maxheap_push(wd, wi, nw, de, e)
                    if nw > ef:
                        nw = _njit_maxheap_pop(wd, wi, nw)
    out_i, out_d = _njit_drain_maxheap(wd, wi, nw)
    return out_i, out_d, hops, ndist


@njit(cache=True, fastmath=True, nogil=True)
def _njit_query(
    q, vecs, nbr0, upper, upper_row, entry, max_level, k, ef,
    deleted, mask, use_mask, visited, tag, max_hops,
):  # fmt: skip
    ep, dep, nd0 = _njit_descend(q, vecs, upper, upper_row, entry, max_level)
    ids, ds, hops, nd = _njit_search_l0(
        q, vecs, nbr0, ep, dep, max(ef, k), deleted, mask, use_mask, visited, tag, max_hops
    )
    kk = min(k, ids.shape[0])
    return ids[:kk], ds[:kk], hops, nd + nd0


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _njit_query_batch(
    Q, vecs, nbr0, upper, upper_row, entry, max_level, k, ef, deleted, visited, tag_base
):
    nq = Q.shape[0]
    out_i = np.full((nq, k), -1, dtype=np.int32)
    out_d = np.full((nq, k), np.inf, dtype=np.float32)
    ndist = np.zeros(nq, dtype=np.int64)
    dummy = np.zeros(1, dtype=np.bool_)
    for i in prange(nq):
        tid = numba.get_thread_id()
        tag = np.uint32(tag_base + i + 1)
        ids, ds, _, nd = _njit_query(
            Q[i], vecs, nbr0, upper, upper_row, entry, max_level, k, ef,
            deleted, dummy, False, visited[tid], tag, -1,
        )  # fmt: skip
        ndist[i] = nd
        for j in range(ids.shape[0]):
            out_i[i, j] = ids[j]
            out_d[i, j] = ds[j]
    return out_i, out_d, ndist


# ---------------------------------------------------------------------------
# Python-side index
# ---------------------------------------------------------------------------


def node_levels(ids: np.ndarray, seed: int, ml: float, max_level: int) -> np.ndarray:
    """Level of each node: floor(-ln(U) * mL) with U from a splitmix64 hash of the id."""
    mix = np.uint64((seed * 0x9E3779B97F4A7C15) % (1 << 64))
    z = ids.astype(np.uint64) + mix
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)
    lv = np.floor(-np.log(u) * ml)
    return np.minimum(lv, max_level).astype(np.int8)


@dataclass
class HNSWGraph:
    nbr0: np.ndarray
    upper: np.ndarray
    upper_row: np.ndarray
    levels: np.ndarray
    entry: int = -1
    max_level: int = -1

    @property
    def capacity(self) -> int:
        return self.nbr0.shape[0]


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        max_level: int = 12,
        seed: int = 42,
        serial_warmup: int = 1024,
        batch_fraction: float = 0.02,
        max_batch: int = 8192,
        capacity: int = 1024,
    ) -> None:
        self.dim = dim
        self.M = int(M)
        self.M0 = 2 * self.M
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.level_cap = int(max_level)
        self.seed = int(seed)
        self.ml = 1.0 / math.log(self.M)
        self.serial_warmup = int(serial_warmup)
        self.batch_fraction = float(batch_fraction)
        self.max_batch = int(max_batch)
        self.n = 0
        self.n_upper = 0
        self.g = self._alloc(max(int(capacity), 1), self._upper_rows_for(max(int(capacity), 1)))
        self._build_visited: np.ndarray | None = None
        self._build_tags = np.zeros(_MAX_THREADS, dtype=np.uint32)
        self._tls = threading.local()
        self._batch_visited: np.ndarray | None = None
        self._batch_tag = 0

    @classmethod
    def from_config(cls, dim: int, cfg: dict[str, Any], capacity: int = 1024) -> HNSWIndex:
        h = cfg["hnsw"]
        return cls(
            dim,
            M=h["M"],
            ef_construction=h["ef_construction"],
            ef_search=h["ef_search"],
            max_level=h["max_level"],
            seed=h["seed"],
            serial_warmup=h["serial_warmup"],
            batch_fraction=h["batch_fraction"],
            max_batch=h["max_batch"],
            capacity=capacity,
        )

    # ---- storage ------------------------------------------------------------

    def _upper_rows_for(self, cap: int) -> int:
        # Expected upper rows ~ n / (M - 1); leave generous headroom.
        return max(64, cap // max(self.M // 4, 1))

    def _alloc(self, cap: int, ucap: int) -> HNSWGraph:
        return HNSWGraph(
            nbr0=np.full((cap, self.M0), EMPTY, dtype=np.int32),
            upper=np.full((ucap, self.M), EMPTY, dtype=np.int32),
            upper_row=np.full(cap, EMPTY, dtype=np.int32),
            levels=np.zeros(cap, dtype=np.int8),
        )

    def ensure_capacity(self, rows: int, upper_rows: int = 0) -> None:
        """Grow by doubling. Copies every array into a new graph object and swaps it
        in; the old object is never written again, so readers holding it are safe."""
        g = self.g
        if rows <= g.capacity and upper_rows <= g.upper.shape[0]:
            return
        cap = g.capacity
        while cap < rows:
            cap *= 2
        ucap = max(g.upper.shape[0], self._upper_rows_for(cap))
        while ucap < upper_rows:
            ucap *= 2
        new = self._alloc(cap, ucap)
        new.nbr0[: g.capacity] = g.nbr0
        new.upper[: g.upper.shape[0]] = g.upper
        new.upper_row[: g.capacity] = g.upper_row
        new.levels[: g.capacity] = g.levels
        new.entry, new.max_level = g.entry, g.max_level
        self.g = new
        self._build_visited = None

    def nbytes(self) -> int:
        g = self.g
        return int(
            g.nbr0[: self.n].nbytes
            + g.upper[: self.n_upper].nbytes
            + g.upper_row[: self.n].nbytes
            + g.levels[: self.n].nbytes
        )

    # ---- build ------------------------------------------------------------

    def add(self, vecs: np.ndarray, end: int) -> None:
        """Insert nodes ``self.n .. end-1``; their vectors must already be in ``vecs``."""
        start = self.n
        if end <= start:
            return
        ids = np.arange(start, end, dtype=np.int64)
        lv = node_levels(ids, self.seed, self.ml, self.level_cap)
        self.ensure_capacity(end, self.n_upper + int(lv.sum()))
        g = self.g
        rows = self.n_upper + np.cumsum(lv, dtype=np.int64) - lv
        up = lv > 0
        g.upper_row[ids[up]] = rows[up]
        g.levels[start:end] = lv
        self.n_upper += int(lv.sum())
        if self._build_visited is None or self._build_visited.shape[1] < g.capacity:
            self._build_visited = np.zeros((_MAX_THREADS, g.capacity), dtype=np.uint32)
            self._build_tags[:] = 0

        pos = start
        if g.entry < 0:
            g.entry, g.max_level = start, int(lv[0])
            pos += 1
            self.n = pos
        while pos < end:
            if pos < self.serial_warmup:
                size = 1
            else:
                size = min(self.max_batch, max(1, int(pos * self.batch_fraction)))
            stop = min(pos + size, end)
            # A node that raises the top level is inserted alone so the new top
            # layer starts from a properly linked entry point.
            over = np.flatnonzero(g.levels[pos:stop] > g.max_level)
            if over.size:
                stop = pos + max(int(over[0]), 1)
            self._insert_batch(vecs, np.arange(pos, stop, dtype=np.int32))
            top = int(g.levels[pos:stop].max())
            if top > g.max_level:
                g.entry = pos + int(np.argmax(g.levels[pos:stop]))
                g.max_level = top
            pos = stop
            self.n = pos

    def _insert_batch(self, vecs: np.ndarray, batch: np.ndarray) -> None:
        g = self.g
        _njit_search_batch_phase(
            batch, vecs, g.levels, g.nbr0, g.upper, g.upper_row, g.entry, g.max_level,
            self.M, self.ef_construction, self._build_visited, self._build_tags,
        )  # fmt: skip
        lay, tgt, src = _njit_collect_reverse(
            batch, g.levels, g.max_level, g.nbr0, g.upper, g.upper_row
        )
        if lay.size == 0:
            return
        key = lay.astype(np.int64) * g.capacity + tgt
        order = np.argsort(key, kind="stable")
        key = key[order]
        seg = np.flatnonzero(np.diff(key)) + 1
        seg = np.concatenate(([0], seg, [key.size])).astype(np.int64)
        _njit_link_phase(
            seg, lay[order], tgt[order], src[order], vecs, g.nbr0, g.upper, g.upper_row
        )

    # ---- search -----------------------------------------------------------

    def _visited(self, cap: int) -> tuple[np.ndarray, np.uint32]:
        tls = self._tls
        v = getattr(tls, "visited", None)
        if v is None or v.shape[0] < cap:
            v = tls.visited = np.zeros(cap, dtype=np.uint32)
            tls.tag = 0
        tls.tag += 1
        if tls.tag >= 0xFFFFFFFF:
            v[:] = 0
            tls.tag = 1
        return v, np.uint32(tls.tag)

    def search(
        self,
        vecs: np.ndarray,
        q: np.ndarray,
        k: int,
        ef: int | None = None,
        deleted: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        max_hops: int = -1,
        g: HNSWGraph | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """Top-k for one query. ``mask`` (bool, len >= n) restricts accepted ids."""
        g = g or self.g
        if g.entry < 0:
            return np.empty(0, np.int32), np.empty(0, np.float32), {"hops": 0, "ndist": 0}
        if deleted is None:
            deleted = np.zeros(g.capacity, dtype=np.uint8)
        use_mask = mask is not None
        if mask is None:
            mask = np.zeros(1, dtype=np.bool_)
        visited, tag = self._visited(g.capacity)
        ids, ds, hops, nd = _njit_query(
            q, vecs, g.nbr0, g.upper, g.upper_row, g.entry, g.max_level, k,
            int(ef or self.ef_search), deleted, mask, use_mask, visited, tag, max_hops,
        )  # fmt: skip
        return ids, ds, {"hops": int(hops), "ndist": int(nd)}

    def search_batch(
        self,
        vecs: np.ndarray,
        Q: np.ndarray,
        k: int,
        ef: int | None = None,
        deleted: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Many queries in parallel (benchmarks). Returns (ids, dists, ndist)."""
        g = self.g
        if deleted is None:
            deleted = np.zeros(g.capacity, dtype=np.uint8)
        visited, base = self._batch_state(g.capacity, Q.shape[0])
        return _njit_query_batch(
            np.ascontiguousarray(Q, dtype=np.float32), vecs, g.nbr0, g.upper, g.upper_row,
            g.entry, g.max_level, k, int(ef or self.ef_search), deleted, visited, base,
        )  # fmt: skip

    def _batch_state(self, cap: int, nq: int) -> tuple[np.ndarray, int]:
        """Per-thread visited rows for batch search; tags stay unique across calls."""
        if self._batch_visited is None or self._batch_visited.shape[1] < cap:
            self._batch_visited = np.zeros((_MAX_THREADS, cap), dtype=np.uint32)
            self._batch_tag = 0
        if self._batch_tag + nq + 1 >= 0xFFFFFFFF:
            self._batch_visited[:] = 0
            self._batch_tag = 0
        base = self._batch_tag
        self._batch_tag += nq
        return self._batch_visited, base

    # ---- persistence ------------------------------------------------------

    def state(self) -> dict[str, np.ndarray]:
        g = self.g
        return {
            "nbr0": g.nbr0[: self.n],
            "upper": g.upper[: self.n_upper],
            "upper_row": g.upper_row[: self.n],
            "levels": g.levels[: self.n],
            "scalars": np.array([self.n, self.n_upper, g.entry, g.max_level], dtype=np.int64),
        }

    def load_state(self, st: dict[str, np.ndarray]) -> None:
        n, n_upper, entry, max_level = (int(x) for x in st["scalars"])
        self.ensure_capacity(max(n, 1), max(n_upper, 1))
        g = self.g
        g.nbr0[:n] = st["nbr0"]
        g.upper[:n_upper] = st["upper"]
        g.upper_row[:n] = st["upper_row"]
        g.levels[:n] = st["levels"]
        g.entry, g.max_level = entry, max_level
        self.n, self.n_upper = n, n_upper


# ---------------------------------------------------------------------------
# pure-Python reference twins (tests only)
# ---------------------------------------------------------------------------


def _ref_d(a: np.ndarray, b: np.ndarray) -> float:
    t = a.astype(np.float64) - b.astype(np.float64)
    return float(t @ t)


def _ref_row(g, rowbase, offset, node):
    row = node if offset < 0 else rowbase[node] + offset
    return [int(e) for e in g[row] if e >= 0]


def ref_greedy(q, vecs, g, rowbase, offset, cur, dcur):
    changed = True
    while changed:
        changed = False
        for e in _ref_row(g, rowbase, offset, cur):
            d = _ref_d(q, vecs[e])
            if d < dcur:
                dcur, cur, changed = d, e, True
    return cur, dcur


def ref_search_layer(q, vecs, g, rowbase, offset, eps, ef):
    visited = set()
    cand: list[tuple[float, int]] = []
    res: list[tuple[float, int]] = []  # max-heap via negation
    for e in eps:
        if e in visited:
            continue
        visited.add(e)
        d = _ref_d(q, vecs[e])
        heapq.heappush(cand, (d, e))
        heapq.heappush(res, (-d, e))
        if len(res) > ef:
            heapq.heappop(res)
    while cand:
        dc, c = cand[0]
        if len(res) >= ef and dc > -res[0][0]:
            break
        heapq.heappop(cand)
        for e in _ref_row(g, rowbase, offset, c):
            if e in visited:
                continue
            visited.add(e)
            d = _ref_d(q, vecs[e])
            if len(res) < ef or d < -res[0][0]:
                heapq.heappush(cand, (d, e))
                heapq.heappush(res, (-d, e))
                if len(res) > ef:
                    heapq.heappop(res)
    out = sorted((-d, e) for d, e in res)
    return [e for _, e in out], [d for d, _ in out]


def ref_select(vecs, cand_ids, cand_d, m):
    if len(cand_ids) < m:
        return list(cand_ids)
    out: list[int] = []
    for e, de in zip(cand_ids, cand_d, strict=True):
        if len(out) >= m:
            break
        if all(_ref_d(vecs[e], vecs[r]) >= de for r in out):
            out.append(int(e))
    return out
