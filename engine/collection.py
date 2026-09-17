"""A collection: vectors + attributes + index, behind one writer lock.

Concurrency model
-----------------
Reads are lock-free; writes are serialised behind a single writer lock.

* Writers never free or shrink memory a reader might hold. Every growable array
  (vector file, tombstones, graph tables, attribute columns) grows by allocating
  a bigger copy and swapping the reference, so a reader that captured the old
  reference keeps searching a complete, slightly stale array.
* A new vector is fully written before any graph edge points at it, and the
  published count is bumped last, so readers never follow an edge into garbage.
* In-place edits that do happen under readers (rewriting a neighbour row while
  pruning, setting a tombstone) are benign races: a reader can see an older or
  newer neighbour list, never an out-of-range id.

The trade-off: a search running concurrently with a batch insert may miss the
newest points or visit a just-deleted one (it is filtered out of the results),
and writes do not scale with cores beyond what one batch insert parallelises.
For a read-heavy ANN workload that is the right side of the trade: reads scale
linearly with the gRPC thread pool because Numba kernels release the GIL.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import apply_dotted, default_config
from index.distance import normalize
from index.flat import search_flat
from index.hnsw import HNSWIndex
from index.pq import ProductQuantizer
from storage.meta import AttrValue, MetaStore
from storage.vectors import VectorStore


class CollectionError(Exception):
    """Base class for errors that map to a client-facing status."""


class InvalidArgument(CollectionError):
    pass


@dataclass
class SearchResult:
    ids: list[str]
    distances: np.ndarray
    internal_ids: np.ndarray
    strategy: str = "flat"
    ef: int = 0
    selectivity: float = 1.0
    attributes: list[dict[str, AttrValue]] | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class Collection:
    CONFIG_FILE = "collection.json"

    def __init__(self, path: Path, cfg: dict[str, Any]) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.name: str = cfg["name"]
        self.dim: int = int(cfg["dim"])
        self.metric: str = cfg["metric"]
        self.index_type: str = cfg["index"]
        cap = int(cfg["initial_capacity"])
        self._lock = threading.Lock()
        self.meta = MetaStore(self.path / "meta.sqlite", capacity=cap)
        self.store = VectorStore(self.path / "vectors", self.dim, capacity=cap)
        self.deleted = np.zeros(cap, dtype=np.uint8)
        self.n = 0  # published count of internal ids (readers only look below it)
        self.n_deleted = 0
        self.hnsw: HNSWIndex | None = None
        if self.index_type in ("hnsw", "hnsw_pq"):
            self.hnsw = HNSWIndex.from_config(self.dim, cfg, capacity=cap)
        # PQ state is published as one (codebooks, codes) tuple so a reader never
        # sees trained codebooks next to not-yet-encoded codes.
        self.pq: ProductQuantizer | None = None
        self._pq_view: tuple[np.ndarray, np.ndarray] | None = None
        if self.index_type == "hnsw_pq":
            self.pq = ProductQuantizer.from_config(self.dim, cfg)

    # ---- lifecycle --------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        name: str,
        dim: int,
        overrides: dict[str, Any] | None = None,
    ) -> Collection:
        path = Path(path)
        if (path / cls.CONFIG_FILE).exists():
            raise FileExistsError(f"collection already exists at {path}")
        cfg = apply_dotted(default_config(), overrides or {})
        cfg.update(name=name, dim=int(dim))
        if cfg["metric"] not in ("l2", "cosine"):
            raise InvalidArgument(f"unknown metric {cfg['metric']!r}")
        if cfg["index"] not in ("flat", "hnsw", "hnsw_pq"):
            raise InvalidArgument(f"unknown index type {cfg['index']!r}")
        if dim <= 0:
            raise InvalidArgument("dim must be positive")
        path.mkdir(parents=True, exist_ok=True)
        (path / cls.CONFIG_FILE).write_text(json.dumps(cfg, indent=2))
        col = cls(path, cfg)
        col.meta.put_config(cfg)
        return col

    @classmethod
    def open(cls, path: str | Path) -> Collection:
        path = Path(path)
        cfg = json.loads((path / cls.CONFIG_FILE).read_text())
        col = cls(path, cfg)
        col._recover()
        return col

    def _recover(self) -> None:
        self.meta.load_from_sqlite()
        n = self.meta.count
        self._ensure_capacity(n)
        for i, u in enumerate(self.meta.internal_to_user):
            if u is None:
                self.deleted[i] = 1
        self.n_deleted = int(self.deleted[:n].sum())
        if self.pq is not None and n >= int(self.cfg["pq"]["train_size"]):
            self._train_pq(n)  # seeded, so this reproduces the original codebooks
        if self.hnsw is not None:
            self.hnsw.add(self.store.array, n)
        self.n = n

    def close(self) -> None:
        with self._lock:
            self.store.close()
            self.meta.close()

    # ---- writes -------------------------------------------------------------

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise InvalidArgument(f"expected vectors of dim {self.dim}, got shape {v.shape}")
        if not np.isfinite(v).all():
            raise InvalidArgument("vectors must be finite")
        if self.metric == "cosine":
            v = normalize(v)
        return np.ascontiguousarray(v)

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        attributes: list[dict[str, AttrValue]] | None = None,
    ) -> int:
        """Insert or replace records. Returns the number of records written."""
        vecs = self._prepare_vectors(vectors)
        ids = [str(u) for u in ids]
        if len(ids) != vecs.shape[0]:
            raise InvalidArgument("ids and vectors have different lengths")
        attrs = attributes if attributes is not None else [{} for _ in ids]
        if len(attrs) != len(ids):
            raise InvalidArgument("ids and attributes have different lengths")
        if not ids:
            return 0
        with self._lock:
            try:
                self.meta.check_kinds(attrs)
            except (TypeError, ValueError) as e:
                raise InvalidArgument(str(e)) from e
            new_ids, replaced = self.meta.plan_upsert(ids)
            self._apply_upsert(new_ids, ids, vecs, attrs, replaced)
        return len(ids)

    def _apply_upsert(
        self,
        new_ids: np.ndarray,
        ids: list[str],
        vecs: np.ndarray,
        attrs: list[dict[str, AttrValue]],
        replaced: np.ndarray,
    ) -> None:
        end = int(new_ids.max()) + 1
        self._ensure_capacity(end)
        # Vector first, then metadata, then graph edges: by the time any edge
        # points at a new node, everything a reader could fetch for it exists.
        self.store.write(new_ids, vecs)
        self._encode(new_ids, vecs, end)
        self.meta.apply_upsert(new_ids, ids, attrs, replaced)
        self._tombstone(replaced)
        if self.hnsw is not None:
            self.hnsw.add(self.store.array, end)
        self.n = max(self.n, end)

    def delete(self, ids: list[str]) -> int:
        with self._lock:
            internal = self.meta.lookup([str(u) for u in ids])
            if internal.size:
                self._apply_delete(internal)
            return int(internal.size)

    def _apply_delete(self, internal: np.ndarray) -> None:
        self.meta.apply_delete(internal)
        self._tombstone(internal)

    def _tombstone(self, internal: np.ndarray) -> None:
        if len(internal):
            fresh = internal[self.deleted[internal] == 0]
            self.deleted[internal] = 1
            self.n_deleted += int(fresh.size)

    def _ensure_capacity(self, rows: int) -> None:
        # Order matters for lock-free readers, which capture the graph first and
        # the vectors/tombstones second: everything a graph can reference must
        # already be at least as large as the graph.
        self.store.ensure_capacity(rows)
        if rows > self.deleted.shape[0]:
            cap = self.deleted.shape[0]
            while cap < rows:
                cap *= 2
            grown = np.zeros(cap, dtype=np.uint8)
            grown[: self.deleted.shape[0]] = self.deleted
            self.deleted = grown
        if self._pq_view is not None and rows > self._pq_view[1].shape[0]:
            codebooks, codes = self._pq_view
            grown = np.zeros((self.store.capacity, codes.shape[1]), dtype=np.uint8)
            grown[: codes.shape[0]] = codes
            self._pq_view = (codebooks, grown)
        if self.hnsw is not None:
            self.hnsw.ensure_capacity(rows)

    # ---- product quantization -------------------------------------------------

    def _encode(self, new_ids: np.ndarray, vecs: np.ndarray, end: int) -> None:
        if self.pq is None:
            return
        if self._pq_view is not None:
            self._pq_view[1][new_ids] = self.pq.encode(vecs)
        elif end >= int(self.cfg["pq"]["train_size"]):
            self._train_pq(end)

    def _train_pq(self, end: int) -> None:
        """One-off: train codebooks on a seeded sample of live vectors, encode all.

        Runs inside the writer (blocking writes, not reads) the first time the
        collection reaches ``pq.train_size`` vectors."""
        alive = np.flatnonzero(self.deleted[:end] == 0)
        size = min(int(self.cfg["pq"]["train_size"]), alive.size)
        rng = np.random.default_rng(int(self.cfg["pq"]["seed"]))
        vecs = self.store.array
        pq = ProductQuantizer.from_config(self.dim, self.cfg)
        pq.train(vecs[np.sort(rng.choice(alive, size, replace=False))])
        codes = np.zeros((self.store.capacity, pq.m), dtype=np.uint8)
        codes[:end] = pq.encode(vecs[:end])
        self.pq = pq
        self._pq_view = (pq.codebooks, codes)

    # ---- reads ------------------------------------------------------------

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        filter: str | None = None,
        ef: int | None = None,
        include_attributes: bool = False,
    ) -> SearchResult:
        q = self._prepare_vectors(vector)[0]
        if k <= 0:
            raise InvalidArgument("k must be positive")
        if ef is not None and ef <= 0:
            raise InvalidArgument("ef must be positive")
        if filter:
            raise InvalidArgument("filtered search is not supported yet")
        # Capture order: graph, then count, vectors and tombstones (see _ensure_capacity).
        g = self.hnsw.g if self.hnsw is not None else None
        pqv = self._pq_view
        n, vecs, deleted = self.n, self.store.array, self.deleted
        if g is None:
            internal, dists = search_flat(vecs, n, q, k, deleted)
            return self._result(internal, dists, "flat", 0, 1.0, include_attributes)
        ef = int(ef or self.cfg["hnsw"]["ef_search"])
        if pqv is not None:
            internal, dists, st = self.hnsw.search_pq(
                vecs, pqv[0], pqv[1], q, k, ef, self._rerank_depth(k, ef), deleted, g=g
            )
            return self._result(internal, dists, "hnsw_pq", ef, 1.0, include_attributes, st)
        internal, dists, st = self.hnsw.search(vecs, q, k, ef, deleted, g=g)
        return self._result(internal, dists, "hnsw", ef, 1.0, include_attributes, st)

    def _rerank_depth(self, k: int, ef: int) -> int:
        p = self.cfg["pq"]
        return max(k, min(int(p["rerank_max"]), int(p["rerank_factor"]) * ef))

    def _result(
        self,
        internal: np.ndarray,
        dists: np.ndarray,
        strategy: str,
        ef: int,
        selectivity: float,
        include_attributes: bool,
        stats: dict[str, Any] | None = None,
    ) -> SearchResult:
        users = self.meta.internal_to_user
        keep = [j for j, i in enumerate(internal) if users[int(i)] is not None]
        internal, dists = internal[keep], dists[keep]
        attrs = [self.meta.attributes(int(i)) for i in internal] if include_attributes else None
        return SearchResult(
            ids=[users[int(i)] for i in internal],
            distances=np.asarray(dists, dtype=np.float32),
            internal_ids=np.asarray(internal),
            strategy=strategy,
            ef=ef,
            selectivity=selectivity,
            attributes=attrs,
            stats=stats or {},
        )

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "metric": self.metric,
            "index": self.index_type,
            "count": self.n - self.n_deleted,
            "deleted": self.n_deleted,
            "index_bytes": self.index_bytes(),
            "lsn": 0,
        }

    def index_bytes(self) -> int:
        """Bytes of the in-memory search structure. In hnsw_pq mode, once trained,
        full vectors stay on disk (memmap) and are read only to re-rank."""
        graph = self.hnsw.nbytes() if self.hnsw is not None else 0
        pqv = self._pq_view
        if pqv is not None:
            return graph + int(pqv[0].nbytes) + self.n * pqv[1].shape[1]
        return self.store.nbytes(self.n) + graph
