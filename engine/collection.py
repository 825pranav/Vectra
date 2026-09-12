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
        self.n = self.meta.count
        self._ensure_capacity(self.n)
        for i, u in enumerate(self.meta.internal_to_user):
            if u is None:
                self.deleted[i] = 1
        self.n_deleted = int(self.deleted[: self.n].sum())

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
        self.store.write(new_ids, vecs)
        self.meta.apply_upsert(new_ids, ids, attrs, replaced)
        self._tombstone(replaced)
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
        self.store.ensure_capacity(rows)
        if rows > self.deleted.shape[0]:
            cap = self.deleted.shape[0]
            while cap < rows:
                cap *= 2
            grown = np.zeros(cap, dtype=np.uint8)
            grown[: self.deleted.shape[0]] = self.deleted
            self.deleted = grown

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
        n, deleted, vecs = self.n, self.deleted, self.store.array
        internal, dists = search_flat(vecs, n, q, k, deleted)
        return self._result(internal, dists, "flat", 0, 1.0, include_attributes)

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
        return self.store.nbytes(self.n)
