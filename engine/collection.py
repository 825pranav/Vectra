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
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import apply_dotted, default_config
from engine.filters import FilterError, evaluate, parse
from engine.planner import Plan, Planner
from index.distance import normalize
from index.flat import search_flat, search_subset
from index.hnsw import HNSWIndex
from index.pq import ProductQuantizer
from ml.early_stop import EarlyStopModel
from storage.meta import AttrValue, MetaStore
from storage.vectors import VectorStore
from storage.wal import Delete, Snapshots, WriteAheadLog, encode_delete, encode_upsert

_ATTR_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
        self.version = 0  # bumped by every write; keys the planner's bitmap cache
        self.planner = Planner(cfg)
        self._churn = 0  # writes since attribute statistics were last refreshed
        self.hnsw: HNSWIndex | None = None
        if self.index_type in ("hnsw", "hnsw_pq"):
            self.hnsw = HNSWIndex.from_config(self.dim, cfg, capacity=cap)
        # PQ state is published as one (codebooks, codes) tuple so a reader never
        # sees trained codebooks next to not-yet-encoded codes.
        self.pq: ProductQuantizer | None = None
        self._pq_view: tuple[np.ndarray, np.ndarray] | None = None
        if self.index_type == "hnsw_pq":
            self.pq = ProductQuantizer.from_config(self.dim, cfg)
        dur = cfg["durability"]
        self.wal = WriteAheadLog(self.path / "wal", fsync=bool(dur["fsync"]))
        self.snapshots = Snapshots(self.path / "snapshots")
        self.snapshot_every = int(dur["snapshot_every"])
        # Optional learned search budget (ml/early_stop.py). Off by default: on
        # the benchmarked datasets it does not beat a well-chosen fixed ef.
        self.budget_model: EarlyStopModel | None = None
        es = cfg["early_stop"]
        if es.get("model") and self.index_type == "hnsw":
            path = Path(es["model"])
            path = path if path.is_absolute() else self.path / path
            self.set_budget_model(EarlyStopModel.load(path))
        self.lsn = 0  # last log sequence number applied
        self._since_snapshot = 0  # rows written since the last snapshot

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
        col.wal.start(1)
        return col

    @classmethod
    def open(cls, path: str | Path) -> Collection:
        path = Path(path)
        cfg = json.loads((path / cls.CONFIG_FILE).read_text())
        col = cls(path, cfg)
        col._recover()
        return col

    def _recover(self) -> None:
        """Newest complete snapshot + replay of every WAL record after it.

        Replay goes through the same ``_apply_*`` functions as live writes. They
        are idempotent against SQLite and the vector file (both may already hold
        records newer than the snapshot), while the in-memory state restored
        from the snapshot is exactly as old as the snapshot's LSN."""
        cur = self.snapshots.current()
        if cur is not None:
            self._restore(*Snapshots.load(cur[1]))
        for rec in self.wal.replay(self.lsn):
            if isinstance(rec, Delete):
                self._apply_delete(rec.internal_ids)
                count = int(rec.internal_ids.size)
            else:
                self._apply_upsert(
                    rec.internal_ids, rec.user_ids, rec.vectors, rec.attrs, rec.replaced
                )
                count = int(rec.internal_ids.size)
            self.lsn = rec.lsn
            self.version += 1
            self._since_snapshot += count
        self.wal.start(self.lsn + 1)
        if self.meta.columns:
            self.meta.refresh_stats()

    def _restore(self, arrays: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        n = int(info["n"])
        self._ensure_capacity(max(n, 1))
        self.meta.restore_state(
            {
                "arrays": {k: v for k, v in arrays.items() if k.startswith("col.")},
                "users": arrays["users"],
                "alive": arrays["alive"],
                "info": info["meta"],
            }
        )
        self.deleted[:n] = arrays["deleted"]
        self.n_deleted = int(info["n_deleted"])
        if self.hnsw is not None:
            self.hnsw.load_state({k[6:]: v for k, v in arrays.items() if k.startswith("graph.")})
        if self.pq is not None and "pq.codebooks" in arrays:
            self.pq.codebooks = arrays["pq.codebooks"]
            codes = np.zeros((self.store.capacity, self.pq.m), dtype=np.uint8)
            codes[:n] = arrays["pq.codes"]
            self._pq_view = (self.pq.codebooks, codes)
        self.n = n
        self.lsn = int(info["lsn"])
        self.version = int(info["version"])

    def snapshot(self) -> int:
        """Persist in-memory state and truncate the WAL. Returns the snapshot LSN."""
        with self._lock:
            self._snapshot()
            return self.lsn

    def _snapshot(self) -> None:
        n = self.n
        self.store.flush()  # vectors [0, n) must be durable before the WAL is cut
        ms = self.meta.snapshot_state()
        arrays: dict[str, np.ndarray] = {
            "deleted": self.deleted[:n].copy(),
            "users": ms["users"],
            "alive": ms["alive"],
            **ms["arrays"],
        }
        if self.hnsw is not None:
            arrays.update({f"graph.{k}": v for k, v in self.hnsw.state().items()})
        if self._pq_view is not None:
            arrays["pq.codebooks"] = self._pq_view[0]
            arrays["pq.codes"] = self._pq_view[1][:n]
        info = {
            "lsn": self.lsn,
            "n": n,
            "n_deleted": self.n_deleted,
            "version": self.version,
            "meta": ms["info"],
        }
        self.snapshots.write(self.lsn, arrays, info)
        self.wal.rotate(self.lsn)
        self._since_snapshot = 0

    def close(self) -> None:
        with self._lock:
            if self._since_snapshot:
                self._snapshot()  # next open skips the replay
            self.wal.close()
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
        for rec in attrs:
            for key in rec:
                if not _ATTR_KEY.match(key):
                    raise InvalidArgument(f"invalid attribute name {key!r}")
        with self._lock:
            try:
                self.meta.check_kinds(attrs)
            except (TypeError, ValueError) as e:
                raise InvalidArgument(str(e)) from e
            new_ids, replaced = self.meta.plan_upsert(ids)
            lsn = self.lsn + 1
            # Durable before applied, applied before acknowledged.
            self.wal.append(encode_upsert(lsn, new_ids, replaced, vecs, ids, attrs))
            self._apply_upsert(new_ids, ids, vecs, attrs, replaced)
            self.lsn = lsn
            self._after_write(len(ids))
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
                lsn = self.lsn + 1
                self.wal.append(encode_delete(lsn, internal))
                self._apply_delete(internal)
                self.lsn = lsn
                self._after_write(int(internal.size))
            return int(internal.size)

    def _after_write(self, count: int) -> None:
        self.version += 1
        self._churn += count
        frac = float(self.cfg["planner"]["stats_refresh_fraction"])
        if self._churn >= frac * max(self.n, 1) and self.meta.columns:
            self.meta.refresh_stats()
            self._churn = 0
        self._since_snapshot += count
        if self.snapshot_every and self._since_snapshot >= self.snapshot_every:
            self._snapshot()

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
        strategy: str | None = None,
    ) -> SearchResult:
        """Top-k search. ``strategy`` forces a filter strategy (tests, benchmarks)."""
        q = self._prepare_vectors(vector)[0]
        if k <= 0:
            raise InvalidArgument("k must be positive")
        if ef is not None and ef <= 0:
            raise InvalidArgument("ef must be positive")
        node = None
        if filter:
            try:
                node = parse(filter)
            except FilterError as e:
                raise InvalidArgument(f"bad filter: {e}") from e
        # Capture order: graph, then PQ view, count, vectors, tombstones, version
        # (see _ensure_capacity for why the graph must come first).
        g = self.hnsw.g if self.hnsw is not None else None
        pqv = self._pq_view
        n, vecs, deleted, version = self.n, self.store.array, self.deleted, self.version
        # Budget: an explicit ef wins; otherwise the learned model if one is
        # loaded (graph + full-precision search only); otherwise the config default.
        model = self.budget_model if ef is None and g is not None and pqv is None else None
        ef = int(ef or self.cfg["hnsw"]["ef_search"])

        if node is None:
            if g is None:
                internal, dists = search_flat(vecs, n, q, k, deleted)
                return self._result(internal, dists, "flat", 0, 1.0, include_attributes)
            if model is not None:
                internal, dists, st = model.search(self.hnsw, vecs, q, k, None, deleted, g=g)
                return self._result(internal, dists, "hnsw+early_stop", st["hops"], 1.0,
                                    include_attributes, st)  # fmt: skip
            internal, dists, st = self._graph(g, pqv, vecs, deleted, q, k, ef, None)
            base = "hnsw_pq" if pqv is not None else "hnsw"
            return self._result(internal, dists, base, ef, 1.0, include_attributes, st)
        try:
            return self._filtered(
                node, strategy, g, pqv, n, vecs, deleted, version, q, k, ef, include_attributes,
                model,
            )  # fmt: skip
        except FilterError as e:
            raise InvalidArgument(f"bad filter: {e}") from e

    def _filtered(
        self, node, strategy, g, pqv, n, vecs, deleted, version, q, k, ef, attrs, model=None
    ):
        force = "brute_force" if g is None else strategy
        cap = 0 if g is None else g.capacity
        try:
            plan = self.planner.plan(node, self.meta, n, version, force, cap)
        except ValueError as e:
            if isinstance(e, FilterError):
                raise
            raise InvalidArgument(str(e)) from e
        if plan.strategy == "brute_force":
            ids = np.flatnonzero(plan.mask[:n] & (deleted[:n] == 0))
            internal, dists = search_subset(vecs, ids, q, k)
            st = {"candidates": int(ids.size)}
            return self._result(internal, dists, "brute_force", 0, plan.selectivity, attrs, st)
        if plan.strategy == "bitmap":
            if model is not None:  # selectivity is one of the model's features
                internal, dists, st = model.search(self.hnsw, vecs, q, k, None, deleted,
                                                   plan.mask, plan.selectivity, g)  # fmt: skip
                return self._result(internal, dists, "bitmap+early_stop", st["hops"],
                                    plan.selectivity, attrs, st)  # fmt: skip
            return self._bitmap(plan, g, pqv, vecs, deleted, q, k, ef, n, attrs)
        fetch, ef_post = self.planner.post_filter_fetch(k, ef, plan.selectivity)
        internal, dists, st = self._graph(g, pqv, vecs, deleted, q, fetch, ef_post, None)
        keep = evaluate(node, self.meta.columns, internal.astype(np.int64))
        internal, dists = internal[keep][:k], dists[keep][:k]
        if internal.shape[0] < k and strategy is None:
            # Too few survived (the estimate was optimistic): redo with a bitmap.
            plan.mask = self.planner.mask(node, self.meta, n, version, g.capacity)
            res = self._bitmap(plan, g, pqv, vecs, deleted, q, k, ef, n, attrs)
            res.strategy = "post_filter+bitmap"
            return res
        return self._result(internal, dists, "post_filter", ef_post, plan.selectivity, attrs, st)

    def _graph(self, g, pqv, vecs, deleted, q, k, ef, mask):
        if pqv is not None:
            depth = self._rerank_depth(k, ef)
            return self.hnsw.search_pq(vecs, pqv[0], pqv[1], q, k, ef, depth, deleted, mask, g=g)
        return self.hnsw.search(vecs, q, k, ef, deleted, mask, g=g)

    def _bitmap(self, plan: Plan, g, pqv, vecs, deleted, q, k, ef, n, attrs):
        # plan.mask spans the graph's whole capacity: the graph may already link
        # ids >= n (published after we read n), and those stay False.
        internal, dists, st = self._graph(g, pqv, vecs, deleted, q, k, ef, plan.mask)
        return self._result(internal, dists, "bitmap", ef, plan.selectivity, attrs, st)

    def set_budget_model(self, model: EarlyStopModel | None) -> None:
        """Use ``model`` to pick the per-query budget when a search passes no ef."""
        if model is not None and self.index_type != "hnsw":
            raise InvalidArgument("learned budgets are only supported for the hnsw index")
        if model is not None:
            model.meta.setdefault("multiplier", float(self.cfg["early_stop"]["multiplier"]))
        self.budget_model = model

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
            "lsn": self.lsn,
        }

    def index_bytes(self) -> int:
        """Bytes of the in-memory search structure. In hnsw_pq mode, once trained,
        full vectors stay on disk (memmap) and are read only to re-rank."""
        graph = self.hnsw.nbytes() if self.hnsw is not None else 0
        pqv = self._pq_view
        if pqv is not None:
            return graph + int(pqv[0].nbytes) + self.n * pqv[1].shape[1]
        return self.store.nbytes(self.n) + graph
