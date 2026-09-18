"""Metadata store: SQLite for durability, NumPy columns for query-time work.

SQLite holds the id map (internal id <-> user id), every attribute value, the
collection config and the per-attribute statistics the planner uses to estimate
filter selectivity. Query-time code never touches SQLite: attributes are mirrored
into one NumPy column per key (float64 with NaN for missing numbers, int32
dictionary codes with -1 for missing strings) so filters evaluate as vectorised
masks and result joins are plain fancy indexing.

All mutating methods are called by the collection's single writer. Columns grow
by copy-and-swap, so a concurrent reader always sees a complete (if slightly
stale) array.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

AttrValue = float | int | str | bool

SCHEMA = """
CREATE TABLE IF NOT EXISTS ids (
    internal_id INTEGER PRIMARY KEY,
    user_id     TEXT NOT NULL,
    alive       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ids_by_user ON ids(user_id);
CREATE TABLE IF NOT EXISTS attrs (
    internal_id INTEGER NOT NULL,
    key         TEXT NOT NULL,
    num         REAL,
    txt         TEXT,
    PRIMARY KEY (internal_id, key)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS attr_kinds (key TEXT PRIMARY KEY, kind TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attr_stats (key TEXT PRIMARY KEY, stats TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def value_kind(v: AttrValue) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int | float):
        return "num"
    if isinstance(v, str):
        return "str"
    raise TypeError(f"unsupported attribute value {v!r}")


class Column:
    """One attribute across all internal ids."""

    def __init__(self, kind: str, capacity: int) -> None:
        self.kind = kind
        self.vocab: list[str] = []
        self.lookup: dict[str, int] = {}
        if kind == "str":
            self.data = np.full(capacity, -1, dtype=np.int32)
        else:
            self.data = np.full(capacity, np.nan, dtype=np.float64)

    def ensure(self, capacity: int) -> None:
        if capacity <= self.data.shape[0]:
            return
        new_cap = max(self.data.shape[0], 1)
        while new_cap < capacity:
            new_cap *= 2
        fill = -1 if self.kind == "str" else np.nan
        grown = np.full(new_cap, fill, dtype=self.data.dtype)
        grown[: self.data.shape[0]] = self.data
        self.data = grown

    def code(self, s: str) -> int:
        c = self.lookup.get(s)
        if c is None:
            c = len(self.vocab)
            self.vocab.append(s)
            self.lookup[s] = c
        return c

    def set(self, i: int, v: AttrValue) -> None:
        self.data[i] = self.code(v) if self.kind == "str" else float(v)

    def clear(self, ids: np.ndarray) -> None:
        self.data[ids] = -1 if self.kind == "str" else np.nan

    def get(self, i: int) -> AttrValue | None:
        x = self.data[i]
        if self.kind == "str":
            return None if x < 0 else self.vocab[x]
        if np.isnan(x):
            return None
        if self.kind == "bool":
            return bool(x)
        return float(x)

    def stats(self, n: int) -> dict[str, Any]:
        """Summary used for selectivity estimation (see engine.filters)."""
        col = self.data[:n]
        if self.kind == "str":
            present = col[col >= 0]
            counts = np.bincount(present, minlength=len(self.vocab)) if present.size else []
            return {
                "kind": "str",
                "n": int(n),
                "null_frac": float(1 - present.size / max(n, 1)),
                "freq": {self.vocab[c]: int(k) for c, k in enumerate(counts) if k},
            }
        present = col[~np.isnan(col)]
        qs = np.quantile(present, np.linspace(0, 1, 101)).tolist() if present.size else []
        uniq, counts = np.unique(present, return_counts=True)
        out = {
            "kind": self.kind,
            "n": int(n),
            "null_frac": float(1 - present.size / max(n, 1)),
            "quantiles": qs,
            "n_distinct": int(uniq.size),
        }
        if uniq.size <= 256:  # low-cardinality: exact value frequencies
            out["freq"] = {repr(float(u)): int(c) for u, c in zip(uniq, counts, strict=True)}
        return out


class MetaStore:
    def __init__(self, path: Path, capacity: int = 1024) -> None:
        self.path = Path(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.capacity = capacity
        self.user_to_internal: dict[str, int] = {}
        self.internal_to_user: list[str | None] = []
        self.columns: dict[str, Column] = {}
        self.stats: dict[str, dict[str, Any]] = {}

    # ---- config ---------------------------------------------------------

    def put_config(self, cfg: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO config VALUES ('collection', ?)", (json.dumps(cfg),)
        )

    def get_config(self) -> dict[str, Any] | None:
        row = self.db.execute("SELECT value FROM config WHERE key='collection'").fetchone()
        return json.loads(row[0]) if row else None

    # ---- id map ---------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of internal ids ever allocated (dense, includes tombstones)."""
        return len(self.internal_to_user)

    def plan_upsert(self, user_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Allocate internal ids for a batch. Returns (new_ids, replaced_ids).

        Pure: nothing is mutated, so the result can be logged to the WAL first.
        A user id that appears twice in one batch keeps its last occurrence.
        """
        start = self.count
        new_ids = np.arange(start, start + len(user_ids), dtype=np.int64)
        replaced = [self.user_to_internal[u] for u in user_ids if u in self.user_to_internal]
        last: dict[str, int] = {}
        for i, u in zip(new_ids, user_ids, strict=True):
            if u in last:
                replaced.append(last[u])
            last[u] = int(i)
        return new_ids, np.asarray(sorted(set(replaced)), dtype=np.int64)

    def lookup(self, user_ids: list[str]) -> np.ndarray:
        return np.asarray(
            [self.user_to_internal[u] for u in user_ids if u in self.user_to_internal],
            dtype=np.int64,
        )

    def check_kinds(self, attrs: list[dict[str, AttrValue]]) -> None:
        """Raise ValueError if a value's type disagrees with its column's type."""
        seen: dict[str, str] = {}
        for rec in attrs:
            for key, v in rec.items():
                kind = value_kind(v)
                expected = self.columns[key].kind if key in self.columns else seen.get(key)
                if expected is not None and expected != kind:
                    raise ValueError(f"attribute {key!r} is {expected}, got {kind} value {v!r}")
                seen[key] = kind

    # ---- apply (idempotent; also used by WAL replay) --------------------

    def apply_upsert(
        self,
        new_ids: np.ndarray,
        user_ids: list[str],
        attrs: list[dict[str, AttrValue]],
        replaced: np.ndarray,
    ) -> None:
        end = int(new_ids.max()) + 1 if len(new_ids) else self.count
        if end > self.count:
            self.internal_to_user.extend([None] * (end - self.count))
        self._ensure(end)
        for i, u in zip(new_ids, user_ids, strict=True):
            self.internal_to_user[int(i)] = u
        for r in replaced:
            self.internal_to_user[int(r)] = None
        for i, u in zip(new_ids, user_ids, strict=True):
            self.user_to_internal[u] = int(i)
        self._clear_attrs(replaced)
        dead = set(replaced.tolist())
        rows = []
        for i, rec in zip(new_ids, attrs, strict=True):
            if int(i) in dead:  # superseded by a later duplicate in the same batch
                continue
            for key, v in rec.items():
                col = self._column(key, value_kind(v))
                col.set(int(i), v)
                if col.kind == "str":
                    rows.append((int(i), key, None, v))
                else:
                    rows.append((int(i), key, float(v), None))
        with self._tx():
            self.db.executemany(
                "INSERT OR REPLACE INTO ids VALUES (?, ?, 1)",
                [(int(i), u) for i, u in zip(new_ids, user_ids, strict=True)],
            )
            if len(replaced):
                self.db.executemany(
                    "UPDATE ids SET alive=0 WHERE internal_id=?", [(int(r),) for r in replaced]
                )
                self.db.executemany(
                    "DELETE FROM attrs WHERE internal_id=?", [(int(r),) for r in replaced]
                )
            self.db.executemany("INSERT OR REPLACE INTO attrs VALUES (?, ?, ?, ?)", rows)

    def apply_delete(self, ids: np.ndarray) -> None:
        for i in ids:
            u = self.internal_to_user[int(i)]
            if u is not None and self.user_to_internal.get(u) == int(i):
                del self.user_to_internal[u]
            self.internal_to_user[int(i)] = None
        self._clear_attrs(ids)
        with self._tx():
            self.db.executemany(
                "UPDATE ids SET alive=0 WHERE internal_id=?", [(int(i),) for i in ids]
            )
            self.db.executemany("DELETE FROM attrs WHERE internal_id=?", [(int(i),) for i in ids])

    # ---- attributes -----------------------------------------------------

    def attributes(self, i: int) -> dict[str, AttrValue]:
        out = {}
        for key, col in self.columns.items():
            v = col.get(i)
            if v is not None:
                out[key] = v
        return out

    def refresh_stats(self) -> None:
        n = self.count
        stats = {key: col.stats(n) for key, col in self.columns.items()}
        with self._tx():
            self.db.executemany(
                "INSERT OR REPLACE INTO attr_stats VALUES (?, ?)",
                [(k, json.dumps(s)) for k, s in stats.items()],
            )
        self.stats = stats

    def _column(self, key: str, kind: str) -> Column:
        col = self.columns.get(key)
        if col is None:
            col = Column(kind, self.capacity)
            self.db.execute("INSERT OR REPLACE INTO attr_kinds VALUES (?, ?)", (key, kind))
            self.columns = {**self.columns, key: col}
        return col

    def _clear_attrs(self, ids: np.ndarray) -> None:
        if len(ids):
            for col in self.columns.values():
                col.clear(np.asarray(ids, dtype=np.int64))

    def _ensure(self, rows: int) -> None:
        while self.capacity < rows:
            self.capacity *= 2
        for col in self.columns.values():
            col.ensure(self.capacity)

    def _tx(self):
        return _Transaction(self.db)

    # ---- load / snapshot --------------------------------------------------

    def load_from_sqlite(self) -> None:
        """Rebuild all in-memory state from SQLite (used when no snapshot exists)."""
        rows = self.db.execute("SELECT internal_id, user_id, alive FROM ids ORDER BY internal_id")
        self.internal_to_user = []
        self.user_to_internal = {}
        for i, u, alive in rows:
            if i >= len(self.internal_to_user):
                self.internal_to_user.extend([None] * (i + 1 - len(self.internal_to_user)))
            if alive:
                self.internal_to_user[i] = u
                self.user_to_internal[u] = i
        self._ensure(max(self.count, 1))
        kinds = dict(self.db.execute("SELECT key, kind FROM attr_kinds"))
        self.columns = {k: Column(kind, self.capacity) for k, kind in kinds.items()}
        for i, key, num, txt in self.db.execute("SELECT internal_id, key, num, txt FROM attrs"):
            col = self.columns[key]
            col.set(i, txt if col.kind == "str" else num)
        self.stats = {
            k: json.loads(s) for k, s in self.db.execute("SELECT key, stats FROM attr_stats")
        }

    def snapshot_state(self) -> dict[str, Any]:
        """Arrays + small JSON needed to restore in-memory state without SQLite scans."""
        arrays = {f"col.{k}": c.data[: self.count] for k, c in self.columns.items()}
        users = np.array(["" if u is None else u for u in self.internal_to_user], dtype=str)
        alive = np.array([u is not None for u in self.internal_to_user], dtype=np.bool_)
        info = {
            "kinds": {k: c.kind for k, c in self.columns.items()},
            "vocab": {k: c.vocab for k, c in self.columns.items() if c.kind == "str"},
            "stats": self.stats,
        }
        return {"arrays": arrays, "users": users, "alive": alive, "info": info}

    def restore_state(self, state: dict[str, Any]) -> None:
        users, alive, info = state["users"], state["alive"], state["info"]
        pairs = zip(users.tolist(), alive, strict=True)
        self.internal_to_user = [u if a else None for u, a in pairs]
        self.user_to_internal = {u: i for i, u in enumerate(self.internal_to_user) if u is not None}
        self.capacity = max(self.capacity, 1)
        self._ensure(max(self.count, 1))
        self.columns = {}
        for key, kind in info["kinds"].items():
            col = Column(kind, self.capacity)
            data = state["arrays"][f"col.{key}"]
            col.data[: data.shape[0]] = data
            if kind == "str":
                col.vocab = list(info["vocab"][key])
                col.lookup = {s: c for c, s in enumerate(col.vocab)}
            self.columns[key] = col
        self.stats = info.get("stats", {})

    def close(self) -> None:
        self.db.close()


class _Transaction:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def __enter__(self) -> None:
        self.db.execute("BEGIN")

    def __exit__(self, exc_type, exc, tb) -> None:
        self.db.execute("COMMIT" if exc_type is None else "ROLLBACK")
