"""Query planner: pick a filter strategy from estimated selectivity.

Three ways to answer "top-k nearest among records matching P":

``post_filter``   (P matches almost everything) plain HNSW with an over-fetched
                  k, then drop non-matching hits. Cheapest when few are dropped;
                  falls back to ``bitmap`` if too few survive.
``bitmap``        (in between) HNSW traversal walks the whole graph but only
                  admits nodes whose bit is set in P's bitmap into the results.
``brute_force``   (P matches very few) skip the graph: exact distances to just
                  the matching ids. With s * N small this is both exact and
                  faster than a graph walk that keeps missing matches.

Selectivity comes from the attribute statistics persisted in SQLite (exact
counting is used instead for small collections or before stats exist). The
thresholds live in config under ``planner`` and are tuned in bench/filters.py.
Bitmaps are cached per (predicate, write version), so repeated filters cost one
evaluation until the next write.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.filters import Node, canonical, estimate, evaluate
from storage.meta import MetaStore

STRATEGIES = ("post_filter", "bitmap", "brute_force")


@dataclass
class Plan:
    strategy: str
    selectivity: float
    exact_selectivity: bool
    mask: np.ndarray | None = None  # bool over ids 0..n-1 (bitmap / brute force)


class Planner:
    def __init__(self, cfg: dict[str, Any]) -> None:
        p = cfg["planner"]
        self.brute_max = float(p["brute_force_max_selectivity"])
        self.post_min = float(p["post_filter_min_selectivity"])
        self.post_max_ef = int(p["post_filter_max_ef"])
        self.exact_below = int(p.get("exact_selectivity_below", 50_000))
        self.cache_size = int(p["mask_cache_size"])
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()

    def mask(self, node: Node, meta: MetaStore, n: int, version: int) -> np.ndarray:
        key = (canonical(node), version)
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None and hit.shape[0] >= n:
                self._cache.move_to_end(key)
                return hit
        m = evaluate(node, meta.columns, n)
        with self._cache_lock:
            self._cache[key] = m
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return m

    def plan(
        self,
        node: Node,
        meta: MetaStore,
        n: int,
        version: int,
        force: str | None = None,
    ) -> Plan:
        mask = None
        est = None if n < self.exact_below else estimate(node, meta.stats)
        if est is None:
            mask = self.mask(node, meta, n, version)
            sel, exact = float(mask.mean()) if n else 0.0, True
        else:
            sel, exact = est, False
        if force is not None:
            if force not in STRATEGIES:
                raise ValueError(f"unknown strategy {force!r}")
            strategy = force
        elif sel <= self.brute_max:
            strategy = "brute_force"
        elif sel >= self.post_min:
            strategy = "post_filter"
        else:
            strategy = "bitmap"
        if strategy != "post_filter" and mask is None:
            mask = self.mask(node, meta, n, version)
        return Plan(strategy, sel, exact, mask)

    def post_filter_fetch(self, k: int, ef: int, selectivity: float) -> tuple[int, int]:
        """(fetch_k, ef) for post-filtering: over-fetch so ~k survive the filter."""
        fetch = math.ceil(1.5 * k / max(selectivity, 1e-3)) + 4
        fetch = min(fetch, self.post_max_ef)
        return fetch, min(max(ef, fetch), self.post_max_ef)
