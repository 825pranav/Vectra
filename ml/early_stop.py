"""Learned early termination for HNSW search (after Li et al., SIGMOD 2020).

A fixed ``ef`` has to be large enough for the *hardest* queries, so easy queries
overpay. Here a LightGBM model looks at cheap signals from the search itself and
sets a per-query budget instead. Two variants, both benchmarked:

``upfront``     run a short probe with a small result heap, predict the ``ef``
                this query needs, resize the heap and let HNSW terminate
                normally. One prediction per query.
``checkpoint``  from the probe on, every ``interval`` expansions predict the
                total number of expansions the query needs; stop once reached
                ("can I stop now?"), re-predicting with fresher features.

Budget sizing uses the fact that an HNSW search at a given ``ef`` performs about
``ef`` expansions, so the checkpoint variant sizes the result heap from the same
prediction (``ef ~ budget``). That matters for speed: a search run under a
generously wide heap admits almost every neighbour into both heaps and costs
~2x per expansion, which would eat the savings (see bench/micro/es_overhead.py).

Features (all O(1)/O(candidates) at a checkpoint, see ``_njit_features``):
distance to the layer-0 entry point, best and k-th distances found so far
relative to it, the gap between them, how far the next candidate is beyond the
k-th, how many unexpanded candidates are still closer than the k-th, how much
the k-th improved since the last checkpoint, expansions and distance
computations so far, recent top-k updates, filter selectivity and dataset size.

Labels come from searches on held-out *training* queries only: the expansion at
which each true neighbour first entered the top-k under a generous search
(checkpoint), and the smallest ``ef`` that recovers everything the generous
search found (upfront). The model predicts log2 of the budget; a multiplier on
the prediction trades latency for recall and traces the curve.

The trained trees are flattened (ml/trees.py) and evaluated inside the Numba
search loop, so a prediction costs about a microsecond, not a Python call.
"""

# Imports: HNSW heap and descent kernels are reused so this search matches the normal one.
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

from index.distance import _njit_l2sq
from index.hnsw import (
    HNSWGraph,
    HNSWIndex,
    _njit_descend,
    _njit_drain_maxheap,
    _njit_grow,
    _njit_maxheap_pop,
    _njit_maxheap_push,
    _njit_minheap_pop,
    _njit_minheap_push,
)
from ml.trees import _njit_predict, flatten, load_forest, save_forest

# Names of the 12 features the model sees at each checkpoint, in array order.
FEATURES = [
    "log_d_entry",
    "best_over_entry",
    "kth_over_entry",
    "kth_over_best",
    "cand_over_kth",
    "log2_promising",
    "kth_change",
    "log2_hops",
    "log2_ndist",
    "recent_topk_updates",
    "selectivity",
    "log2_n",
]
N_FEAT = len(FEATURES)
# Search modes: fixed ef (normal or label recording), predict ef once, or re-predict a stop point.
MODE_FIXED, MODE_UPFRONT, MODE_CHECKPOINT = 0, 1, 2
# These kernels use inf as "no k-th result yet". Full fastmath lets LLVM assume
# no infinities (``x < inf`` folds to true), so keep every flag except ninf/nnan:
# reassociation is what vectorises the distance loop.
_FASTMATH = {"reassoc", "contract", "arcp", "nsz", "afn"}
_CAP = 1e3  # ratios are clamped: before k results exist the k-th distance is inf


# Fill `out` with the 12 cheap features describing how the search is going right now.
@njit(cache=True, fastmath=_FASTMATH, nogil=True)
def _njit_features(out, d_ep, topk, d_cand, promising, prev_kth, hops, ndist, recent, sel, log2n):
    eps = 1e-12
    k = topk.shape[0]
    # Best and k-th distances so far; "full" means k results have been found.
    d_best = topk[0]
    d_kth = topk[k - 1]
    full = d_kth < np.inf
    out[0] = math.log(d_ep + eps)
    out[1] = min(d_best / (d_ep + eps), _CAP)
    out[2] = min(d_kth / (d_ep + eps), _CAP) if full else _CAP
    out[3] = min(d_kth / (d_best + eps), _CAP) if full else _CAP
    out[4] = min(d_cand / (d_kth + eps), _CAP) if full else 0.0
    out[5] = math.log2(promising + 1.0)
    out[6] = d_kth / prev_kth if full and prev_kth < np.inf and prev_kth > 0 else 1.0
    out[7] = math.log2(hops + 1.0)
    out[8] = math.log2(ndist + 1.0)
    out[9] = recent
    out[10] = sel
    out[11] = log2n


# NumPy twin of the feature kernel, for tests.
def ref_features(d_ep, topk, d_cand, promising, prev_kth, hops, ndist, recent, sel, log2n):
    """NumPy twin of ``_njit_features``."""
    eps = 1e-12
    d_best, d_kth = float(topk[0]), float(topk[-1])
    full = np.isfinite(d_kth)
    return np.array(
        [
            math.log(d_ep + eps),
            min(d_best / (d_ep + eps), _CAP),
            min(d_kth / (d_ep + eps), _CAP) if full else _CAP,
            min(d_kth / (d_best + eps), _CAP) if full else _CAP,
            min(d_cand / (d_kth + eps), _CAP) if full else 0.0,
            math.log2(promising + 1.0),
            d_kth / prev_kth if full and np.isfinite(prev_kth) and prev_kth > 0 else 1.0,
            math.log2(hops + 1.0),
            math.log2(ndist + 1.0),
            recent,
            sel,
            log2n,
        ]
    )


# HNSW layer-0 search where a tree model chooses how much work to do for this query.
@njit(cache=True, fastmath=_FASTMATH, nogil=True)
# Returns top-k ids and distances, work counters, and (when recording) features and hit logs.
def _njit_search_es(
    q, vecs, nbr0, upper, upper_row, entry, max_level, k, ef_max,
    deleted, mask, use_mask, visited, tag,
    mode, probe, probe_ef, interval, mult, ef_min, forced_ef, sel, log2n,
    feature, threshold, left, right, value, roots,
    record, max_ckpt,
):  # fmt: skip
    """HNSW search whose budget is set by the flattened model (see module doc).

    ``MODE_FIXED`` runs a plain search at ``ef_max`` (with ``record``, it logs
    features at every checkpoint and every insertion into the running top-k,
    which is how training labels are made). ``forced_ef > 0`` replaces the
    upfront prediction (also for labels).
    """
    # Descend the upper layers as usual; start with the small probe ef unless mode is fixed.
    ep, dep, nd0 = _njit_descend(q, vecs, upper, upper_row, entry, max_level)
    ef = ef_max if mode == MODE_FIXED else min(max(probe_ef, k), ef_max)
    # Heaps sized for the largest allowed ef, plus a sorted running top-k for features.
    cap = max(64, 4 * ef_max)
    cd = np.empty(cap, dtype=np.float32)
    ci = np.empty(cap, dtype=np.int32)
    wd = np.empty(ef_max + 1, dtype=np.float32)
    wi = np.empty(ef_max + 1, dtype=np.int32)
    topk = np.full(k, np.inf, dtype=np.float32)
    feats = np.zeros((max_ckpt if record else 1, 12), dtype=np.float64)
    x = np.zeros(12, dtype=np.float64)
    log_i = np.empty(64 if record else 1, dtype=np.int32)
    log_h = np.empty(64 if record else 1, dtype=np.int32)
    n_log = 0
    n_feat = 0
    # Seed the walk with the entry point.
    visited[ep] = tag
    nc = _njit_minheap_push(cd, ci, 0, dep, ep)
    nw = 0
    if deleted[ep] == 0 and (not use_mask or mask[ep]):
        nw = _njit_maxheap_push(wd, wi, 0, dep, ep)
        topk[0] = dep
        if record:
            log_i[0] = ep
            log_h[0] = 0
            n_log = 1
    hops = 0
    ndist = nd0
    recent = 0
    budget = -1.0
    probed = False
    last_eval = -1
    prev_kth = np.float32(np.inf)
    width = nbr0.shape[1]
    # Main best-first loop; stops on normal HNSW termination or when the predicted budget is used.
    while nc > 0:
        if nw >= ef and cd[0] > wd[0]:
            break
        if budget >= 0.0 and hops >= budget:
            break
        # At the probe point or a checkpoint, compute features and maybe ask the model.
        if hops != last_eval:
            at_probe = mode != MODE_FIXED and not probed and hops >= probe
            at_ckpt = (
                hops > 0
                and hops % interval == 0
                and (record or (mode == MODE_CHECKPOINT and probed))
            )
            if at_probe or at_ckpt:
                last_eval = hops
                promising = 0
                # Count queued candidates that are still closer than the current k-th result.
                for t in range(nc):
                    if cd[t] < topk[k - 1]:
                        promising += 1
                _njit_features(x, dep, topk, cd[0], promising, prev_kth, hops, ndist,
                               recent / max(interval, 1), sel, log2n)  # fmt: skip
                recent = 0
                prev_kth = topk[k - 1]
                # When recording training data, keep this checkpoint's features.
                if record and n_feat < max_ckpt:
                    feats[n_feat, :] = x
                    n_feat += 1
                # Ask the model: predict log2 budget, scale by mult, and clamp it to a valid ef.
                if at_probe or mode == MODE_CHECKPOINT:
                    probed = True
                    if at_probe and forced_ef > 0:
                        ef = forced_ef
                    else:
                        b = 2.0 ** _njit_predict(x, feature, threshold, left, right, value, roots)
                        b = b * mult
                        ef = int(round(b))
                        if mode == MODE_CHECKPOINT:
                            budget = float(round(b))
                    ef = min(max(ef, ef_min, k), ef_max)
                    # Shrink the result heap to the new ef right away.
                    while nw > ef:
                        nw = _njit_maxheap_pop(wd, wi, nw)
                    continue  # re-check termination under the new budget
        # Expand the closest candidate, same as the normal layer-0 search.
        c = ci[0]
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
                    if de < topk[k - 1]:
                        # insertion-sort into the running top-k
                        t = k - 1
                        while t > 0 and topk[t - 1] > de:
                            topk[t] = topk[t - 1]
                            t -= 1
                        topk[t] = de
                        recent += 1
                    # log ties with the k-th too: integer-valued data (SIFT) has
                    # exact ties, and a tie can still end up in the final top-k
                    # Record when (at which hop) each node entered the top-k; used to make labels.
                    if record and de <= topk[k - 1]:
                        if n_log == log_i.shape[0]:
                            log_i, log_h = _njit_grow(log_i, log_h)
                        log_i[n_log] = e
                        log_h[n_log] = hops
                        n_log += 1
    # Return results trimmed to k plus the recorded data.
    out_i, out_d = _njit_drain_maxheap(wd, wi, nw)
    kk = min(k, out_i.shape[0])
    return out_i[:kk], out_d[:kk], hops, ndist, feats[:n_feat], log_i[:n_log], log_h[:n_log]


# Placeholder forest with no trees, passed in fixed mode when no model is needed.
_EMPTY_FOREST = {
    "feature": np.zeros(1, np.int32),
    "threshold": np.zeros(1, np.float64),
    "left": np.zeros(1, np.int32),
    "right": np.zeros(1, np.int32),
    "value": np.zeros(1, np.float64),
    "roots": np.zeros(0, np.int32),
}


# Python wrapper: fills in defaults (tombstones, mask, visited array) and runs the Numba search.
def run_es(
    index: HNSWIndex,
    vecs: np.ndarray,
    q: np.ndarray,
    k: int,
    ef_max: int,
    mode: int,
    forest: dict[str, np.ndarray] | None = None,
    *,
    probe: int = 16,
    probe_ef: int = 32,
    interval: int = 16,
    mult: float = 1.0,
    ef_min: int = 10,
    forced_ef: int = 0,
    selectivity: float = 1.0,
    deleted: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    record: bool = False,
    max_ckpt: int = 256,
    g: HNSWGraph | None = None,
):
    # Use the caller's captured graph, or the current one.
    g = g or index.g
    f = forest or _EMPTY_FOREST
    if deleted is None:
        deleted = np.zeros(g.capacity, dtype=np.uint8)
    use_mask = mask is not None
    if mask is None:
        mask = np.zeros(1, dtype=np.bool_)
    visited, tag = index._visited(g.capacity)
    return _njit_search_es(
        q, vecs, g.nbr0, g.upper, g.upper_row, g.entry, g.max_level, k, ef_max,
        deleted, mask, use_mask, visited, tag,
        mode, probe, probe_ef, interval, float(mult), ef_min, forced_ef, float(selectivity),
        math.log2(max(index.n, 1)),
        f["feature"], f["threshold"], f["left"], f["right"], f["value"], f["roots"],
        record, max_ckpt,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# labels + training
# ---------------------------------------------------------------------------


# Build training data by running generous searches on training queries with known true answers.
def collect(
    index: HNSWIndex,
    vecs: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Training rows for both variants from searches on training ``queries``."""
    k, ef_max, probe, probe_ef = cfg["k"], cfg["ef_max"], cfg["probe"], cfg["probe_ef"]
    interval, grid = cfg["interval"], cfg["ef_grid"]
    ck_x, ck_y, ck_q = [], [], []
    up_x, up_y = [], []
    # One query at a time; truth is its exact top-k from brute force.
    for qi, q in enumerate(queries):
        truth = set(gt[qi, :k].tolist())
        # checkpoint rows: features at every checkpoint of a generous search,
        # label = expansion at which the last reachable true neighbour showed up
        ids, _, hops, _, feats, log_i, log_h = run_es(
            index, vecs, q, k, ef_max, MODE_FIXED, interval=interval, record=True
        )
        # Find the hop at which each true neighbour first entered the top-k; the latest is the label.
        found = truth & set(ids.tolist())
        first: dict[int, int] = {}
        for node, h in zip(log_i.tolist(), log_h.tolist(), strict=True):
            first.setdefault(node, h)
        need = max(max((first.get(n, hops) for n in found), default=hops), 1)
        for j in range(feats.shape[0]):
            ck_x.append(feats[j])
            ck_y.append(math.log2(need))
            ck_q.append(qi)
        # upfront row: probe features, label = smallest ef (after the probe)
        # that recovers everything the generous search found
        # Try ef values from small to large and keep the first that finds everything the big search did.
        best, probe_feats = grid[-1], None
        for ef in grid:
            got, _, _, _, pf, _, _ = run_es(
                index, vecs, q, k, ef_max, MODE_UPFRONT, probe=probe, probe_ef=probe_ef,
                interval=interval, forced_ef=ef, record=True, max_ckpt=1,
            )  # fmt: skip
            if probe_feats is None and pf.shape[0]:
                probe_feats = pf[0]
            if len(truth & set(got.tolist())) >= len(found):
                best = ef
                break
        if probe_feats is not None:
            up_x.append(probe_feats)
            up_y.append(math.log2(best))
    # Return checkpoint rows (with query ids for grouping) and upfront rows.
    return {
        "ck_x": np.asarray(ck_x),
        "ck_y": np.asarray(ck_y),
        "ck_q": np.asarray(ck_q),
        "up_x": np.asarray(up_x),
        "up_y": np.asarray(up_y),
    }


# Train one LightGBM model to predict log2 of the budget, then flatten it for Numba.
def train(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, params: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Fit a LightGBM regressor on log2(budget); returns the flattened forest.

    A query-level split (``groups``) holds out 10% of training queries for early
    stopping, so rows from one query never land on both sides. Forests are kept
    small on purpose: every tree costs a few nanoseconds per prediction and the
    checkpoint variant predicts several times per query.
    """
    import lightgbm as lgb

    # Hold out about 10% of queries (whole queries, not rows) for early stopping of training.
    rng = np.random.default_rng(params.get("seed", 42))
    if groups is None:
        groups = np.arange(len(y))
    uq = np.unique(groups)
    val_q = set(rng.choice(uq, max(1, len(uq) // 10), replace=False).tolist())
    is_val = np.array([g in val_q for g in groups])
    # Small, deterministic LightGBM settings so the forest is cheap to evaluate.
    lgb_params = {
        "objective": "regression",
        "learning_rate": params.get("learning_rate", 0.1),
        "num_leaves": params.get("num_leaves", 15),
        "min_data_in_leaf": params.get("min_data_in_leaf", 50),
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "seed": params.get("seed", 42),
        "deterministic": True,
        "num_threads": params.get("num_threads", 8),
        "verbose": -1,
    }
    # Train with early stopping on the held-out queries, then dump the best model to flat arrays.
    dtrain = lgb.Dataset(x[~is_val], y[~is_val], feature_name=FEATURES)
    dval = lgb.Dataset(x[is_val], y[is_val], reference=dtrain)
    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=params.get("num_rounds", 60),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    return flatten(booster.dump_model(num_iteration=booster.best_iteration))


# Wraps a trained forest and its settings; this is what a collection loads when early stop is on.
class EarlyStopModel:
    """A trained budget model plus the search settings it was trained with."""

    # Keep the forest and settings; the "mode" in meta decides upfront vs checkpoint behaviour.
    def __init__(self, forest: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        self.forest = forest
        self.meta = meta
        self.mode = MODE_UPFRONT if meta["mode"] == "upfront" else MODE_CHECKPOINT

    # Load a saved model JSON from disk.
    @classmethod
    def load(cls, path: str | Path) -> EarlyStopModel:
        forest, meta = load_forest(Path(path))
        return cls(forest, meta)

    # Write the model to a JSON file.
    def save(self, path: str | Path) -> None:
        save_forest(Path(path), self.forest, self.meta)

    # Run one early-stopped search with the settings the model was trained with.
    def search(
        self,
        index: HNSWIndex,
        vecs: np.ndarray,
        q: np.ndarray,
        k: int,
        mult: float | None = None,
        deleted: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        selectivity: float = 1.0,
        g: HNSWGraph | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        m = self.meta
        ids, ds, hops, nd, *_ = run_es(
            index, vecs, q, k, m["ef_max"], self.mode, self.forest,
            probe=m["probe"], probe_ef=m["probe_ef"], interval=m["interval"],
            mult=m.get("multiplier", 1.0) if mult is None else mult,
            ef_min=m.get("ef_min", 10), selectivity=selectivity,
            deleted=deleted, mask=mask, g=g,
        )  # fmt: skip
        return ids, ds, {"hops": int(hops), "ndist": int(nd)}


# Offline pipeline: label the queries once, then train both the checkpoint and upfront models.
def fit_models(
    index: HNSWIndex,
    vecs: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, EarlyStopModel]:
    """Label ``queries`` and train both variants with settings from ``cfg``."""
    cfg = {"probe_ef": 32, **cfg}
    rows = collect(index, vecs, queries, gt, cfg)
    base = {
        "ef_max": cfg["ef_max"],
        "probe": cfg["probe"],
        "probe_ef": cfg["probe_ef"],
        "interval": cfg["interval"],
        "ef_min": cfg.get("ef_min", 10),
        "k": int(cfg["k"]),
        "features": FEATURES,
    }
    params = cfg.get("lightgbm", {})
    return {
        "checkpoint": EarlyStopModel(
            train(rows["ck_x"], rows["ck_y"], rows["ck_q"], params), {**base, "mode": "checkpoint"}
        ),
        "upfront": EarlyStopModel(
            train(rows["up_x"], rows["up_y"], None, params), {**base, "mode": "upfront"}
        ),
    }


# Command-line entry: open a collection, compute exact answers, train, and save one model.
def main() -> None:
    """Train a model for a collection's index from a queries file (offline).

    python -m ml.early_stop <collection_dir> <queries.npy> <config.yaml> <out.json>
    """
    # Imported here so the core library doesn't depend on the engine package.
    import sys

    from engine.collection import Collection
    from engine.config import load_yaml
    from index.flat import search_flat

    col_dir, qpath, cfg_path, out = sys.argv[1:5]
    cfg = load_yaml(cfg_path)["training"]
    col = Collection.open(col_dir)
    try:
        # Exact top-k per query by brute force serves as ground truth for labels.
        queries = np.load(qpath).astype(np.float32)
        vecs = col.store.array
        gt = np.stack([search_flat(vecs, col.n, q, cfg["k"], col.deleted)[0] for q in queries])
        models = fit_models(col.hnsw, vecs, queries, gt, cfg)
        models[cfg.get("mode", "checkpoint")].save(out)
        print(json.dumps({"saved": out}))
    finally:
        col.close()


# Run main() only when this file is executed as a script.
if __name__ == "__main__":
    main()
