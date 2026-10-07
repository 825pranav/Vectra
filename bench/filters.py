"""Filtered search: the planner vs every single fixed strategy.

    python -m bench.run configs/bench/sift1m_filters.yaml

SIFT1M base vectors get seeded synthetic attributes (``price`` uniform in
[0, 1000), ``category`` Zipf-distributed over 50 values, ``in_stock`` 70% true)
and are loaded into a real collection through the normal upsert path. Each test
query is paired with a predicate drawn from templates whose selectivity spans
~1e-5 .. 0.95, so the workload mixes very selective and very broad filters.

For every strategy -- forced post_filter, forced bitmap, forced brute_force,
and the planner choosing per query -- we time the full ``Collection.search``
call (parse, plan, bitmap, search, attribute check) single-threaded and score
recall@10 against the exact filtered top-10.

Threshold tuning (``mode: tune``, configs/bench/sift1m_filters_tune.yaml) runs
the planner itself over a grid of thresholds on a workload built from *learn*
queries and recommends the fastest setting whose mean recall meets
``min_recall``; configs/index/default.yaml ships that setting, and the test run
above always uses whatever default.yaml ships.
"""

from __future__ import annotations

# Stdlib, NumPy, shared bench helpers, the real Collection, filter parser/evaluator and planner.
import json
import time
from typing import Any

import numpy as np

from bench.common import (
    CACHE,
    COLORS,
    INK2,
    MUTED,
    RESULTS,
    direct_labels,
    new_figure,
    pinned_to_cpu,
    save_figure,
    save_json,
    system_info,
)
from bench.datasets import load as load_dataset
from engine.collection import Collection
from engine.config import default_config
from engine.filters import evaluate, parse
from engine.planner import Planner

# The four methods compared: three forced strategies plus the planner choosing per query.
STRATEGIES = ["post_filter", "bitmap", "brute_force", "planner"]
STRATEGY_COLORS = {
    "post_filter": COLORS["faiss-hnsw"],
    "bitmap": COLORS["vectra-hnsw-pq"],
    "brute_force": COLORS["faiss-ivfpq"],
    "planner": COLORS["vectra-hnsw"],
}


# Seeded synthetic tags per row: uniform price, Zipf-skewed category, 70% in stock.
def attributes(n: int, n_cat: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    price = rng.integers(0, 1000, n).astype(np.float64)
    zipf = 1.0 / np.arange(1, n_cat + 1)
    cat = rng.choice(n_cat, n, p=zipf / zipf.sum())
    stock = rng.random(n) < 0.7
    return price, cat, stock


# Load SIFT1M plus synthetic tags into a real Collection via upsert (cached after the first run).
def build_collection(cfg: dict[str, Any], base: np.ndarray) -> Collection:
    # Reuse the cached collection unless a rebuild was asked for; otherwise start from an empty
    # folder.
    path = CACHE / f"{cfg['dataset']}_filters_collection_seed{cfg['seed']}"
    if (path / Collection.CONFIG_FILE).exists() and not cfg.get("rebuild"):
        return Collection.open(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    # Generate the tags and create the collection with fsync and periodic snapshots off for bulk
    # load.
    price, cat, stock = attributes(len(base), cfg["n_categories"], cfg["seed"])
    over = {
        "initial_capacity": len(base),
        "durability.fsync": False,  # bulk load; the crash test covers durability
        "durability.snapshot_every": 0,
        **cfg.get("collection", {}),
    }
    col = Collection.create(path, "filters", base.shape[1], over)
    t = time.perf_counter()
    step = 100_000
    # Upsert in chunks of 100k rows through the normal write path.
    for s in range(0, len(base), step):
        e = min(s + step, len(base))
        attrs = [
            {"price": float(price[i]), "category": f"c{cat[i]}", "in_stock": bool(stock[i])}
            for i in range(s, e)
        ]
        col.upsert([str(i) for i in range(s, e)], base[s:e], attrs)
    col.meta.refresh_stats()
    print(f"loaded {len(base)} records in {time.perf_counter() - t:.0f}s")
    col.close()  # snapshot, so later runs open without a replay
    return Collection.open(path)


# Build n random filter strings from templates so selectivity covers very rare to very broad.
def predicates(rng: np.random.Generator, n: int, n_cat: int) -> list[str]:
    """Templates spanning selectivities from ~1e-5 to ~0.95."""
    out = []
    for _ in range(n):
        t = int(rng.integers(0, 8))
        p = int(rng.integers(0, 1000))
        c = int(rng.integers(0, n_cat))
        if t == 0:
            out.append(f"price < {int(rng.integers(500, 951))}")  # 0.5 .. 0.95
        elif t == 1:
            out.append(f"in_stock == true AND price >= {int(rng.integers(0, 300))}")  # ~0.5-0.7
        elif t == 2:
            out.append(f"price < {int(rng.integers(50, 400))}")  # 0.05 .. 0.4
        elif t == 3:
            out.append(f"category == 'c{int(rng.integers(0, 6))}'")  # zipf head: ~0.03-0.2
        elif t == 4:
            out.append(f"category == 'c{c}' AND in_stock == true")  # ~0.001-0.15
        elif t == 5:
            out.append(f"price < {int(rng.integers(1, 20))}")  # 0.001 .. 0.02
        elif t == 6:
            out.append(f"price == {p}")  # 0.001
        else:
            out.append(f"price == {p} AND category == 'c{c}'")  # ~1e-5 .. 1e-4
    return out


# Exact filtered top-k per query, plus each filter's true selectivity: the recall answer key.
def filtered_gt(col: Collection, Q: np.ndarray, preds: list[str], k: int):
    """Exact filtered top-k (float64 BLAS over all rows, non-matching masked out)."""
    base = np.asarray(col.store.array[: col.n], dtype=np.float64)
    norms = (base * base).sum(1)
    gts, sels = [], []
    # Distances for 32 queries at a time, then mask out rows that fail each query's filter.
    for s in range(0, len(Q), 32):
        dist = norms[:, None] - 2.0 * (base @ Q[s : s + 32].astype(np.float64).T)
        for j, text in enumerate(preds[s : s + 32]):
            m = evaluate(parse(text), col.meta.columns, col.n)
            sels.append(float(m.mean()))
            d = np.where(m, dist[:, j], np.inf)
            # Fewer than k rows may match; an empty filter gives an empty answer.
            kk = min(k, int(m.sum()))
            if kk == 0:
                gts.append(np.empty(0, np.int64))
                continue
            part = np.argpartition(d, kk - 1)[:kk]
            gts.append(part[np.argsort(d[part], kind="stable")])
    return gts, np.asarray(sels)


# Time Collection.search for one strategy over all queries; also record recall and chosen strategy.
def measure(col, Q, preds, gts, k, ef, strategy, warmup, runs) -> dict[str, np.ndarray]:
    force = None if strategy == "planner" else strategy
    n = len(Q)
    # Untimed warmup on up to 200 queries.
    for _ in range(warmup):
        for i in range(min(n, 200)):
            col.search(Q[i], k, preds[i], ef, strategy=force)
    lat = np.empty((runs, n))
    recall = np.zeros(n)
    chosen = []
    # Timed passes; recall and the planner's choice are only recorded on the first pass.
    for r in range(runs):
        for i in range(n):
            t = time.perf_counter()
            res = col.search(Q[i], k, preds[i], ef, strategy=force)
            lat[r, i] = time.perf_counter() - t
            if r == 0:
                want = set(gts[i].tolist())
                got = {int(x) for x in res.ids}
                recall[i] = len(want & got) / len(want) if want else 1.0
                chosen.append(res.strategy)
    return {"lat_ms": np.median(lat, axis=0) * 1e3, "recall": recall, "chosen": chosen}


# Overall mean/p99 latency and recall per strategy, then the same split into selectivity buckets.
def summarize(sels, results, buckets) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": {}, "buckets": []}
    for s, r in results.items():
        out["overall"][s] = {
            "mean_ms": float(r["lat_ms"].mean()),
            "p99_ms": float(np.percentile(r["lat_ms"], 99)),
            "recall": float(r["recall"].mean()),
        }
    # Group queries by selectivity range and average each strategy within the group.
    for lo, hi in zip(buckets, buckets[1:], strict=False):
        idx = np.flatnonzero((sels >= lo) & (sels < hi))
        if not idx.size:
            continue
        row = {"lo": lo, "hi": hi, "queries": int(idx.size)}
        for s, r in results.items():
            row[s] = {
                "mean_ms": float(r["lat_ms"][idx].mean()),
                "recall": float(r["recall"][idx].mean()),
            }
        out["buckets"].append(row)
    return out


# Build a query workload: queries, random filters, exact answers and selectivities.
def _workload(col, queries, cfg, seed_offset):
    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    Q = queries[: cfg["n_queries"]]
    preds = predicates(rng, len(Q), cfg["n_categories"])
    gts, sels = filtered_gt(col, Q, preds, int(cfg["k"]))
    return Q, preds, gts, sels


# Swap the planner's thresholds on a live collection by building a new Planner.
def _use_thresholds(col: Collection, planner_cfg: dict[str, Any]) -> None:
    col.cfg["planner"] = planner_cfg
    col.planner = Planner(col.cfg)


# Experiment entry: either tune planner thresholds on learn queries, or test all strategies.
def run(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    import numba

    # Set threads, load data, build or open the collection, and pin to one CPU for timing.
    numba.set_num_threads(int(cfg["threads"]))
    ds = load_dataset(cfg["dataset"])
    col = build_collection(cfg, ds["base"])
    shipped = default_config()["planner"]
    k, ef, w, r = int(cfg["k"]), int(cfg["ef"]), cfg["warmup"], cfg["runs"]
    out: dict[str, Any] = {"experiment": "filters", "config": cfg, "system": system_info(1)}
    pin = pinned_to_cpu(cfg.get("latency_cpu"))
    pin.__enter__()
    try:
        # Tune mode: grid-search the two planner thresholds and recommend the fastest that meets
        # min_recall.
        if cfg.get("mode") == "tune":
            # the planner itself, over a grid of thresholds, on LEARN queries only
            Q, preds, gts, sels = _workload(col, ds["learn"], cfg, 1)
            grid = []
            for bm in cfg["grid"]["brute_force_max_selectivity"]:
                for pm in cfg["grid"]["post_filter_min_selectivity"]:
                    if pm <= bm:
                        continue
                    _use_thresholds(col, {**shipped, "brute_force_max_selectivity": bm,
                                          "post_filter_min_selectivity": pm})  # fmt: skip
                    res = measure(col, Q, preds, gts, k, ef, "planner", w, r)
                    row = {"brute_force_max_selectivity": bm, "post_filter_min_selectivity": pm,
                           "mean_ms": float(res["lat_ms"].mean()),
                           "recall": float(res["recall"].mean())}  # fmt: skip
                    grid.append(row)
                    print(f"[tune] brute<={bm:<6} post>={pm:<5} mean={row['mean_ms']:.3f}ms "
                          f"recall={row['recall']:.4f}", flush=True)  # fmt: skip
            ok = [g for g in grid if g["recall"] >= cfg["min_recall"]]
            out["grid"] = grid
            out["recommended"] = min(ok, key=lambda g: g["mean_ms"]) if ok else None
            print("recommended:", out["recommended"])
            save_json(name, out)
            return out
        # Test mode: use the shipped thresholds and run every strategy on held-out test queries.
        _use_thresholds(col, shipped)
        out["planner_config"] = shipped
        Q, preds, gts, sels = _workload(col, ds["query"], cfg, 2)
        results = {}
        for s in STRATEGIES:
            results[s] = measure(col, Q, preds, gts, k, ef, s, w, r)
            print(f"[test] {s:12s} mean={results[s]['lat_ms'].mean():7.3f}ms "
                  f"recall={results[s]['recall'].mean():.4f}", flush=True)  # fmt: skip
        # Summarise, count how often the planner picked each method, then save and plot.
        summary = summarize(sels, results, cfg["buckets"])
        chosen = results["planner"]["chosen"]
        summary["planner_choices"] = {c: chosen.count(c) for c in sorted(set(chosen))}
        out["test"] = summary
        save_json(name, out)
        plot(out, name, cfg.get("title", name))
    # Always unpin the CPU and close the collection.
    finally:
        pin.__exit__(None, None, None)
        col.close()
    return out


# Plot latency vs selectivity bucket for each strategy, with rings where recall is below 0.9.
def plot(out: dict[str, Any], name: str, title: str) -> None:
    fig, ax = new_figure()
    buckets = out["test"]["buckets"]
    x = [np.sqrt(max(b["lo"], 1e-6) * b["hi"]) for b in buckets]  # geometric bucket centre
    anchors = []
    # One line per strategy, labelled at its last point.
    for s in STRATEGIES:
        y = [b[s]["mean_ms"] for b in buckets]
        anchors.append(((x[-1], y[-1]), s, STRATEGY_COLORS[s]))
        ax.plot(x, y, color=STRATEGY_COLORS[s], linewidth=2.6 if s == "planner" else 2,
                marker="o", markersize=5, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                label=s)  # fmt: skip
        low = [(xi, yi) for xi, yi, b in zip(x, y, buckets, strict=True) if b[s]["recall"] < 0.9]
        if low:  # a hollow ring marks buckets where the strategy misses the recall bar
            ax.scatter(*zip(*low, strict=True), s=90, facecolors="none",
                       edgecolors=STRATEGY_COLORS[s], linewidths=1.2, zorder=5)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(right=x[-1] * 8)  # room for the end-of-line labels
    direct_labels(ax, anchors)
    ax.set_xlabel("filter selectivity (fraction of records matching)")
    ax.set_ylabel("mean latency per query (ms, 1 thread, log scale)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.text(0.99, 0.98, "open ring = recall@10 below 0.9", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color=MUTED)  # fmt: skip
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    save_figure(fig, name)


# Redraw the plot from saved results.
def replot(cfg: dict[str, Any], name: str) -> None:
    plot(json.loads((RESULTS / f"{name}.json").read_text()), name, cfg.get("title", name))
