"""Learned early termination vs fixed ef, at equal recall.

    python -m bench.run configs/bench/sift1m_early_stop.yaml

1. Build (or load the cached) HNSW graph for the dataset.
2. Label ``n_train`` queries from the *learn* split (brute-force ground truth)
   and train both model variants. Test queries are never seen here.
3. On the held-out test queries, single-threaded, one query at a time:
   * fixed-ef baseline swept over ``fixed_ef``;
   * upfront and checkpoint variants swept over ``multipliers``.
4. Report mean latency (median of 5 timed passes, 3 warmup passes) and recall@k
   per point, and the latency reduction at each recall target, interpolating
   both curves at exactly that recall.
"""

from __future__ import annotations

# Stdlib for cache keys, JSON and timing; Numba thread control; the shared bench helpers;
# the HNSW wrapper from bench.ann; dataset loaders; and the learned early-stop models.
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numba
import numpy as np

from bench.ann import SiftdbHNSW
from bench.common import (
    CACHE,
    COLORS,
    INK2,
    MUTED,
    RESULTS,
    direct_labels,
    new_figure,
    pinned_to_cpu,
    recall_at_k,
    save_figure,
    save_json,
    system_info,
)
from bench.datasets import load as load_dataset
from bench.datasets.gt import cached_topk
from ml.early_stop import EarlyStopModel, fit_models


# Run each search config over all queries and return recall, latency and work stats per config.
def _measure_all(searches, Q, gt, k, warmup, runs) -> list[dict[str, Any]]:
    """Time every configuration, *interleaved*: each timed pass cycles through all
    of them, so thermal/turbo drift over the run hits every curve equally."""
    n = len(Q)
    # Warmup passes on up to 500 queries so JIT compilation is never timed.
    for _ in range(warmup):
        for fn in searches:
            for q in Q[: min(n, 500)]:
                fn(q)
    # Arrays for per-run timings, found ids, hop counts and distance counts.
    per = np.empty((len(searches), runs, n))
    found = np.full((len(searches), n, k), -1, dtype=np.int64)
    hops = np.zeros((len(searches), n))
    ndist = np.zeros((len(searches), n))
    # Timed passes; ids and stats are only recorded on the first pass (they never change).
    for r in range(runs):
        for c, fn in enumerate(searches):
            for i, q in enumerate(Q):
                t = time.perf_counter()
                ids, _, st = fn(q)
                per[c, r, i] = time.perf_counter() - t
                if r == 0:
                    found[c, i, : len(ids)] = ids
                    hops[c, i] = st["hops"]
                    ndist[c, i] = st["ndist"]
    # Turn the raw timings into one summary dict per config.
    out = []
    for c in range(len(searches)):
        lat = np.median(per[c], axis=0) * 1e3
        out.append(
            {
                "recall": recall_at_k(found[c], gt, k),
                "lat_mean_ms": float(np.median(per[c].mean(axis=1)) * 1e3),
                "lat_p50_ms": float(np.percentile(lat, 50)),
                "lat_p99_ms": float(np.percentile(lat, 99)),
                "hops_mean": float(hops[c].mean()),
                "ndist_mean": float(ndist[c].mean()),
            }
        )
    return out


# Interpolate a curve to get the latency (or other cost) at an exact recall target.
def latency_at_recall(
    points: list[dict[str, Any]], target: float, key: str = "lat_mean_ms"
) -> float | None:
    """Cost (``key``) at exactly ``target`` recall, log-linear between the two
    neighbouring points; the cheapest crossing if a curve is not monotone."""
    pts = sorted(points, key=lambda p: p[key])
    best = None
    # Check each pair of neighbouring points that brackets the target and keep the cheapest.
    for a, b in zip(pts, pts[1:], strict=False):
        lo, hi = sorted((a, b), key=lambda p: p["recall"])
        if lo["recall"] <= target <= hi["recall"] and hi["recall"] > lo["recall"]:
            w = (target - lo["recall"]) / (hi["recall"] - lo["recall"])
            v = float(np.exp(np.log(lo[key]) * (1 - w) + np.log(hi[key]) * w))
            best = v if best is None else min(best, v)
    return best


# Load the trained early-stop models from cache, or label learn queries and train them.
def _models(cfg, index, base, learn, dataset) -> dict[str, EarlyStopModel]:
    t = cfg["training"]
    # Cache key is a hash of dataset, index settings and training settings.
    key = hashlib.sha1(json.dumps([dataset, cfg["index"], t], sort_keys=True).encode()).hexdigest()
    paths = {
        m: CACHE / f"{dataset}_early_stop_{m}_{key[:10]}.json" for m in ("upfront", "checkpoint")
    }
    if all(p.exists() for p in paths.values()) and not cfg.get("rebuild"):
        return {m: EarlyStopModel.load(p) for m, p in paths.items()}
    # Ground truth for the learn split only, so test queries never influence training.
    q = learn[: t["n_train"]]
    gt = cached_topk(CACHE / f"{dataset}_learn_gt_k{t['k']}_n{len(q)}.npy", base, q, t["k"])
    t0 = time.perf_counter()
    models = fit_models(index, base, q, gt, t)
    print(f"trained early-stop models on {len(q)} learn queries in {time.perf_counter() - t0:.0f}s")
    for m, p in paths.items():
        models[m].save(p)
    return models


# Experiment entry: compare fixed-ef search with the two learned variants at equal recall.
def run(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    # Load the dataset and test queries, then build or load the HNSW graph and the models.
    numba.set_num_threads(int(cfg["threads"]))
    ds = load_dataset(cfg["dataset"])
    base, k = ds["base"], int(cfg["k"])
    Q, gt = ds["query"], ds["gt"][:, :k]
    if cfg.get("n_queries"):
        Q, gt = Q[: cfg["n_queries"]], gt[: cfg["n_queries"]]
    eng = SiftdbHNSW({"name": "vectra-hnsw", **cfg["index"]}, ds, cfg["dataset"])
    build = eng.build(bool(cfg.get("rebuild", False)))
    index = eng.index
    models = _models(cfg, index, base, ds["learn"], cfg["dataset"])
    w, r = int(cfg["warmup"]), int(cfg["runs"])

    # Build one search function per point: fixed ef values, then each variant at each multiplier.
    labels, searches = [], []
    for ef in cfg["fixed_ef"]:
        labels.append(("fixed-ef", {"ef": ef}))
        searches.append(lambda q, ef=ef: index.search(base, q, k, ef))
    for variant in ("upfront", "checkpoint"):
        for mult in cfg["multipliers"]:
            labels.append((variant, {"multiplier": mult}))
            searches.append(
                lambda q, m=mult, mod=models[variant]: mod.search(index, base, q, k, mult=m)
            )
    # Time everything pinned to one CPU, then group the points into a curve per method.
    with pinned_to_cpu(cfg.get("latency_cpu")):
        points = _measure_all(searches, Q, gt, k, w, r)
    curves: dict[str, list[dict[str, Any]]] = {"fixed-ef": [], "upfront": [], "checkpoint": []}
    for (variant, knob), pt in zip(labels, points, strict=True):
        curves[variant].append({**knob, **pt})
        print(f"[{variant}] {knob} recall={pt['recall']:.4f} mean={pt['lat_mean_ms']:.3f}ms "
              f"hops={pt['hops_mean']:.0f} ndist={pt['ndist_mean']:.0f}")  # fmt: skip

    # For each recall target, latency and distance-count savings of each variant vs fixed ef.
    summary = {}
    for target in cfg["recall_targets"]:
        base_lat = latency_at_recall(curves["fixed-ef"], target)
        base_nd = latency_at_recall(curves["fixed-ef"], target, "ndist_mean")
        row = {"fixed-ef_ms": base_lat, "fixed-ef_ndist": base_nd}
        for variant in ("upfront", "checkpoint"):
            v = latency_at_recall(curves[variant], target)
            nd = latency_at_recall(curves[variant], target, "ndist_mean")
            row[f"{variant}_ms"] = v
            row[f"{variant}_reduction_pct"] = (
                None if v is None or base_lat is None else 100.0 * (1 - v / base_lat)
            )
            row[f"{variant}_ndist_reduction_pct"] = (
                None if nd is None or base_nd is None else 100.0 * (1 - nd / base_nd)
            )
        summary[str(target)] = row
        print(f"recall {target}: {json.dumps(row)}")
    # Save results JSON and the plot.
    out = {
        "experiment": "early_stop",
        "config": cfg,
        "system": system_info(1),
        "build": build,
        "curves": curves,
        "latency_at_recall": summary,
    }
    save_json(name, out)
    plot(out, name, cfg.get("title", name), k, cfg["recall_targets"])
    return out


# Plot colour for each method.
VARIANT_COLORS = {"fixed-ef": COLORS["vectra-hnsw"], "upfront": COLORS["faiss-hnsw"],
                  "checkpoint": COLORS["vectra-hnsw-es"]}  # fmt: skip


# Draw recall vs latency curves for all methods, with vertical lines at the recall targets.
def plot(out, name, title, k, targets, xmin: float | None = None) -> None:
    fig, ax = new_figure()
    xmin = xmin if xmin is not None else out["config"].get("plot_xmin", 0.85)
    anchors = []
    # One line per method, sorted by latency, labelled at its last point.
    for label, pts in out["curves"].items():
        pts = sorted(pts, key=lambda p: p["lat_mean_ms"])
        x = [p["recall"] for p in pts]
        y = [p["lat_mean_ms"] * 1e3 for p in pts]  # microseconds
        color = VARIANT_COLORS[label]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=label)  # fmt: skip
        anchors.append(((x[-1], y[-1]), label, color))
    # Mark each recall target.
    for t in targets:
        ax.axvline(t, color=MUTED, linewidth=0.8)
        ax.text(t, 1.0, f" {t}", transform=ax.get_xaxis_transform(), va="top", fontsize=8,
                color=MUTED)  # fmt: skip
    ax.set_xlim(left=xmin, right=1.0)
    ax.set_yscale("log")
    direct_labels(ax, anchors)
    ax.set_xlabel(f"recall@{k}")
    ax.set_ylabel("mean latency per query (µs, 1 thread, log scale)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    save_figure(fig, name)


# Redraw the plot from a saved results JSON without re-running anything.
def replot(cfg: dict[str, Any], name: str) -> None:
    out = json.loads((RESULTS / f"{name}.json").read_text())
    plot(out, name, cfg.get("title", name), int(cfg["k"]), cfg["recall_targets"])


if __name__ == "__main__":
    # Allow running this file directly with a config path.
    import sys

    from engine.config import load_yaml

    c = Path(sys.argv[1])
    run(load_yaml(c), c.stem)
