"""Recall@k vs QPS curves for vectra and FAISS engines on one dataset.

    python -m bench.run configs/bench/sift1m_hnsw.yaml

Each engine is built once (cached under data/cache, keyed by its parameters),
then swept over its search knob (ef for HNSW, nprobe for IVF). Per operating
point we record: recall@k, batch QPS at the configured thread count, and
single-thread per-query latency (mean, p50, p99). FAISS is used here, and only
here, as a baseline.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numba
import numpy as np

from bench.common import (
    CACHE,
    COLORS,
    INK2,
    MUTED,
    RESULTS,
    direct_labels,
    median,
    new_figure,
    pinned_to_cpu,
    recall_at_k,
    save_figure,
    save_json,
    system_info,
    timed,
)
from bench.datasets import load as load_dataset
from index.hnsw import HNSWIndex


def _warm_compile_hnsw(dim: int, M: int) -> None:
    """Trigger (or load cached) JIT compilation outside any timed region."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, dim)).astype(np.float32)
    h = HNSWIndex(dim, M=M, ef_construction=32, serial_warmup=256)
    h.add(x, len(x))
    h.search(x, x[0], 10)
    h.search_batch(x, x[:64], 10, 16)


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------


class Engine:
    param = "ef"

    def __init__(self, spec: dict[str, Any], ds: dict[str, np.ndarray], dataset: str) -> None:
        self.spec = spec
        self.name = spec["name"]
        self.ds = ds
        self.dataset = dataset
        self.base = ds["base"]
        self.dim = self.base.shape[1]

    def cache_stem(self) -> str:
        raise NotImplementedError

    def build(self, rebuild: bool) -> dict[str, Any]:
        raise NotImplementedError

    def batch(self, Q: np.ndarray, k: int, p: int) -> np.ndarray:
        raise NotImplementedError

    def single(self, q: np.ndarray, k: int, p: int) -> np.ndarray:
        raise NotImplementedError

    def single_thread(self):
        """Context for latency runs (FAISS must drop to one OpenMP thread)."""
        return _Null()

    def _meta_path(self):
        return CACHE / f"{self.cache_stem()}.json"


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class SiftdbHNSW(Engine):
    def cache_stem(self) -> str:
        s = self.spec
        seed = s.get("seed", 42)
        return f"{self.dataset}_vectra_hnsw_M{s['M']}_efc{s['ef_construction']}_seed{seed}"

    def build(self, rebuild: bool) -> dict[str, Any]:
        s = self.spec
        self.index = HNSWIndex(
            self.dim,
            M=s["M"],
            ef_construction=s["ef_construction"],
            seed=s.get("seed", 42),
            capacity=len(self.base),
        )
        path = CACHE / f"{self.cache_stem()}.npz"
        if path.exists() and self._meta_path().exists() and not rebuild:
            with np.load(path) as z:
                self.index.load_state({k: z[k] for k in z.files})
            meta = json.loads(self._meta_path().read_text())
        else:
            _warm_compile_hnsw(self.dim, s["M"])
            t = time.perf_counter()
            self.index.add(self.base, len(self.base))
            meta = {"build_s": time.perf_counter() - t, "build_threads": numba.get_num_threads()}
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(path, **self.index.state())
            self._meta_path().write_text(json.dumps(meta))
        _warm_compile_hnsw(self.dim, s["M"])
        meta["index_bytes"] = self.index.nbytes() + self.base.nbytes
        meta["graph_bytes"] = self.index.nbytes()
        return meta

    def batch(self, Q, k, p):
        return self.index.search_batch(self.base, Q, k, p)[0]

    def single(self, q, k, p):
        return self.index.search(self.base, q, k, p)[0]


class SiftdbHNSWPQ(SiftdbHNSW):
    """Same graph as vectra-hnsw, traversed on PQ codes, exact re-rank of
    ``rerank_factor * ef`` candidates from the full-precision vectors."""

    def build(self, rebuild: bool) -> dict[str, Any]:
        from index.pq import ProductQuantizer

        meta = super().build(rebuild)
        s = self.spec
        stem = f"{self.dataset}_vectra_pq_m{s['m']}_n{s['train_size']}_seed{s.get('seed', 42)}"
        path = CACHE / f"{stem}.npz"
        self.pq = ProductQuantizer(self.dim, m=s["m"], iters=s.get("kmeans_iters", 20),
                                   seed=s.get("seed", 42))  # fmt: skip
        if path.exists() and not rebuild:
            with np.load(path) as z:
                self.pq.codebooks, self.codes = z["codebooks"], z["codes"]
            pq_meta = json.loads((CACHE / f"{stem}.json").read_text())
        else:
            rng = np.random.default_rng(s.get("seed", 42))
            sample = self.base[rng.choice(len(self.base), s["train_size"], replace=False)]
            t = time.perf_counter()
            self.pq.train(sample)
            t_train = time.perf_counter() - t
            t = time.perf_counter()
            self.codes = self.pq.encode(self.base)
            pq_meta = {"pq_train_s": t_train, "pq_encode_s": time.perf_counter() - t}
            np.savez(path, codebooks=self.pq.codebooks, codes=self.codes)
            (CACHE / f"{stem}.json").write_text(json.dumps(pq_meta))
        meta.update(pq_meta)
        meta["build_s"] = meta["build_s"] + pq_meta["pq_train_s"] + pq_meta["pq_encode_s"]
        meta["code_bytes"] = int(self.codes.nbytes + self.pq.nbytes())
        meta["vector_bytes_float"] = int(self.base.nbytes)
        # full-precision vectors stay on disk (memmap) and are only touched to re-rank
        meta["index_bytes"] = meta["graph_bytes"] + meta["code_bytes"]
        self.factor = int(s.get("rerank_factor", 2))
        return meta

    def _r(self, p: int) -> int:
        return max(10, self.factor * int(p))

    def batch(self, Q, k, p):
        cb = self.pq.codebooks
        return self.index.search_pq_batch(self.base, cb, self.codes, Q, k, p, self._r(p))[0]

    def single(self, q, k, p):
        cb = self.pq.codebooks
        return self.index.search_pq(self.base, cb, self.codes, q, k, p, self._r(p))[0]


class _FaissSingle:
    def __init__(self, threads: int) -> None:
        self.threads = threads

    def __enter__(self):
        import faiss

        faiss.omp_set_num_threads(1)

    def __exit__(self, *a):
        import faiss

        faiss.omp_set_num_threads(self.threads)
        return False


class FaissHNSW(Engine):
    def cache_stem(self) -> str:
        s = self.spec
        return f"{self.dataset}_faiss_hnsw_M{s['M']}_efc{s['ef_construction']}"

    def build(self, rebuild: bool) -> dict[str, Any]:
        import faiss

        s = self.spec
        path = CACHE / f"{self.cache_stem()}.faiss"
        if path.exists() and self._meta_path().exists() and not rebuild:
            self.index = faiss.read_index(str(path))
            meta = json.loads(self._meta_path().read_text())
        else:
            self.index = faiss.IndexHNSWFlat(self.dim, s["M"])
            self.index.hnsw.efConstruction = s["ef_construction"]
            t = time.perf_counter()
            self.index.add(self.base)
            meta = {
                "build_s": time.perf_counter() - t,
                "build_threads": faiss.omp_get_max_threads(),
            }
            CACHE.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(path))
            self._meta_path().write_text(json.dumps(meta))
        meta["index_bytes"] = int(faiss.serialize_index(self.index).nbytes)
        return meta

    def set(self, p):
        self.index.hnsw.efSearch = int(p)

    def batch(self, Q, k, p):
        self.set(p)
        return self.index.search(Q, k)[1]

    def single(self, q, k, p):
        self.set(p)
        return self.index.search(q[None, :], k)[1][0]

    def single_thread(self):
        import faiss

        return _FaissSingle(faiss.omp_get_max_threads())


class FaissIVFPQ(Engine):
    param = "nprobe"

    def cache_stem(self) -> str:
        s = self.spec
        refine = f"_refine{s['refine_k_factor']}" if s.get("refine_k_factor") else ""
        return f"{self.dataset}_faiss_ivf{s['nlist']}_pq{s['m']}{refine}_seed{s.get('seed', 42)}"

    def build(self, rebuild: bool) -> dict[str, Any]:
        import faiss

        s = self.spec
        path = CACHE / f"{self.cache_stem()}.faiss"
        if path.exists() and self._meta_path().exists() and not rebuild:
            self.index = faiss.read_index(str(path))
            meta = json.loads(self._meta_path().read_text())
        else:
            key = f"IVF{s['nlist']},PQ{s['m']}"
            if s.get("refine_k_factor"):
                key += ",RFlat"
            self.index = faiss.index_factory(self.dim, key)
            rng = np.random.default_rng(s.get("seed", 42))
            train = self.base[rng.choice(len(self.base), s["train_size"], replace=False)]
            t = time.perf_counter()
            self.index.train(train)
            self.index.add(self.base)
            meta = {
                "build_s": time.perf_counter() - t,
                "build_threads": faiss.omp_get_max_threads(),
            }
            CACHE.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(path))
            self._meta_path().write_text(json.dumps(meta))
        if s.get("refine_k_factor"):
            # IndexRefineFlat keeps full float vectors in RAM for the exact re-rank
            self._owner = self.index  # downcast wrappers do not own the C++ object
            self.index = faiss.downcast_index(self.index)
            self.index.k_factor = float(s["refine_k_factor"])
            self.ivf = faiss.extract_index_ivf(faiss.downcast_index(self.index.base_index))
        else:
            self.ivf = faiss.extract_index_ivf(self.index)
        meta["index_bytes"] = int(faiss.serialize_index(self.index).nbytes)
        return meta

    def batch(self, Q, k, p):
        self.ivf.nprobe = int(p)
        return self.index.search(Q, k)[1]

    def single(self, q, k, p):
        self.ivf.nprobe = int(p)
        return self.index.search(q[None, :], k)[1][0]

    def single_thread(self):
        import faiss

        return _FaissSingle(faiss.omp_get_max_threads())


ENGINES: dict[str, type[Engine]] = {
    "vectra_hnsw": SiftdbHNSW,
    "vectra_hnsw_pq": SiftdbHNSWPQ,
    "faiss_hnsw": FaissHNSW,
    "faiss_ivfpq": FaissIVFPQ,
}


def register(kind: str, cls: type[Engine]) -> None:
    ENGINES[kind] = cls


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def measure_point(eng: Engine, Q, gt, k, p, warmup, runs, n_lat, cpu=None) -> dict[str, Any]:
    times, ids = timed(lambda: eng.batch(Q, k, p), warmup, runs)
    lat_q = Q[:n_lat]
    with eng.single_thread(), pinned_to_cpu(cpu):
        for _ in range(warmup):
            for q in lat_q[:200]:
                eng.single(q, k, p)
        per_run = np.empty((runs, len(lat_q)))
        for r in range(runs):
            for i, q in enumerate(lat_q):
                t = time.perf_counter()
                eng.single(q, k, p)
                per_run[r, i] = time.perf_counter() - t
    lat = np.median(per_run, axis=0) * 1e3  # ms, per query, median over runs
    return {
        eng.param: p,
        "recall": recall_at_k(np.asarray(ids), gt, k),
        "qps": len(Q) / median(times),
        "batch_s_runs": times,
        "lat_mean_ms": float(lat.mean()),
        "lat_p50_ms": float(np.percentile(lat, 50)),
        "lat_p99_ms": float(np.percentile(lat, 99)),
    }


def qps_at_recall(points: list[dict[str, Any]], target: float, key: str = "qps") -> float | None:
    """Interpolate throughput at a target recall (log-linear between neighbours)."""
    pts = sorted(points, key=lambda r: r["recall"])
    for a, b in zip(pts, pts[1:], strict=False):
        if a["recall"] <= target <= b["recall"]:
            if b["recall"] == a["recall"]:
                return max(a[key], b[key])
            w = (target - a["recall"]) / (b["recall"] - a["recall"])
            return float(np.exp(np.log(a[key]) * (1 - w) + np.log(b[key]) * w))
    return None


def run(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    threads = int(cfg["threads"])
    numba.set_num_threads(threads)
    try:
        import faiss

        faiss.omp_set_num_threads(threads)
    except ImportError:
        pass
    ds = load_dataset(cfg["dataset"])
    k = int(cfg["k"])
    Q, gt = ds["query"], ds["gt"]
    if cfg.get("n_queries"):
        Q, gt = Q[: cfg["n_queries"]], gt[: cfg["n_queries"]]
    out: dict[str, Any] = {
        "experiment": "ann_curves",
        "config": cfg,
        "system": system_info(threads),
        "engines": [],
    }
    out["system"]["numba_threading_layer"] = _threading_layer()
    built = []
    for spec in cfg["engines"]:  # build everything first: fail fast, measure later
        eng = ENGINES[spec["kind"]](spec, ds, cfg["dataset"])
        print(f"[{eng.name}] building...", flush=True)
        meta = eng.build(bool(cfg.get("rebuild", False)))
        print(f"[{eng.name}] build {meta['build_s']:.1f}s, {meta['index_bytes'] / 2**20:.0f} MiB")
        eng.batch(Q[:32], k, spec["sweep"][0])  # smoke-test the search path too
        built.append((spec, eng, meta))
    for spec, eng, meta in built:
        points = []
        for p in spec["sweep"]:
            pt = measure_point(
                eng, Q, gt, k, p, cfg["warmup"], cfg["runs"], cfg["latency_queries"],
                cfg.get("latency_cpu"),
            )  # fmt: skip
            points.append(pt)
            print(
                f"[{eng.name}] {eng.param}={p:<5} recall={pt['recall']:.4f} "
                f"qps={pt['qps']:9.0f} p50={pt['lat_p50_ms']:.3f}ms p99={pt['lat_p99_ms']:.3f}ms",
                flush=True,
            )
        out["engines"].append({"name": eng.name, "kind": spec["kind"], **meta, "points": points})
    for target in cfg.get("recall_targets", []):
        out.setdefault("qps_at_recall", {})[str(target)] = {
            e["name"]: qps_at_recall(e["points"], target) for e in out["engines"]
        }
    save_json(name, out)
    replot(cfg, name)
    return out


def replot(cfg: dict[str, Any], name: str) -> None:
    out = json.loads((RESULTS / f"{name}.json").read_text())
    plot(out, name, cfg.get("title", name), int(cfg["k"]), cfg.get("recall_targets", []),
         cfg.get("plot_xmin", 0.8))  # fmt: skip


def _threading_layer() -> str | None:
    try:
        return numba.threading_layer()
    except ValueError:
        return None


def plot(
    out: dict[str, Any], name: str, title: str, k: int, targets: list[float], xmin: float = 0.8
) -> None:
    fig, ax = new_figure()
    engines = out["engines"]
    anchors = []
    for e in engines:
        pts = sorted(e["points"], key=lambda r: r["recall"])
        x = [p["recall"] for p in pts]
        y = [p["qps"] for p in pts]
        color = COLORS.get(e["name"], MUTED)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=e["name"])  # fmt: skip
        visible = [(a, b) for a, b in zip(x, y, strict=True) if a >= xmin]
        if visible:
            anchors.append((visible[0], e["name"], color))
    for t in targets:
        ax.axvline(t, color=MUTED, linewidth=0.8)
        ax.text(t, 1.0, f" recall {t}", transform=ax.get_xaxis_transform(), va="top",
                fontsize=8, color=MUTED)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlim(left=xmin, right=1.0)
    if len(engines) <= 4:
        direct_labels(ax, anchors)
    ax.set_xlabel(f"recall@{k}")
    ax.set_ylabel(f"queries / second ({out['system']['threads']} threads, log scale)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="best", labelcolor=INK2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    save_figure(fig, name)
