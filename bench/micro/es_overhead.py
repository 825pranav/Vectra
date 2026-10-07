"""Where does the early-stop kernel spend its time? (SIFT1M, learn queries)

    python -m bench.micro.es_overhead

Compares the plain query kernel with the early-stop kernel doing the *same*
traversal (fixed mode, same ef), then with a wide result heap, isolating
per-call overhead from per-hop overhead.
"""

from __future__ import annotations

# Timer, NumPy, the cache folder, SIFT1M loader, HNSW index and early-stop kernel modes.
import time

import numpy as np

from bench.common import CACHE
from bench.datasets.sift1m import load
from index.hnsw import HNSWIndex
from ml.early_stop import MODE_CHECKPOINT, MODE_FIXED, MODE_UPFRONT, run_es


# Time a per-query function: warm up, then take the best of a few runs; returns us/query and hops.
def timeit(fn, qs, reps=3):
    # Warm up the JIT and caches on the first 200 queries.
    for q in qs[:200]:
        fn(q)
    best = np.inf
    hops = 0
    # Repeat the full pass and keep the fastest, to cut timing noise.
    for _ in range(reps):
        t = time.perf_counter()
        hops = 0
        for q in qs:
            hops += fn(q)
        best = min(best, (time.perf_counter() - t) / len(qs))
    return best * 1e6, hops / len(qs)


# Compare plain HNSW search with the early-stop kernel under different modes on 2000 learn queries.
def main() -> None:
    # Load SIFT1M and the prebuilt cached HNSW graph instead of rebuilding it.
    ds = load()
    x, qs = ds["base"], ds["learn"][:2000]
    h = HNSWIndex(128, M=16, ef_construction=200, capacity=len(x))
    with np.load(CACHE / "sift1m_vectra_hnsw_M16_efc200_seed42.npz") as z:
        h.load_state({k: z[k] for k in z.files})
    # A one-leaf "forest" that always predicts log2(66), i.e. a constant budget of 66 hops.
    const66 = {
        "feature": np.zeros(1, np.int32), "threshold": np.zeros(1), "left": np.zeros(1, np.int32),
        "right": np.zeros(1, np.int32), "value": np.array([np.log2(66)]),
        "roots": np.array([~0], np.int32),
    }  # fmt: skip
    # Each row: a label and a function that runs one query and returns its hop count.
    rows = [
        ("plain search, ef=64", lambda q: h.search(x, q, 10, 64)[2]["hops"]),
        ("early-stop kernel, fixed ef=64", lambda q: run_es(h, x, q, 10, 64, MODE_FIXED)[2]),
        (
            "upfront, forced ef=64 after 16-hop probe at 512",
            lambda q: run_es(h, x, q, 10, 512, MODE_UPFRONT, probe=16, forced_ef=64)[2],
        ),
        (
            "checkpoint, constant budget 66 hops, ef_max 512",
            lambda q: run_es(h, x, q, 10, 512, MODE_CHECKPOINT, const66, interval=16)[2],
        ),
        (
            "checkpoint, constant budget 66 hops, ef_max 64",
            lambda q: run_es(h, x, q, 10, 64, MODE_CHECKPOINT, const66, interval=16)[2],
        ),
        ("plain search, ef=512", lambda q: h.search(x, q, 10, 512)[2]["hops"]),
        ("early-stop kernel, fixed ef=512", lambda q: run_es(h, x, q, 10, 512, MODE_FIXED)[2]),
    ]
    # Time every variant and print cost per query and per hop.
    for name, fn in rows:
        us, hops = timeit(fn, qs)
        print(f"{name:45s} {us:7.1f} us/query  {hops:6.1f} hops  {us / hops:5.2f} us/hop")


if __name__ == "__main__":
    main()
