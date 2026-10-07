"""Benchmark datasets: name -> {base, query, gt, learn}.

``query``/``gt`` are the held-out test split used for every reported number;
``learn`` is the only split any tuning or training may touch.
"""

from __future__ import annotations

# Only NumPy is needed here; each dataset module is imported lazily inside load().
import numpy as np


# Bench entry: dataset name in, dict of base/query/gt/learn arrays out (used by every experiment).
def load(name: str) -> dict[str, np.ndarray]:
    # Import each loader only when asked, so loading one dataset never pulls in the others.
    if name == "sift1m":
        from bench.datasets import sift1m

        return sift1m.load()
    if name == "msmarco300k":
        from bench.datasets import msmarco

        return msmarco.load()
    raise KeyError(f"unknown dataset {name!r}")
