"""Benchmark datasets: name -> {base, query, gt, learn}.

``query``/``gt`` are the held-out test split used for every reported number;
``learn`` is the only split any tuning or training may touch.
"""

from __future__ import annotations

import numpy as np


def load(name: str) -> dict[str, np.ndarray]:
    if name == "sift1m":
        from bench.datasets import sift1m

        return sift1m.load()
    if name == "msmarco300k":
        from bench.datasets import msmarco

        return msmarco.load()
    raise KeyError(f"unknown dataset {name!r}")
