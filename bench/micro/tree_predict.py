"""Cost of one in-kernel prediction vs forest size.

    python -m bench.micro.tree_predict

Each tree walk is a chain of dependent loads and data-dependent branches, so a
prediction costs roughly (trees x depth) cache-resident steps. This sizes the
budget model: a checkpoint model is evaluated several times per query, and a
query is only ~100-200 us.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

from ml.trees import _njit_predict, flatten


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 10))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 + 0.1 * rng.normal(size=20000)
    for rounds, leaves in [(400, 31), (200, 31), (100, 15), (60, 15), (40, 7), (20, 7)]:
        params = {"objective": "regression", "num_leaves": leaves, "verbose": -1, "seed": 0}
        f = flatten(lgb.train(params, lgb.Dataset(x, y), rounds).dump_model())
        args = (f["feature"], f["threshold"], f["left"], f["right"], f["value"], f["roots"])
        _njit_predict(x[0], *args)
        t = time.perf_counter()
        for i in range(20000):
            _njit_predict(x[i], *args)
        us = (time.perf_counter() - t) / 20000 * 1e6
        print(f"{rounds:4d} trees x {leaves:2d} leaves: {us:6.2f} us per prediction "
              f"(incl. ~0.3 us Python call)")  # fmt: skip


if __name__ == "__main__":
    main()
