import math

import lightgbm as lgb
import numpy as np
import pytest

from index.flat import search_flat
from index.hnsw import HNSWIndex
from ml.early_stop import (
    MODE_CHECKPOINT,
    MODE_FIXED,
    MODE_UPFRONT,
    EarlyStopModel,
    _njit_features,
    fit_models,
    ref_features,
    run_es,
)
from ml.trees import _njit_predict, flatten, load_forest, predict, ref_predict, save_forest


@pytest.fixture(scope="module")
def booster():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 10))
    y = np.sin(x[:, 0]) * 2 + x[:, 3] ** 2 - x[:, 7] + rng.normal(size=3000) * 0.1
    params = {"objective": "regression", "num_leaves": 15, "verbose": -1, "seed": 1,
              "deterministic": True}  # fmt: skip
    return lgb.train(params, lgb.Dataset(x, y), num_boost_round=60), x


def test_flattened_forest_matches_lightgbm(booster, tmp_path):
    b, x = booster
    forest = flatten(b.dump_model())
    want = b.predict(x[:500])
    got = np.array([predict(r, forest) for r in x[:500]])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    twin = np.array([ref_predict(r, forest) for r in x[:50]])
    np.testing.assert_allclose(twin, want[:50], rtol=1e-9, atol=1e-9)
    save_forest(tmp_path / "f.json", forest, {"a": 1})
    f2, meta = load_forest(tmp_path / "f.json")
    assert meta == {"a": 1}
    assert predict(x[0], f2) == pytest.approx(want[0], abs=1e-9)


def test_features_match_twin():
    topk = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    out = np.zeros(10)
    _njit_features(out, 8.0, topk, 6.0, 37, 500, 0.25, 0.5, 20.0)
    np.testing.assert_allclose(out, ref_features(8.0, topk, 6.0, 37, 500, 0.25, 0.5, 20.0),
                               rtol=1e-6)  # fmt: skip
    inf_topk = np.array([1.0, np.inf], dtype=np.float32)
    _njit_features(out, 8.0, inf_topk, 6.0, 3, 50, 0.0, 1.0, 10.0)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, ref_features(8.0, inf_topk, 6.0, 3, 50, 0.0, 1.0, 10.0),
                               rtol=1e-6)  # fmt: skip


@pytest.fixture(scope="module")
def graph():
    rng = np.random.default_rng(21)
    centers = rng.normal(size=(80, 32)).astype(np.float32) * 2
    x = (centers[rng.integers(0, 80, 8000)] + rng.normal(size=(8000, 32))).astype(np.float32)
    q = (centers[rng.integers(0, 80, 300)] + rng.normal(size=(300, 32))).astype(np.float32)
    h = HNSWIndex(32, capacity=len(x))
    h.add(x, len(x))
    gt = np.array([search_flat(x, len(x), qq, 10)[0] for qq in q])
    return h, x, q, gt


def test_fixed_mode_equals_plain_search(graph):
    h, x, q, _ = graph
    for qq in q[:40]:
        want, wd, _ = h.search(x, qq, 10, ef=48)
        got, gd, *_ = run_es(h, x, qq, 10, 48, MODE_FIXED)
        assert np.array_equal(got, want)
        np.testing.assert_allclose(gd, wd)


def test_upfront_forced_ef(graph):
    h, x, q, _ = graph
    for qq in q[:20]:
        full, *_ = run_es(h, x, qq, 10, 128, MODE_FIXED)
        same, *_ = run_es(h, x, qq, 10, 128, MODE_UPFRONT, probe=8, forced_ef=128)
        assert np.array_equal(full, same)
        _, _, hops_small, *_ = run_es(h, x, qq, 10, 128, MODE_UPFRONT, probe=8, forced_ef=10)
        _, _, hops_big, *_ = run_es(h, x, qq, 10, 128, MODE_FIXED)
        assert hops_small <= hops_big


def _constant_forest(v: float):
    return {
        "feature": np.zeros(1, np.int32),
        "threshold": np.zeros(1, np.float64),
        "left": np.zeros(1, np.int32),
        "right": np.zeros(1, np.int32),
        "value": np.array([v], np.float64),
        "roots": np.array([~0], np.int32),
    }


def test_checkpoint_budget_stops_on_time(graph):
    h, x, q, _ = graph
    forest = _constant_forest(math.log2(40))
    assert _njit_predict(np.zeros(10), *forest.values()) == pytest.approx(math.log2(40))
    _, _, hops, *_ = run_es(h, x, q[0], 10, 256, MODE_CHECKPOINT, forest, interval=16)
    assert hops == 40
    # a budget already exceeded at the first checkpoint stops right there
    _, _, hops, *_ = run_es(h, x, q[0], 10, 256, MODE_CHECKPOINT, _constant_forest(2.0),
                            interval=16)  # fmt: skip
    assert hops == 16


def test_record_mode_logs_topk_entries(graph):
    h, x, q, _ = graph
    ids, _, hops, _, feats, log_i, log_h = run_es(
        h, x, q[0], 10, 64, MODE_FIXED, interval=8, record=True
    )
    assert feats.shape[1] == 10 and 0 < feats.shape[0] <= hops // 8
    assert np.all(np.diff(feats[:, 5]) > 0)  # log2(hops) grows checkpoint to checkpoint
    assert set(ids.tolist()) <= set(log_i.tolist())  # every final top-k entry was logged
    assert np.all(np.diff(log_h) >= 0) and log_h.max() <= hops


def test_trained_models_beat_budget_floor(graph):
    h, x, q, gt = graph
    cfg = {"k": 10, "ef_max": 256, "probe": 8, "interval": 8,
           "ef_grid": [10, 16, 24, 32, 48, 64, 96, 128, 256],
           "lightgbm": {"num_rounds": 60, "num_threads": 2}}  # fmt: skip
    models = fit_models(h, x, q[:200], gt[:200], cfg)
    for name, m in models.items():
        assert isinstance(m, EarlyStopModel)
        found = np.array([m.search(h, x, qq, 10, mult=1.5)[0] for qq in q[200:]])
        rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, gt[200:], strict=True)])
        assert rec >= 0.85, (name, rec)
