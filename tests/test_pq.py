import numpy as np
import pytest

from index.flat import search_flat
from index.hnsw import HNSWIndex, _njit_rerank
from index.pq import (
    ProductQuantizer,
    _njit_adc_scan,
    _njit_encode,
    _njit_lut,
    kmeans,
    ref_adc_scan,
    ref_encode,
    ref_lut,
    search_pq_scan,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    centers = rng.normal(size=(64, 32)).astype(np.float32) * 3
    x = centers[rng.integers(0, 64, 6000)] + rng.normal(size=(6000, 32)) * 0.7
    q = centers[rng.integers(0, 64, 100)] + rng.normal(size=(100, 32)) * 0.7
    return x.astype(np.float32), q.astype(np.float32)


@pytest.fixture(scope="module")
def pq(data):
    x, _ = data
    p = ProductQuantizer(32, m=8, iters=15, seed=3)
    p.train(x)
    return p


def test_encode_matches_reference(pq, data):
    x, _ = data
    got = _njit_encode(x[:500], pq.codebooks)
    want = ref_encode(x[:500], pq.codebooks)
    # float32 vs float64 argmin can disagree only on near-exact ties
    assert (got == want).mean() > 0.999


def test_lut_matches_reference(pq, data):
    _, q = data
    np.testing.assert_allclose(_njit_lut(q[0], pq.codebooks), ref_lut(q[0], pq.codebooks),
                               rtol=1e-4, atol=1e-4)  # fmt: skip


def test_adc_scan_matches_reference(pq, data):
    x, q = data
    codes = pq.encode(x)
    lut = pq.lut(q[0])
    np.testing.assert_allclose(_njit_adc_scan(codes, lut, len(codes)),
                               ref_adc_scan(codes, lut, len(codes)), rtol=1e-5)  # fmt: skip


def test_adc_equals_distance_to_reconstruction(pq, data):
    """ADC(q, code) must equal the naive ||q - decode(code)||^2."""
    x, q = data
    codes = pq.encode(x[:300])
    recon = pq.decode(codes).astype(np.float64)
    for qq in q[:5]:
        naive = ((recon - qq.astype(np.float64)) ** 2).sum(1)
        adc = _njit_adc_scan(codes, pq.lut(qq), len(codes))
        np.testing.assert_allclose(adc, naive, rtol=1e-4, atol=1e-3)


def test_reconstruction_error_bounded(pq, data):
    x, _ = data
    recon = pq.decode(pq.encode(x))
    err = ((x - recon) ** 2).sum(1).mean()
    spread = ((x - x.mean(0)) ** 2).sum(1).mean()
    assert err < 0.1 * spread
    # and much better than a codebook of random data points (no training)
    untrained = ProductQuantizer(32, m=8, iters=0, seed=3)
    untrained.train(x)
    err0 = ((x - untrained.decode(untrained.encode(x))) ** 2).sum(1).mean()
    assert err < err0


def test_kmeans_is_deterministic_and_improves():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 4)).astype(np.float32)
    a = kmeans(x, 16, 10, np.random.default_rng(5))
    b = kmeans(x, 16, 10, np.random.default_rng(5))
    assert np.array_equal(a, b)
    c0 = kmeans(x, 16, 0, np.random.default_rng(5))

    def sse(c):
        return ((x[:, None] - c[None]) ** 2).sum(-1).min(1).sum()

    assert sse(a) < sse(c0)


def test_pq_scan_topk_recall(pq, data):
    x, q = data
    codes = pq.encode(x)
    hits = 0
    for qq in q:
        approx, _ = search_pq_scan(codes, len(codes), pq.lut(qq), 50)
        exact, _ = search_flat(x, len(x), qq, 10)
        hits += len(set(exact) & set(approx))
    assert hits / (10 * len(q)) > 0.9  # true top-10 inside the PQ top-50


def test_rerank_is_exact(data):
    x, q = data
    ids = np.arange(len(x), dtype=np.int32)
    got, gd = _njit_rerank(q[0], x, ids, 10)
    want, wd = search_flat(x, len(x), q[0], 10)
    assert list(got) == list(want)
    np.testing.assert_allclose(gd, wd, rtol=1e-5)


def test_hnsw_pq_recall_after_rerank(pq, data):
    x, q = data
    h = HNSWIndex(32, capacity=len(x))
    h.add(x, len(x))
    codes = pq.encode(x)
    gt = np.array([search_flat(x, len(x), qq, 10)[0] for qq in q])
    found = np.array(
        [h.search_pq(x, pq.codebooks, codes, qq, 10, ef=64, rerank=200)[0] for qq in q]
    )
    rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, gt, strict=True)])
    assert rec >= 0.9
    batch, _, _ = h.search_pq_batch(x, pq.codebooks, codes, q, 10, ef=64, rerank=200)
    assert np.array_equal(batch, found)
    # reranked distances are exact
    ids, ds, st = h.search_pq(x, pq.codebooks, codes, q[0], 10, ef=64, rerank=200)
    np.testing.assert_allclose(ds, ((x[ids] - q[0]) ** 2).sum(1), rtol=1e-4)
    assert st["reranked"] == 200


def test_invalid_split():
    with pytest.raises(ValueError):
        ProductQuantizer(30, m=8)
