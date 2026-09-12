import heapq

import numba
import numpy as np
import pytest

from index.flat import search_flat
from index.hnsw import (
    HNSWIndex,
    _njit_greedy,
    _njit_maxheap_pop,
    _njit_maxheap_push,
    _njit_minheap_pop,
    _njit_minheap_push,
    _njit_search_layer,
    _njit_select,
    node_levels,
    ref_greedy,
    ref_search_layer,
    ref_select,
)


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    pairs = zip(found, truth, strict=True)
    return float(np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in pairs]))


@pytest.fixture(scope="module")
def fixture_10k():
    """10k x 64 Gaussian mixture with wide clusters (hard enough that ef matters)."""
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(100, 64)).astype(np.float32) * 2
    x = centers[rng.integers(0, 100, 10_000)] + rng.normal(size=(10_000, 64))
    q = centers[rng.integers(0, 100, 200)] + rng.normal(size=(200, 64))
    x, q = x.astype(np.float32), q.astype(np.float32)
    gt = np.array([search_flat(x, len(x), qq, 10)[0] for qq in q])
    return x, q, gt


@pytest.fixture(scope="module")
def built_10k(fixture_10k):
    x, _, _ = fixture_10k
    h = HNSWIndex(x.shape[1], capacity=len(x))  # default params
    h.add(x, len(x))
    return h


# ---- kernel twins -----------------------------------------------------------


def test_heaps_match_heapq(rng):
    vals = rng.random(300).astype(np.float32)
    hd = np.empty(300, np.float32)
    hi = np.empty(300, np.int32)
    n = 0
    for i, v in enumerate(vals):
        n = _njit_minheap_push(hd, hi, n, v, i)
    out = []
    while n:
        out.append(hi[0])
        n = _njit_minheap_pop(hd, hi, n)
    assert out == [i for _, i in sorted((v, i) for i, v in enumerate(vals))]

    n = 0
    for i, v in enumerate(vals):
        n = _njit_maxheap_push(hd, hi, n, v, i)
    out = []
    while n:
        out.append(hi[0])
        n = _njit_maxheap_pop(hd, hi, n)
    ref = [(-v, i) for i, v in enumerate(vals)]
    heapq.heapify(ref)
    assert out == [heapq.heappop(ref)[1] for _ in range(len(vals))]


@pytest.fixture(scope="module")
def random_graph():
    rng = np.random.default_rng(3)
    n, d, m = 400, 16, 8
    vecs = rng.normal(size=(n, d)).astype(np.float32)
    g = np.full((n, m), -1, np.int32)
    for i in range(n):
        k = rng.integers(1, m + 1)
        g[i, :k] = rng.choice(n, size=k, replace=False)
    return vecs, g


def test_greedy_matches_reference(random_graph, rng):
    vecs, g = random_graph
    rowbase = np.zeros(1, np.int32)
    for _ in range(20):
        q = rng.normal(size=vecs.shape[1]).astype(np.float32)
        s = int(rng.integers(len(vecs)))
        d0 = float(((q - vecs[s]) ** 2).sum())
        cur, _, _ = _njit_greedy(q, vecs, g, rowbase, -1, s, np.float32(d0))
        ref_cur, _ = ref_greedy(q, vecs, g, rowbase, -1, s, d0)
        assert cur == ref_cur


def test_search_layer_matches_reference(random_graph, rng):
    vecs, g = random_graph
    rowbase = np.zeros(1, np.int32)
    for ef in (1, 5, 32):
        for _ in range(10):
            q = rng.normal(size=vecs.shape[1]).astype(np.float32)
            eps = rng.choice(len(vecs), size=3, replace=False).astype(np.int32)
            epd = np.array([((q - vecs[e]) ** 2).sum() for e in eps], np.float32)
            visited = np.zeros(len(vecs), np.uint32)
            ids, ds = _njit_search_layer(q, vecs, g, rowbase, -1, eps, epd, ef, visited, 1)
            ref_ids, ref_ds = ref_search_layer(q, vecs, g, rowbase, -1, list(eps), ef)
            assert list(ids) == ref_ids
            np.testing.assert_allclose(ds, ref_ds, rtol=1e-4)


def test_search_layer_row_offset(random_graph, rng):
    """Upper-layer addressing: node x reads row rowbase[x] + offset."""
    vecs, g = random_graph
    n = len(vecs)
    perm = rng.permutation(n).astype(np.int32)
    g2 = np.full((n + 1, g.shape[1]), -1, np.int32)
    g2[perm + 1] = g  # node x's row moved to perm[x] + 1
    q = rng.normal(size=vecs.shape[1]).astype(np.float32)
    eps = np.array([0], np.int32)
    epd = np.array([((q - vecs[0]) ** 2).sum()], np.float32)
    a, _ = _njit_search_layer(q, vecs, g, perm, -1, eps, epd, 16, np.zeros(n, np.uint32), 1)
    b, _ = _njit_search_layer(q, vecs, g2, perm, 1, eps, epd, 16, np.zeros(n, np.uint32), 1)
    assert list(a) == list(b)


def test_select_matches_reference(rng):
    vecs = rng.normal(size=(200, 8)).astype(np.float32)
    base = vecs[0]
    cand = np.arange(1, 200, dtype=np.int32)
    d = ((vecs[cand] - base) ** 2).sum(1).astype(np.float32)
    order = np.argsort(d)
    ci, cd = cand[order], d[order]
    for m in (4, 16, 64):
        out = np.empty(m, np.int32)
        cnt = _njit_select(vecs, ci, cd, len(ci), m, out)
        assert list(out[:cnt]) == ref_select(vecs, ci, cd, m)
    # fewer candidates than m -> keep them all
    out = np.empty(16, np.int32)
    assert _njit_select(vecs, ci[:5], cd[:5], 5, 16, out) == 5


def test_node_levels_distribution():
    lv = node_levels(np.arange(200_000), seed=42, ml=1 / np.log(16), max_level=12)
    # P(level >= 1) = 1/M for the standard HNSW level distribution
    assert abs((lv >= 1).mean() - 1 / 16) < 0.003
    assert np.array_equal(lv, node_levels(np.arange(200_000), 42, 1 / np.log(16), 12))
    assert not np.array_equal(lv, node_levels(np.arange(200_000), 43, 1 / np.log(16), 12))


# ---- index invariants -------------------------------------------------------


def test_recall_10k_default_params(built_10k, fixture_10k):
    x, q, gt = fixture_10k
    found = np.array([built_10k.search(x, qq, 10)[0] for qq in q])
    assert recall_at_k(found, gt) >= 0.90


def test_batch_and_single_agree(built_10k, fixture_10k):
    x, q, _ = fixture_10k
    batch_ids, _, _ = built_10k.search_batch(x, q, 10, 64)
    single = np.array([built_10k.search(x, qq, 10, 64)[0] for qq in q])
    assert np.array_equal(batch_ids, single)


def test_graph_invariants(built_10k):
    g, n = built_10k.g, built_10k.n
    nbr = g.nbr0[:n]
    assert nbr.max() < n
    # links are packed: no valid id after a -1
    valid = nbr >= 0
    assert np.all(valid[:, :-1] | ~valid[:, 1:])
    assert not np.any(nbr == np.arange(n)[:, None])  # no self loops
    assert valid.sum(1).min() >= 1
    assert g.levels[g.entry] == g.max_level


def test_build_is_deterministic_across_threads(fixture_10k):
    x = fixture_10k[0][:3000]
    before = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = HNSWIndex(x.shape[1], capacity=16)
        a.add(x, len(x))
        numba.set_num_threads(before)
        b = HNSWIndex(x.shape[1], capacity=16)
        b.add(x, len(x))
    finally:
        numba.set_num_threads(before)
    assert np.array_equal(a.g.nbr0[: a.n], b.g.nbr0[: b.n])
    assert np.array_equal(a.g.upper[: a.n_upper], b.g.upper[: b.n_upper])


def test_incremental_growth(fixture_10k):
    x, q, _ = fixture_10k
    h = HNSWIndex(x.shape[1], capacity=4)
    for end in (1, 2, 50, 700, 3000):
        h.add(x, end)
    assert h.n == 3000 and h.g.capacity >= 3000
    gt = np.array([search_flat(x, 3000, qq, 10)[0] for qq in q])
    found = np.array([h.search(x, qq, 10, 64)[0] for qq in q])
    assert recall_at_k(found, gt) >= 0.9


def test_tombstones_never_returned(built_10k, fixture_10k):
    x, q, _ = fixture_10k
    deleted = np.zeros(built_10k.g.capacity, np.uint8)
    first = np.array([built_10k.search(x, qq, 10)[0] for qq in q[:50]])
    deleted[np.unique(first[:, :3])] = 1
    for qq in q[:50]:
        ids, _, _ = built_10k.search(x, qq, 10, deleted=deleted)
        assert not deleted[ids].any()
        assert len(ids) == 10


def test_mask_restricts_results(built_10k, fixture_10k):
    x, q, _ = fixture_10k
    mask = np.zeros(built_10k.g.capacity, np.bool_)
    mask[::3] = True
    for qq in q[:30]:
        ids, _, _ = built_10k.search(x, qq, 10, ef=128, mask=mask)
        assert mask[ids].all() and len(ids) == 10


def test_max_hops_budget(built_10k, fixture_10k):
    x, q, _ = fixture_10k
    _, _, st = built_10k.search(x, q[0], 10, ef=200, max_hops=5)
    assert st["hops"] == 5


def test_state_roundtrip(built_10k, fixture_10k):
    x, q, _ = fixture_10k
    h2 = HNSWIndex(x.shape[1])
    h2.load_state(built_10k.state())
    for qq in q[:20]:
        assert np.array_equal(built_10k.search(x, qq, 10)[0], h2.search(x, qq, 10)[0])


def test_empty_index_search(fixture_10k):
    h = HNSWIndex(64)
    ids, _, _ = h.search(fixture_10k[0], fixture_10k[1][0], 10)
    assert len(ids) == 0
