import numpy as np

from index.flat import search_flat, search_subset


def brute_topk(vecs, q, k, dead=frozenset()):
    d = np.linalg.norm(vecs.astype(np.float64) - q.astype(np.float64), axis=1) ** 2
    order = [i for i in np.argsort(d, kind="stable") if i not in dead]
    return order[:k]


def test_exact_topk(small_data):
    q = small_data[123] + 0.01
    ids, dists = search_flat(small_data, len(small_data), q, 10)
    assert list(ids) == brute_topk(small_data, q, 10)
    assert np.all(np.diff(dists) >= 0)


def test_deleted_are_skipped(small_data):
    q = small_data[0]
    deleted = np.zeros(len(small_data), dtype=np.uint8)
    ids_before, _ = search_flat(small_data, len(small_data), q, 5, deleted)
    deleted[ids_before[0]] = 1
    ids_after, _ = search_flat(small_data, len(small_data), q, 5, deleted)
    assert ids_before[0] not in ids_after
    assert list(ids_after) == brute_topk(small_data, q, 5, {ids_before[0]})


def test_k_larger_than_n(small_data):
    ids, dists = search_flat(small_data, 7, small_data[0], 50)
    assert len(ids) == 7 == len(dists)


def test_empty_index(small_data):
    ids, dists = search_flat(small_data, 0, small_data[0], 5)
    assert len(ids) == 0 and len(dists) == 0


def test_subset_search(small_data, rng):
    q = small_data[42]
    subset = rng.choice(len(small_data), size=300, replace=False).astype(np.int32)
    ids, dists = search_subset(small_data, subset, q, 10)
    d = np.linalg.norm(small_data[subset].astype(np.float64) - q, axis=1) ** 2
    want = subset[np.argsort(d, kind="stable")[:10]]
    assert list(ids) == list(want)
    assert set(ids) <= set(subset.tolist())


def test_subset_empty(small_data):
    ids, _ = search_subset(small_data, np.empty(0, np.int32), small_data[0], 5)
    assert len(ids) == 0
