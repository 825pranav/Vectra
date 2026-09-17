import numpy as np
import pytest

from engine.collection import Collection, InvalidArgument


@pytest.fixture(params=["flat", "hnsw"])
def col(request, tmp_path):
    c = Collection.create(tmp_path / "c", "c", dim=32, overrides={"index": request.param})
    yield c
    c.close()


def test_upsert_and_search(col, small_data):
    ids = [f"v{i}" for i in range(500)]
    col.upsert(ids, small_data[:500], [{"price": float(i), "cat": f"c{i % 5}"} for i in range(500)])
    res = col.search(small_data[17], k=5, include_attributes=True)
    assert res.ids[0] == "v17"
    assert res.distances[0] == 0.0
    assert res.attributes[0] == {"price": 17.0, "cat": "c2"}


def test_upsert_replaces(col, small_data):
    col.upsert(["a", "b"], small_data[:2], [{"p": 1.0}, {"p": 2.0}])
    col.upsert(["a"], small_data[100], [{"p": 9.0}])
    res = col.search(small_data[100], k=2, include_attributes=True)
    assert res.ids[0] == "a" and res.attributes[0] == {"p": 9.0}
    # the old vector for "a" must be gone
    res_old = col.search(small_data[0], k=2)
    assert "a" not in res_old.ids or res_old.distances[list(res_old.ids).index("a")] > 0
    assert col.stats()["count"] == 2


def test_duplicate_ids_in_one_batch(col, small_data):
    col.upsert(["x", "x", "x"], small_data[:3], [{"v": 1.0}, {"v": 2.0}, {"v": 3.0}])
    assert col.stats()["count"] == 1
    res = col.search(small_data[2], k=1, include_attributes=True)
    assert res.ids == ["x"] and res.distances[0] == 0.0 and res.attributes[0] == {"v": 3.0}


def test_delete(col, small_data):
    col.upsert([f"v{i}" for i in range(10)], small_data[:10])
    assert col.delete(["v3", "nope"]) == 1
    res = col.search(small_data[3], k=10)
    assert "v3" not in res.ids
    assert col.stats()["count"] == 9


@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_persistence(tmp_path, small_data, index):
    c = Collection.create(tmp_path / "p", "p", dim=32, overrides={"index": index})
    c.upsert([f"v{i}" for i in range(50)], small_data[:50], [{"i": float(i)} for i in range(50)])
    c.delete(["v7"])
    c.store.flush()
    c.close()
    c2 = Collection.open(tmp_path / "p")
    assert c2.stats()["count"] == 49
    res = c2.search(small_data[20], k=3, include_attributes=True)
    assert res.ids[0] == "v20" and res.attributes[0] == {"i": 20.0}
    assert "v7" not in c2.search(small_data[7], k=50).ids
    c2.close()


def test_cosine_metric(tmp_path, rng):
    c = Collection.create(
        tmp_path / "cos", "cos", dim=8, overrides={"metric": "cosine", "index": "flat"}
    )
    v = rng.normal(size=(3, 8)).astype(np.float32)
    c.upsert(["a", "b", "c"], v)
    # scaling a vector must not change cosine ranking
    res = c.search(v[1] * 1000, k=1)
    assert res.ids == ["b"]
    c.close()


def test_validation(col, small_data):
    with pytest.raises(InvalidArgument):
        col.search(small_data[0], k=0)
    with pytest.raises(InvalidArgument):
        col.upsert(["a"], np.ones((1, 5), dtype=np.float32))  # wrong dim
    with pytest.raises(InvalidArgument):
        col.upsert(["a"], np.array([[np.nan] * 32], dtype=np.float32))
    with pytest.raises(InvalidArgument):
        col.upsert(["a", "b"], small_data[:1])
    # attribute type mismatch across upserts
    col.upsert(["a"], small_data[:1], [{"price": 5.0}])
    with pytest.raises(InvalidArgument):
        col.upsert(["b"], small_data[1:2], [{"price": "cheap"}])


@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_capacity_growth(tmp_path, small_data, index):
    c = Collection.create(
        tmp_path / "g", "g", dim=32, overrides={"index": index, "initial_capacity": 4}
    )
    c.upsert([f"v{i}" for i in range(200)], small_data[:200])
    assert c.search(small_data[150], k=1).ids == ["v150"]
    c.close()


def test_hnsw_matches_flat_on_small_collection(tmp_path, small_data):
    flat = Collection.create(tmp_path / "f", "f", dim=32, overrides={"index": "flat"})
    hnsw = Collection.create(tmp_path / "h", "h", dim=32, overrides={"index": "hnsw"})
    ids = [f"v{i}" for i in range(len(small_data))]
    flat.upsert(ids, small_data)
    hnsw.upsert(ids, small_data)
    rng = np.random.default_rng(1)
    hits = 0
    for q in small_data[rng.choice(len(small_data), 50, replace=False)] + 0.05:
        a = flat.search(q, k=10).ids
        b = hnsw.search(q, k=10, ef=128).ids
        hits += len(set(a) & set(b))
    assert hits / 500 >= 0.95
    assert hnsw.search(small_data[0], k=3).strategy == "hnsw"
    flat.close()
    hnsw.close()


def test_upserts_in_many_small_batches(tmp_path, small_data):
    c = Collection.create(tmp_path / "b", "b", dim=32, overrides={"initial_capacity": 8})
    for start in range(0, 600, 37):
        stop = min(start + 37, 600)
        c.upsert([f"v{i}" for i in range(start, stop)], small_data[start:stop])
    for i in (0, 36, 37, 300, 599):
        assert c.search(small_data[i], k=1).ids == [f"v{i}"]
    c.close()


def test_hnsw_pq_collection(tmp_path, small_data):
    over = {"index": "hnsw_pq", "pq.m": 8, "pq.train_size": 1000, "pq.kmeans_iters": 8}
    c = Collection.create(tmp_path / "pq", "pq", dim=32, overrides=over)
    ids = [f"v{i}" for i in range(len(small_data))]
    c.upsert(ids[:600], small_data[:600])
    assert c.search(small_data[5], k=1).strategy == "hnsw"  # not trained yet
    c.upsert(ids[600:], small_data[600:])  # crosses train_size -> trains + encodes all
    res = c.search(small_data[1500], k=5)
    assert res.strategy == "hnsw_pq" and res.ids[0] == "v1500"
    assert res.distances[0] == 0.0  # re-rank distances are exact
    stats = c.stats()
    # float vectors stay on disk; the in-memory index is graph + codebooks + codes
    assert stats["index_bytes"] == c.hnsw.nbytes() + c.pq.nbytes() + c.n * 8
    c.close()
    c2 = Collection.open(tmp_path / "pq")
    assert c2.search(small_data[1500], k=1).ids == ["v1500"]
    c2.close()
