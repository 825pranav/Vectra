import numpy as np
import pytest

from engine.collection import Collection, InvalidArgument


@pytest.fixture
def col(tmp_path):
    c = Collection.create(tmp_path / "c", "c", dim=32, overrides={"index": "flat"})
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


def test_persistence(tmp_path, small_data):
    c = Collection.create(tmp_path / "p", "p", dim=32, overrides={"index": "flat"})
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


def test_capacity_growth(tmp_path, small_data):
    c = Collection.create(
        tmp_path / "g", "g", dim=32, overrides={"index": "flat", "initial_capacity": 4}
    )
    c.upsert([f"v{i}" for i in range(200)], small_data[:200])
    assert c.search(small_data[150], k=1).ids == ["v150"]
    c.close()
