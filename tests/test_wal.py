import os

import numpy as np
import pytest

from bench.crash import crash_once
from engine.collection import Collection
from storage.wal import (
    Delete,
    Snapshots,
    Upsert,
    WriteAheadLog,
    decode,
    encode_delete,
    encode_upsert,
)


def test_record_roundtrip(rng):
    v = rng.normal(size=(3, 5)).astype(np.float32)
    rec = decode(
        encode_upsert(7, np.array([4, 5, 6]), np.array([1]), v, ["a", "b", "c"], [{"x": 1.5}] * 3)
    )
    assert isinstance(rec, Upsert) and rec.lsn == 7
    assert rec.internal_ids.tolist() == [4, 5, 6] and rec.replaced.tolist() == [1]
    assert np.array_equal(rec.vectors, v)
    assert rec.user_ids == ["a", "b", "c"] and rec.attrs[0] == {"x": 1.5}
    d = decode(encode_delete(8, np.array([2, 3])))
    assert isinstance(d, Delete) and d.lsn == 8 and d.internal_ids.tolist() == [2, 3]


def _write(wal, lsns):
    for lsn in lsns:
        wal.append(encode_delete(lsn, np.array([lsn])))


def test_replay_after_lsn(tmp_path):
    wal = WriteAheadLog(tmp_path, fsync=False)
    wal.start(1)
    _write(wal, range(1, 11))
    wal.close()
    got = [r.lsn for r in WriteAheadLog(tmp_path).replay(6)]
    assert got == [7, 8, 9, 10]


@pytest.mark.parametrize("cut", [1, 5, 11])
def test_torn_tail_is_truncated(tmp_path, cut):
    wal = WriteAheadLog(tmp_path, fsync=False)
    wal.start(1)
    _write(wal, range(1, 6))
    wal.close()
    ((_, seg),) = WriteAheadLog(tmp_path).segments()
    full = seg.stat().st_size
    with open(seg, "r+b") as f:
        f.truncate(full - cut)  # tear the last record
    w2 = WriteAheadLog(tmp_path)
    assert [r.lsn for r in w2.replay(0)] == [1, 2, 3, 4]
    assert seg.stat().st_size < full - cut or cut == 0
    # appending after recovery produces a clean log
    w2.start(5)
    _write(w2, [5])
    w2.close()
    assert [r.lsn for r in WriteAheadLog(tmp_path).replay(0)] == [1, 2, 3, 4, 5]


def test_corrupt_record_stops_replay(tmp_path):
    wal = WriteAheadLog(tmp_path, fsync=False)
    wal.start(1)
    _write(wal, range(1, 4))
    wal.close()
    ((_, seg),) = wal.segments()
    data = bytearray(seg.read_bytes())
    data[-3] ^= 0xFF  # flip a byte in the last record's body
    seg.write_bytes(bytes(data))
    assert [r.lsn for r in WriteAheadLog(tmp_path).replay(0)] == [1, 2]


def test_rotate_drops_covered_segments(tmp_path):
    wal = WriteAheadLog(tmp_path, fsync=False)
    wal.start(1)
    _write(wal, range(1, 4))
    wal.rotate(3)
    _write(wal, [4, 5])
    wal.close()
    segs = WriteAheadLog(tmp_path).segments()
    assert [s for s, _ in segs] == [4]
    assert [r.lsn for r in WriteAheadLog(tmp_path).replay(3)] == [4, 5]


def test_snapshot_publish_is_atomic(tmp_path):
    snaps = Snapshots(tmp_path)
    assert snaps.current() is None
    snaps.write(5, {"a": np.arange(3)}, {"lsn": 5})
    # a half-written newer snapshot (crash before CURRENT moved) is ignored
    (tmp_path / "tmp-00000000000000000009").mkdir()
    lsn, path = snaps.current()
    arrays, info = Snapshots.load(path)
    assert lsn == 5 and info == {"lsn": 5} and arrays["a"].tolist() == [0, 1, 2]
    snaps.write(9, {"a": np.arange(4)}, {"lsn": 9})
    assert snaps.current()[0] == 9
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["00000000000000000009"]


# ---- collection-level ---------------------------------------------------------


@pytest.mark.parametrize("index", ["flat", "hnsw", "hnsw_pq"])
def test_recovery_without_clean_close(tmp_path, small_data, index):
    over = {"index": index, "durability.snapshot_every": 700, "pq.m": 8, "pq.train_size": 500}
    c = Collection.create(tmp_path / "c", "c", dim=32, overrides=over)
    ids = [f"v{i}" for i in range(len(small_data))]
    attrs = [{"i": float(i), "cat": f"c{i % 3}"} for i in range(len(small_data))]
    for s in range(0, 1600, 100):
        c.upsert(ids[s : s + 100], small_data[s : s + 100], attrs[s : s + 100])
    c.delete(["v3", "v1200", "v1599"])
    c.upsert(["v5"], small_data[1700:1701], [{"i": -1.0}])
    before = {q: c.search(small_data[q], k=5, ef=128).ids for q in (0, 5, 900, 1500)}
    lsn = c.lsn
    # simulate a crash: drop the object without close() (no final snapshot)
    c.wal.close()
    c.meta.close()
    del c
    r = Collection.open(tmp_path / "c")
    assert r.lsn == lsn and r.stats()["count"] == 1600 - 3
    assert "v3" not in r.meta.user_to_internal
    assert r.search(small_data[1700], k=1).ids == ["v5"]
    assert r.search(small_data[1700], k=1, include_attributes=True).attributes[0] == {"i": -1.0}
    after = {q: r.search(small_data[q], k=5, ef=128).ids for q in (0, 5, 900, 1500)}
    assert after == before
    assert r.search(small_data[0], k=3, filter="cat == 'c1'").ids[0] in {
        f"v{i}" for i in range(1, 1600, 3)
    }
    r.close()


def test_clean_close_snapshots_and_empties_replay(tmp_path, small_data):
    c = Collection.create(tmp_path / "c", "c", dim=32)
    c.upsert([f"v{i}" for i in range(300)], small_data[:300])
    c.close()
    r = Collection.open(tmp_path / "c")
    assert r.snapshots.current()[0] == r.lsn == 1
    assert list(r.wal.replay(r.lsn)) == []
    assert r.search(small_data[42], k=1).ids == ["v42"]
    r.close()


def test_invalid_attribute_name_rejected_before_logging(tmp_path, small_data):
    from engine.collection import InvalidArgument

    c = Collection.create(tmp_path / "c", "c", dim=32)
    with pytest.raises(InvalidArgument):
        c.upsert(["a"], small_data[:1], [{"bad key": 1.0}])
    assert c.lsn == 0
    c.close()


@pytest.mark.skipif(os.environ.get("VECTRA_SKIP_CRASH") == "1", reason="crash test disabled")
def test_kill9_recovery(tmp_path):
    """Short version of bench/crash.py: a few real kill -9 cycles."""
    for seed in range(4):
        res = crash_once(tmp_path, seed=1000 + seed, max_acks=80)
        assert res["ok"], res
