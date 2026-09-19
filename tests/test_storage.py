import numpy as np

from storage.vectors import VectorStore


def test_growth_preserves_rows(tmp_path, rng):
    s = VectorStore(tmp_path, dim=4, capacity=2)
    x = rng.normal(size=(100, 4)).astype(np.float32)
    for i in range(100):
        s.write(np.array([i]), x[i : i + 1])
    assert s.capacity >= 100
    assert np.array_equal(s.array[:100], x)
    s.close()
    r = VectorStore(tmp_path, dim=4)
    assert np.array_equal(r.array[:100], x)
    assert len(list(tmp_path.glob("vectors.*.f32"))) == 1  # older generations cleaned up
    r.close()


def test_interrupted_growth_is_ignored(tmp_path, rng):
    """A crash mid-growth leaves a partial ``.tmp`` next generation; the complete
    older generation must still be the one that opens."""
    s = VectorStore(tmp_path, dim=4, capacity=8)
    x = rng.normal(size=(8, 4)).astype(np.float32)
    s.write(np.arange(8), x)
    s.close()
    (tmp_path / "vectors.1.tmp").write_bytes(b"\x00" * 40)  # torn copy
    r = VectorStore(tmp_path, dim=4)
    assert r.capacity == 8 and np.array_equal(r.array[:8], x)
    assert not list(tmp_path.glob("*.tmp"))
    r.close()


def test_old_mapping_stays_valid_for_readers(tmp_path, rng):
    s = VectorStore(tmp_path, dim=4, capacity=4)
    x = rng.normal(size=(4, 4)).astype(np.float32)
    s.write(np.arange(4), x)
    reader_view = s.array  # a reader captured the old generation
    s.ensure_capacity(64)
    assert np.array_equal(reader_view, x)  # still readable after the swap
    assert np.array_equal(s.array[:4], x)
    del reader_view
    s.close()
