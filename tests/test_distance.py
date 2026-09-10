import numpy as np

from index.distance import (
    _njit_gather_l2sq,
    _njit_scan_l2sq,
    normalize,
    ref_gather_l2sq,
    ref_scan_l2sq,
)


def test_scan_matches_reference(small_data, rng):
    q = small_data[7]
    got = _njit_scan_l2sq(q, small_data, len(small_data))
    want = ref_scan_l2sq(q, small_data, len(small_data))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)
    assert got.dtype == np.float32


def test_scan_respects_n(small_data):
    q = small_data[0]
    assert _njit_scan_l2sq(q, small_data, 100).shape == (100,)


def test_gather_matches_reference(small_data, rng):
    q = small_data[3]
    ids = rng.choice(len(small_data), size=257, replace=False).astype(np.int32)
    got = _njit_gather_l2sq(q, small_data, ids)
    want = ref_gather_l2sq(q, small_data, ids)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)


def test_self_distance_zero(small_data):
    q = small_data[11]
    d = _njit_scan_l2sq(q, small_data, len(small_data))
    assert d[11] == 0.0


def test_normalize_unit_rows(rng):
    x = rng.normal(size=(50, 16)).astype(np.float32) * 100
    n = normalize(x)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, rtol=1e-5)
    z = normalize(np.zeros((2, 4), dtype=np.float32))
    assert np.all(z == 0)
