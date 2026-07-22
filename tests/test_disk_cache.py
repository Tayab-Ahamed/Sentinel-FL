"""
tests/test_disk_cache.py — Unit tests for DiskCache.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.training.cache import DiskCache


@pytest.fixture
def cache(tmp_path) -> DiskCache:
    """Return a fresh DiskCache backed by a temporary directory."""
    return DiskCache(data_dir=tmp_path)


@pytest.fixture
def sample_arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, size=100).astype(np.int64)
    return X, y


class TestDiskCacheBasics:
    def test_get_returns_none_on_miss(self, cache):
        assert cache.get("nonexistent") is None

    def test_has_returns_false_on_miss(self, cache):
        assert cache.has("nonexistent") is False

    def test_put_then_get_round_trip(self, cache, sample_arrays):
        X, y = sample_arrays
        cache.put("test_key", X, y)
        result = cache.get("test_key")
        assert result is not None
        X2, y2 = result
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)

    def test_has_returns_true_after_put(self, cache, sample_arrays):
        X, y = sample_arrays
        cache.put("my_key", X, y)
        assert cache.has("my_key")

    def test_put_preserves_dtype(self, cache):
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.arange(10, dtype=np.int64)
        cache.put("dtype_test", X, y)
        X2, y2 = cache.get("dtype_test")
        assert X2.dtype == np.float32
        assert y2.dtype == np.int64

    def test_put_preserves_shape(self, cache, sample_arrays):
        X, y = sample_arrays
        cache.put("shape_test", X, y)
        X2, y2 = cache.get("shape_test")
        assert X2.shape == X.shape
        assert y2.shape == y.shape


class TestDiskCacheInvalidate:
    def test_invalidate_removes_entry(self, cache, sample_arrays):
        X, y = sample_arrays
        cache.put("to_remove", X, y)
        assert cache.has("to_remove")
        removed = cache.invalidate("to_remove")
        assert removed is True
        assert cache.get("to_remove") is None

    def test_invalidate_nonexistent_returns_false(self, cache):
        assert cache.invalidate("ghost") is False

    def test_invalidate_then_put_works(self, cache, sample_arrays):
        X, y = sample_arrays
        cache.put("key", X, y)
        cache.invalidate("key")
        cache.put("key", X, y)
        result = cache.get("key")
        assert result is not None


class TestDiskCacheClear:
    def test_clear_removes_all_entries(self, cache, sample_arrays):
        X, y = sample_arrays
        cache.put("a", X, y)
        cache.put("b", X, y)
        removed = cache.clear()
        assert removed == 2
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_clear_empty_cache_returns_zero(self, cache):
        assert cache.clear() == 0


class TestDiskCacheKeyHandling:
    def test_keys_with_slashes_are_sanitised(self, cache, sample_arrays):
        """Keys with / should still work (sanitised to _)."""
        X, y = sample_arrays
        cache.put("mnist/train", X, y)
        result = cache.get("mnist/train")
        assert result is not None

    def test_different_keys_are_independent(self, cache):
        X1 = np.zeros((5, 2), dtype=np.float32)
        X2 = np.ones((5, 2), dtype=np.float32)
        y = np.zeros(5, dtype=np.int64)
        cache.put("key1", X1, y)
        cache.put("key2", X2, y)
        r1, _ = cache.get("key1")
        r2, _ = cache.get("key2")
        np.testing.assert_array_equal(r1, X1)
        np.testing.assert_array_equal(r2, X2)

    def test_overwrite_replaces_value(self, cache):
        X_old = np.zeros((5, 2), dtype=np.float32)
        X_new = np.ones((5, 2), dtype=np.float32)
        y = np.zeros(5, dtype=np.int64)
        cache.put("key", X_old, y)
        # Second put should overwrite
        cache.put("key", X_new, y)
        result_X, _ = cache.get("key")
        np.testing.assert_array_equal(result_X, X_new)
