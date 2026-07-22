"""
ai/training/cache.py — Filesystem-backed numpy array cache.

Stores ``(X, y)`` dataset pairs as ``.npz`` files under
``<data_dir>/.cache/<key>.npz``.  Writes are atomic (temp-file + rename)
so a crash mid-write never leaves a corrupt cache entry.

Cache keys are arbitrary strings (e.g. ``"mnist_train_v2"``).  Callers are
responsible for choosing stable, version-aware keys so stale data is never
silently served after a library upgrade.

Usage::

    cache = DiskCache(data_dir="datasets")
    result = cache.get("mnist_train")
    if result is None:
        X, y = _download_and_process()
        cache.put("mnist_train", X, y)
    else:
        X, y = result
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_SUBDIR = ".cache"


class DiskCache:
    """Numpy ``.npz`` cache for (X, y) dataset arrays.

    Args:
        data_dir: Root directory under which ``.cache/`` is created.
    """

    def __init__(self, data_dir: str | Path = "datasets") -> None:
        self._cache_dir = Path(data_dir) / _CACHE_SUBDIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return cached ``(X, y)`` or ``None`` on cache miss.

        Args:
            key: Cache key string (filename-safe).

        Returns:
            ``(X, y)`` tuple or ``None`` if the key is not cached.
        """
        path = self._path(key)
        if not path.exists():
            logger.debug("DiskCache miss: key=%s", key)
            return None
        try:
            data = np.load(str(path), allow_pickle=False)
            X, y = data["X"], data["y"]
            logger.debug("DiskCache hit: key=%s X=%s y=%s", key, X.shape, y.shape)
            return X, y
        except Exception as exc:  # corrupt file → treat as miss
            logger.warning("DiskCache: corrupt entry key=%s (%s), treating as miss.", key, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, X: np.ndarray, y: np.ndarray) -> None:
        """Write ``(X, y)`` to the cache atomically.

        Writes to a temp file first, then renames so a crash never produces a
        half-written cache entry.

        Args:
            key: Cache key string.
            X: Feature array.
            y: Label array.
        """
        path = self._path(key)
        # numpy.savez_compressed appends .npz automatically when the path
        # does not already end with .npz.  We write to a .tmp.npz sidecar
        # then rename to the final path for near-atomic writes.
        sidecar = path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(str(sidecar), X=X, y=y)
            # On Windows, rename raises FileExistsError if destination exists.
            if path.exists():
                path.unlink()
            sidecar.rename(path)
            logger.debug("DiskCache put: key=%s X=%s y=%s", key, X.shape, y.shape)
        except Exception:
            sidecar.unlink(missing_ok=True)
            raise

    def invalidate(self, key: str) -> bool:
        """Remove a single cache entry.

        Args:
            key: Cache key to delete.

        Returns:
            ``True`` if the entry existed and was removed, ``False`` otherwise.
        """
        path = self._path(key)
        if path.exists():
            path.unlink()
            logger.debug("DiskCache invalidated: key=%s", key)
            return True
        return False

    def clear(self) -> int:
        """Remove all cache entries.

        Returns:
            Number of entries removed.
        """
        removed = 0
        for p in self._cache_dir.glob("*.npz"):
            # Skip tmp files (pattern: *.tmp)
            if p.suffix == ".npz":
                p.unlink()
                removed += 1
        logger.info("DiskCache cleared %d entries from %s", removed, self._cache_dir)
        return removed

    def has(self, key: str) -> bool:
        """Return ``True`` if the key exists in the cache."""
        return self._path(key).exists()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        """Return the ``.npz`` path for a given cache key."""
        # Sanitise key so it is always a safe filename
        safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._cache_dir / f"{safe}.npz"
