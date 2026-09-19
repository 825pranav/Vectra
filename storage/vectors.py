"""Raw float32 vectors in a flat file, memory-mapped and addressed by internal id.

This is the full-precision copy of every vector: HNSW traverses it directly and
the PQ path re-ranks from it. Growth works by *generation*: a bigger file
``vectors.<gen+1>.f32`` is created, the old contents are copied over and the
reference is swapped. Readers that grabbed the old array keep a valid mapping
until they drop it, so growth never blocks or invalidates in-flight searches
(Windows cannot resize a file while it is mapped, which rules out ftruncate).
"""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

import numpy as np

_GEN_RE = re.compile(r"^vectors\.(\d+)\.f32$")


class VectorStore:
    def __init__(self, directory: Path, dim: int, capacity: int = 1024) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        gens = sorted(
            (int(m.group(1)), p) for p in self.dir.iterdir() if (m := _GEN_RE.match(p.name))
        )
        for tmp in self.dir.glob("vectors.*.tmp"):  # growth interrupted by a crash
            with contextlib.suppress(OSError):
                tmp.unlink()
        if gens:
            self._gen, path = gens[-1]
            rows = path.stat().st_size // (4 * self.dim)
            self._mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(rows, self.dim))
            for _, old in gens[:-1]:
                with contextlib.suppress(OSError):
                    old.unlink()
        else:
            self._gen = 0
            self._mm = self._create(self._path(0), max(int(capacity), 1))
        self._stale: list[Path] = []

    def _path(self, gen: int) -> Path:
        return self.dir / f"vectors.{gen}.f32"

    def _create(self, path: Path, rows: int) -> np.memmap:
        return np.memmap(path, dtype=np.float32, mode="w+", shape=(rows, self.dim))

    @property
    def capacity(self) -> int:
        return self._mm.shape[0]

    @property
    def array(self) -> np.ndarray:
        """Plain ndarray view over the mapping (what Numba kernels receive)."""
        return np.asarray(self._mm)

    def ensure_capacity(self, rows: int) -> None:
        if rows <= self.capacity:
            return
        new_rows = self.capacity
        while new_rows < rows:
            new_rows *= 2
        # Build the next generation under a temporary name and rename it only once
        # it is complete and fsynced: open() trusts the highest-numbered file, so
        # a half-copied file must never carry a final name. (Plain writes, not a
        # memmap: Windows cannot rename a file while it is mapped.)
        new_path = self._path(self._gen + 1)
        tmp = new_path.with_suffix(".tmp")
        old = np.asarray(self._mm)
        with open(tmp, "wb") as f:
            step = max(1, (64 << 20) // (4 * self.dim))
            for s in range(0, old.shape[0], step):
                f.write(old[s : s + step].tobytes())
            f.truncate(new_rows * self.dim * 4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, new_path)
        new_mm = np.memmap(new_path, dtype=np.float32, mode="r+", shape=(new_rows, self.dim))
        self._stale.append(self._path(self._gen))
        self._gen += 1
        self._mm = new_mm
        self._cleanup()

    def _cleanup(self) -> None:
        # The old mapping may still be referenced by a reader; retry later if so.
        keep = []
        for p in self._stale:
            try:
                p.unlink()
            except OSError:
                keep.append(p)
        self._stale = keep

    def write(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        ids = np.asarray(ids)
        if ids.size:
            self.ensure_capacity(int(ids.max()) + 1)
            self._mm[ids] = vecs

    def flush(self) -> None:
        self._mm.flush()

    def nbytes(self, n: int) -> int:
        return int(n) * self.dim * 4

    def close(self) -> None:
        self._mm.flush()
        del self._mm
        self._cleanup()
