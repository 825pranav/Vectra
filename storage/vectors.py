"""Raw float32 vectors in a flat file, memory-mapped and addressed by internal id.

This is the full-precision copy of every vector: HNSW traverses it directly and
the PQ path re-ranks from it. Growth works by *generation*: a bigger file
``vectors.<gen+1>.f32`` is created, the old contents are copied over and the
reference is swapped. Readers that grabbed the old array keep a valid mapping
until they drop it, so growth never blocks or invalidates in-flight searches
(Windows cannot resize a file while it is mapped, which rules out ftruncate).
"""

# Imports: os for fsync/rename, regex to find vector file generations, NumPy for the memmap.
from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

import numpy as np

# Matches vector file names like vectors.3.f32 and captures the generation number.
_GEN_RE = re.compile(r"^vectors\.(\d+)\.f32$")


# Full-precision vector store: one float32 row per internal id, in a memory-mapped file.
# Collection writes rows here; HNSW, flat scans and PQ re-rank read from .array.
class VectorStore:
    # Open (or create) the store: map the newest generation file and clean up leftovers.
    def __init__(self, directory: Path, dim: int, capacity: int = 1024) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        # List existing generation files, sorted so the last one is the newest.
        gens = sorted(
            (int(m.group(1)), p) for p in self.dir.iterdir() if (m := _GEN_RE.match(p.name))
        )
        for tmp in self.dir.glob("vectors.*.tmp"):  # growth interrupted by a crash
            with contextlib.suppress(OSError):
                tmp.unlink()
        # Existing data: map the newest file (row count from file size) and delete older generations.
        if gens:
            self._gen, path = gens[-1]
            rows = path.stat().st_size // (4 * self.dim)
            self._mm = np.memmap(path, dtype=np.float32, mode="r+", shape=(rows, self.dim))
            for _, old in gens[:-1]:
                with contextlib.suppress(OSError):
                    old.unlink()
        # Fresh collection: create generation 0 with the starting capacity.
        else:
            self._gen = 0
            self._mm = self._create(self._path(0), max(int(capacity), 1))
        self._stale: list[Path] = []

    # File path for a given generation number.
    def _path(self, gen: int) -> Path:
        return self.dir / f"vectors.{gen}.f32"

    # Create a new zero-filled memory-mapped file with `rows` rows.
    def _create(self, path: Path, rows: int) -> np.memmap:
        return np.memmap(path, dtype=np.float32, mode="w+", shape=(rows, self.dim))

    # How many rows the current file can hold (not how many are used).
    @property
    def capacity(self) -> int:
        return self._mm.shape[0]

    # Current rows as a NumPy array; Numba kernels and searches read this.
    @property
    def array(self) -> np.ndarray:
        """Plain ndarray view over the mapping (what Numba kernels receive)."""
        return np.asarray(self._mm)

    # Grow the file so it holds at least `rows` rows; called before writing new rows.
    def ensure_capacity(self, rows: int) -> None:
        if rows <= self.capacity:
            return
        # Double the capacity until it is big enough.
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
        # Copy old rows into the temp file in ~64 MB chunks, pad to the new size and fsync it.
        with open(tmp, "wb") as f:
            step = max(1, (64 << 20) // (4 * self.dim))
            for s in range(0, old.shape[0], step):
                f.write(old[s : s + step].tobytes())
            f.truncate(new_rows * self.dim * 4)
            f.flush()
            os.fsync(f.fileno())
        # Atomic rename to the final name, map it, and swap it in; the old file is queued for deletion.
        os.replace(tmp, new_path)
        new_mm = np.memmap(new_path, dtype=np.float32, mode="r+", shape=(new_rows, self.dim))
        self._stale.append(self._path(self._gen))
        self._gen += 1
        self._mm = new_mm
        self._cleanup()

    # Try to delete old generation files; ones still mapped by a reader are kept for a later retry.
    def _cleanup(self) -> None:
        # The old mapping may still be referenced by a reader; retry later if so.
        keep = []
        for p in self._stale:
            try:
                p.unlink()
            except OSError:
                keep.append(p)
        self._stale = keep

    # Write vectors into their rows (row = internal id), growing the file first if needed.
    def write(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        ids = np.asarray(ids)
        if ids.size:
            self.ensure_capacity(int(ids.max()) + 1)
            self._mm[ids] = vecs

    # Push dirty memmap pages to disk (used before taking a snapshot).
    def flush(self) -> None:
        self._mm.flush()

    # Bytes used by n vectors (for stats).
    def nbytes(self, n: int) -> int:
        return int(n) * self.dim * 4

    # Flush, drop the mapping and remove any stale old files on shutdown.
    def close(self) -> None:
        self._mm.flush()
        del self._mm
        self._cleanup()
