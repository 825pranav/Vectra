"""Write-ahead log + snapshots.

Every upsert/delete is appended here, and fsynced, *before* it is applied or
acknowledged. The collection periodically snapshots its in-memory state (graph
arrays, PQ codebooks and codes, attribute columns, tombstones) and starts a new
log segment; recovery = load the newest complete snapshot, replay the log tail.

Record framing (little endian)::

    u32 payload_len | u32 crc32(payload) | payload
    payload = u8 op | u64 lsn | body

    UPSERT body: u32 n | u32 dim | u32 n_replaced | i64[n] internal ids |
                 i64[n_replaced] replaced ids | f32[n*dim] vectors |
                 u32 meta_len | utf-8 JSON {"ids": [...], "attrs": [...]}
    DELETE body: u32 n | i64[n] internal ids

A crash mid-append leaves a short or CRC-failing tail record. It was never
acknowledged (acks happen after fsync), so recovery drops it and truncates the
segment there before appending again.

Snapshots are written to ``snapshots/tmp-<lsn>/``, fsynced, renamed to
``snapshots/<lsn>/`` and then published by atomically replacing ``CURRENT``.
A crash at any point leaves either the old or the new snapshot current, never a
half-written one.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OP_UPSERT = 1
OP_DELETE = 2
_HDR = struct.Struct("<II")
_OPLSN = struct.Struct("<BQ")


@dataclass
class Upsert:
    lsn: int
    internal_ids: np.ndarray
    replaced: np.ndarray
    vectors: np.ndarray
    user_ids: list[str]
    attrs: list[dict[str, Any]]


@dataclass
class Delete:
    lsn: int
    internal_ids: np.ndarray


Record = Upsert | Delete


def encode_upsert(
    lsn: int,
    internal_ids: np.ndarray,
    replaced: np.ndarray,
    vectors: np.ndarray,
    user_ids: list[str],
    attrs: list[dict[str, Any]],
) -> bytes:
    n, dim = vectors.shape
    meta = json.dumps({"ids": user_ids, "attrs": attrs}, separators=(",", ":")).encode()
    return b"".join(
        [
            _OPLSN.pack(OP_UPSERT, lsn),
            struct.pack("<III", n, dim, len(replaced)),
            np.asarray(internal_ids, dtype="<i8").tobytes(),
            np.asarray(replaced, dtype="<i8").tobytes(),
            np.ascontiguousarray(vectors, dtype="<f4").tobytes(),
            struct.pack("<I", len(meta)),
            meta,
        ]
    )


def encode_delete(lsn: int, internal_ids: np.ndarray) -> bytes:
    ids = np.asarray(internal_ids, dtype="<i8")
    return _OPLSN.pack(OP_DELETE, lsn) + struct.pack("<I", ids.size) + ids.tobytes()


def decode(payload: bytes) -> Record:
    op, lsn = _OPLSN.unpack_from(payload, 0)
    off = _OPLSN.size
    if op == OP_DELETE:
        (n,) = struct.unpack_from("<I", payload, off)
        ids = np.frombuffer(payload, dtype="<i8", count=n, offset=off + 4).astype(np.int64)
        return Delete(lsn, ids)
    if op != OP_UPSERT:
        raise ValueError(f"unknown WAL op {op}")
    n, dim, nr = struct.unpack_from("<III", payload, off)
    off += 12
    ids = np.frombuffer(payload, dtype="<i8", count=n, offset=off).astype(np.int64)
    off += 8 * n
    rep = np.frombuffer(payload, dtype="<i8", count=nr, offset=off).astype(np.int64)
    off += 8 * nr
    vecs = np.frombuffer(payload, dtype="<f4", count=n * dim, offset=off).reshape(n, dim)
    off += 4 * n * dim
    (mlen,) = struct.unpack_from("<I", payload, off)
    meta = json.loads(payload[off + 4 : off + 4 + mlen])
    return Upsert(lsn, ids, rep, vecs.astype(np.float32), meta["ids"], meta["attrs"])


class WriteAheadLog:
    def __init__(self, directory: Path, fsync: bool = True) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._f = None
        self.last_lsn = 0

    # ---- segments -----------------------------------------------------------

    def segments(self) -> list[tuple[int, Path]]:
        segs = []
        for p in self.dir.glob("*.log"):
            with contextlib.suppress(ValueError):
                segs.append((int(p.stem), p))
        return sorted(segs)

    def _open_segment(self, start_lsn: int) -> None:
        if self._f is not None:
            self._f.close()
        self._f = open(self.dir / f"{start_lsn:020d}.log", "ab")  # noqa: SIM115
        _fsync_dir(self.dir)

    # ---- read ---------------------------------------------------------------

    def replay(self, after_lsn: int) -> Iterator[Record]:
        """Yield records with lsn > ``after_lsn``; truncate a torn tail in place."""
        for _, path in self.segments():
            data = path.read_bytes()
            off = 0
            while off < len(data):
                if off + _HDR.size > len(data):
                    break
                size, crc = _HDR.unpack_from(data, off)
                body = data[off + _HDR.size : off + _HDR.size + size]
                if len(body) < size or zlib.crc32(body) != crc:
                    break
                rec = decode(body)
                self.last_lsn = max(self.last_lsn, rec.lsn)
                if rec.lsn > after_lsn:
                    yield rec
                off += _HDR.size + size
            if off < len(data):
                # torn / corrupt tail: never acknowledged, drop it
                with open(path, "r+b") as f:
                    f.truncate(off)
                    f.flush()
                    os.fsync(f.fileno())
                break

    # ---- write --------------------------------------------------------------

    def start(self, next_lsn: int) -> None:
        """Begin appending (after recovery). Reuses the newest segment if any."""
        segs = self.segments()
        self._open_segment(segs[-1][0] if segs else next_lsn)

    def append(self, payload: bytes) -> None:
        frame = _HDR.pack(len(payload), zlib.crc32(payload)) + payload
        self._f.write(frame)
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())

    def rotate(self, snapshot_lsn: int) -> None:
        """Start a fresh segment and drop segments fully covered by a snapshot."""
        old = self.segments()
        self._open_segment(snapshot_lsn + 1)
        for start, path in old:
            if path.name != f"{snapshot_lsn + 1:020d}.log" and start <= snapshot_lsn:
                with contextlib.suppress(OSError):
                    path.unlink()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


# ---- snapshots ------------------------------------------------------------------


class Snapshots:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def current(self) -> tuple[int, Path] | None:
        cur = self.dir / "CURRENT"
        if not cur.exists():
            return None
        lsn = int(cur.read_text().strip())
        path = self.dir / f"{lsn:020d}"
        return (lsn, path) if path.exists() else None

    def write(self, lsn: int, arrays: dict[str, np.ndarray], info: dict[str, Any]) -> Path:
        tmp = self.dir / f"tmp-{lsn:020d}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for name, arr in arrays.items():
            path = tmp / f"{name}.npy"
            with open(path, "wb") as f:
                np.save(f, arr, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
        with open(tmp / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f)
            f.flush()
            os.fsync(f.fileno())
        final = self.dir / f"{lsn:020d}"
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        _fsync_dir(self.dir)
        cur_tmp = self.dir / "CURRENT.tmp"
        with open(cur_tmp, "w", encoding="utf-8") as f:
            f.write(str(lsn))
            f.flush()
            os.fsync(f.fileno())
        os.replace(cur_tmp, self.dir / "CURRENT")
        _fsync_dir(self.dir)
        for p in self.dir.iterdir():
            if p.is_dir() and p != final:
                shutil.rmtree(p, ignore_errors=True)
        return final

    @staticmethod
    def load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays = {p.stem: np.load(p, allow_pickle=False) for p in path.glob("*.npy")}
        info = json.loads((path / "info.json").read_text(encoding="utf-8"))
        return arrays, info


def _fsync_dir(path: Path) -> None:
    """Persist a directory entry (rename/create). Not supported on Windows."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
