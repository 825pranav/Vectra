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

# Imports: struct + zlib for binary framing and CRC32 checksums, os for fsync and atomic renames.
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

# Op codes stored in each record, plus the fixed binary headers:
# _HDR = (payload length, CRC32), _OPLSN = (op byte, 64-bit log sequence number).
OP_UPSERT = 1
OP_DELETE = 2
_HDR = struct.Struct("<II")
_OPLSN = struct.Struct("<BQ")


# A decoded save record: everything needed to redo one upsert call on recovery.
@dataclass
class Upsert:
    lsn: int
    internal_ids: np.ndarray
    replaced: np.ndarray
    vectors: np.ndarray
    user_ids: list[str]
    attrs: list[dict[str, Any]]


# A decoded delete record: the internal rows to tombstone.
@dataclass
class Delete:
    lsn: int
    internal_ids: np.ndarray


# Either kind of record; replay() yields these to Collection._recover().
Record = Upsert | Delete


# Serialise one upsert call into a payload: op + LSN, counts, id arrays, raw float32 vectors,
# then the user ids and tags as JSON. append() adds the length + CRC frame.
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


# Serialise a delete: op + LSN, count, then the internal ids as int64.
def encode_delete(lsn: int, internal_ids: np.ndarray) -> bytes:
    ids = np.asarray(internal_ids, dtype="<i8")
    return _OPLSN.pack(OP_DELETE, lsn) + struct.pack("<I", ids.size) + ids.tobytes()


# Reverse of the encoders: payload bytes -> Upsert or Delete record (used by replay).
def decode(payload: bytes) -> Record:
    op, lsn = _OPLSN.unpack_from(payload, 0)
    off = _OPLSN.size
    # Delete is short: read the count, then that many int64 ids.
    if op == OP_DELETE:
        (n,) = struct.unpack_from("<I", payload, off)
        ids = np.frombuffer(payload, dtype="<i8", count=n, offset=off + 4).astype(np.int64)
        return Delete(lsn, ids)
    if op != OP_UPSERT:
        raise ValueError(f"unknown WAL op {op}")
    # Upsert: walk an offset through the body in the same order encode_upsert wrote it.
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


# Append-only log split into segment files named by their first LSN (wal/<lsn>.log).
class WriteAheadLog:
    # Remember the folder; the file isn't opened until start() runs after recovery.
    def __init__(self, directory: Path, fsync: bool = True) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._f = None
        self.last_lsn = 0

    # ---- segments -----------------------------------------------------------

    # List log segment files as (start LSN, path), oldest first.
    def segments(self) -> list[tuple[int, Path]]:
        segs = []
        for p in self.dir.glob("*.log"):
            with contextlib.suppress(ValueError):
                segs.append((int(p.stem), p))
        return sorted(segs)

    # Close the current segment and open (or create) one for appending, then fsync the folder entry.
    def _open_segment(self, start_lsn: int) -> None:
        if self._f is not None:
            self._f.close()
        self._f = open(self.dir / f"{start_lsn:020d}.log", "ab")  # noqa: SIM115
        _fsync_dir(self.dir)

    # ---- read ---------------------------------------------------------------

    # Recovery: read segments in order, check each frame, yield records newer than the snapshot.
    def replay(self, after_lsn: int) -> Iterator[Record]:
        """Yield records with lsn > ``after_lsn``; truncate a torn tail in place."""
        for _, path in self.segments():
            data = path.read_bytes()
            off = 0
            # Read frame by frame; stop at a short header, a short body or a CRC mismatch.
            while off < len(data):
                if off + _HDR.size > len(data):
                    break
                size, crc = _HDR.unpack_from(data, off)
                body = data[off + _HDR.size : off + _HDR.size + size]
                if len(body) < size or zlib.crc32(body) != crc:
                    break
                # Good frame: decode it, track the highest LSN seen, and yield it if the snapshot
                # doesn't cover it.
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

    # After recovery, open the newest segment (or a new one) for appends.
    def start(self, next_lsn: int) -> None:
        """Begin appending (after recovery). Reuses the newest segment if any."""
        segs = self.segments()
        self._open_segment(segs[-1][0] if segs else next_lsn)

    # Durable write: frame the payload with length + CRC, write, flush, and fsync if enabled.
    # Collection calls this before applying a change, so with fsync on a confirmed save is on disk.
    def append(self, payload: bytes) -> None:
        frame = _HDR.pack(len(payload), zlib.crc32(payload)) + payload
        self._f.write(frame)
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())

    # After a snapshot at snapshot_lsn, start a new segment and delete old ones it fully covers.
    def rotate(self, snapshot_lsn: int) -> None:
        """Start a fresh segment and drop segments fully covered by a snapshot."""
        old = self.segments()
        self._open_segment(snapshot_lsn + 1)
        for start, path in old:
            if path.name != f"{snapshot_lsn + 1:020d}.log" and start <= snapshot_lsn:
                with contextlib.suppress(OSError):
                    path.unlink()

    # Close the open segment file.
    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


# ---- snapshots ------------------------------------------------------------------


# Point-in-time copies of in-memory state, so recovery only replays the log after them.
class Snapshots:
    # Snapshots live in their own folder.
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    # Find the snapshot that CURRENT names, as (lsn, path), or None if there isn't a usable one.
    def current(self) -> tuple[int, Path] | None:
        cur = self.dir / "CURRENT"
        if not cur.exists():
            return None
        lsn = int(cur.read_text().strip())
        path = self.dir / f"{lsn:020d}"
        return (lsn, path) if path.exists() else None

    # Write a snapshot crash-safely: arrays + info into a temp folder, then rename, then publish.
    def write(self, lsn: int, arrays: dict[str, np.ndarray], info: dict[str, Any]) -> Path:
        # Start from a clean temp folder named after the LSN.
        tmp = self.dir / f"tmp-{lsn:020d}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        # Save each array as its own .npy file and fsync it.
        for name, arr in arrays.items():
            path = tmp / f"{name}.npy"
            with open(path, "wb") as f:
                np.save(f, arr, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
        # Save the small JSON info dict alongside and fsync it.
        with open(tmp / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f)
            f.flush()
            os.fsync(f.fileno())
        # Atomically rename the finished temp folder to its final name and persist the rename.
        final = self.dir / f"{lsn:020d}"
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        _fsync_dir(self.dir)
        # Publish it: write CURRENT.tmp, fsync, then atomically replace CURRENT.
        cur_tmp = self.dir / "CURRENT.tmp"
        with open(cur_tmp, "w", encoding="utf-8") as f:
            f.write(str(lsn))
            f.flush()
            os.fsync(f.fileno())
        os.replace(cur_tmp, self.dir / "CURRENT")
        _fsync_dir(self.dir)
        # Delete every other snapshot folder (older snapshots and leftover temp folders).
        for p in self.dir.iterdir():
            if p.is_dir() and p != final:
                shutil.rmtree(p, ignore_errors=True)
        return final

    # Read a snapshot folder back: {array name: ndarray} plus the info dict.
    @staticmethod
    def load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays = {p.stem: np.load(p, allow_pickle=False) for p in path.glob("*.npy")}
        info = json.loads((path / "info.json").read_text(encoding="utf-8"))
        return arrays, info


# fsync a directory so file creates/renames inside it survive a power cut.
def _fsync_dir(path: Path) -> None:
    """Persist a directory entry (rename/create). Not supported on Windows."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
