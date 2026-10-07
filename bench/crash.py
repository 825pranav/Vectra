"""Crash-recovery test: kill -9 a writer mid-stream, recover, verify every ack.

    python -m bench.run configs/bench/crash.yaml      # N runs, results JSON
    python -m bench.crash worker <dir> <seed>         # (internal) the writer

The worker upserts small random batches (and occasionally deletes) into a
collection, printing one ``ACK`` line to stdout after each call *returns*. The
parent reads those lines, hard-kills the worker at a random moment (SIGKILL on
POSIX, TerminateProcess on Windows: no cleanup, no atexit, no flush), then
opens the collection in-process and checks:

* every acknowledged upsert is present with bit-identical vector and attributes;
* every acknowledged delete is absent;
* any record that exists at all (acked or not) has the right vector, i.e. the
  torn tail was discarded rather than half-applied;
* the index is consistent (graph size == record count) and searchable.

Process-kill is what this exercises. fsync additionally protects against OS
crash / power loss, which a test on a live machine cannot simulate.
"""

from __future__ import annotations

# Stdlib for spawning and killing the worker process, a reader thread, and timing; NumPy.
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

# Repo root (for PYTHONPATH) and a small vector size to keep each run fast.
ROOT = Path(__file__).resolve().parent.parent
DIM = 8


# Deterministic vector for record i, so the checker can recompute what should be stored.
def vector_for(i: int) -> np.ndarray:
    return np.random.default_rng(1_000_003 + i).normal(size=DIM).astype(np.float32)


# Deterministic tags for record i (number, text and bool) for the same reason.
def attrs_for(i: int) -> dict[str, Any]:
    return {"seq": float(i), "tag": f"t{i % 7}", "odd": bool(i % 2)}


# Child process: keep upserting (and sometimes deleting) and print ACK only after each call returns.
def worker(path: str, seed: int, snapshot_every: int) -> None:
    # Fresh collection with frequent snapshots, so kills also land during snapshot writes.
    from engine.collection import Collection

    over = {"durability.snapshot_every": snapshot_every, "initial_capacity": 64}
    col = Collection.create(path, "crash", DIM, over)
    rng = np.random.default_rng(seed)
    seq = 0
    print("READY", flush=True)
    # Loop forever until the parent kills us; each batch has a random size.
    while True:
        b = int(rng.integers(1, 40))
        ids = [f"w{seq + j}" for j in range(b)]
        vecs = np.stack([vector_for(seq + j) for j in range(b)])
        col.upsert(ids, vecs, [attrs_for(seq + j) for j in range(b)])
        print(f"ACK U {seq} {seq + b}", flush=True)
        seq += b
        # Sometimes delete a few earlier records too.
        if seq > 20 and rng.random() < 0.2:
            victims = sorted({int(v) for v in rng.integers(0, seq, size=3)})
            # announce first: if we die after the delete is logged but before the
            # ACK, these ids may legitimately be gone
            print("INTENT D " + " ".join(map(str, victims)), flush=True)
            col.delete([f"w{v}" for v in victims])
            print("ACK D " + " ".join(map(str, victims)), flush=True)


# Parent side of one test: start a worker, kill -9 it at a random ack, reopen, check every ack.
def crash_once(workdir: Path, seed: int, max_acks: int = 150) -> dict[str, Any]:
    """One kill-and-recover cycle. Returns a verdict dict (``ok`` + details)."""
    from engine.collection import Collection

    # Pick a random snapshot interval and launch the worker as a separate Python process.
    rnd = random.Random(seed)
    path = workdir / f"run{seed}"
    snapshot_every = rnd.choice([50, 200, 1000])
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "bench.crash", "worker", str(path), str(seed), str(snapshot_every)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Shared state filled by the reader thread: ack lines, delete intents, and two events.
    lines: list[str] = []
    intents: list[str] = []
    got_ready = threading.Event()
    enough = threading.Event()
    kill_after = rnd.randint(0, max_acks)

    # Reader thread: collect READY, INTENT and ACK lines from the worker's stdout.
    def read() -> None:
        # Only complete lines count as acks: a line torn by the kill is dropped.
        for raw in proc.stdout:
            if not raw.endswith(b"\n"):
                break
            line = raw.decode().strip()
            if line == "READY":
                got_ready.set()
            elif line.startswith("INTENT"):
                intents.append(line)
            elif line.startswith("ACK"):
                lines.append(line)
                if len(lines) >= kill_after:
                    enough.set()

    # Wait for READY, then for enough acks, then sleep a tiny random time and hard-kill the worker.
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    if not got_ready.wait(120):
        proc.kill()
        return {"ok": False, "seed": seed, "error": proc.stderr.read().decode()[-2000:]}
    enough.wait(60)
    time.sleep(rnd.random() * 0.03)  # land somewhere inside the next write
    proc.kill()
    proc.wait()
    reader.join(5)

    # Turn ack lines into the set of acked upsert ids and acked delete ids.
    acked: set[int] = set()
    deleted: set[int] = set()
    for line in lines:
        parts = line.split()
        if parts[1] == "U":
            acked.update(range(int(parts[2]), int(parts[3])))
        else:
            deleted.update(int(v) for v in parts[2:])
    # A delete that was announced but never acknowledged may or may not have
    # reached the log before the kill: either outcome is correct for its ids.
    in_flight: set[int] = set()
    n_acked_deletes = sum(1 for line in lines if line.split()[1] == "D")
    if len(intents) > n_acked_deletes:
        in_flight = {int(v) for v in intents[-1].split()[2:]}

    # Reopen the collection in-process; this runs WAL replay and snapshot recovery.
    col = Collection.open(path)
    try:
        problems = []
        live = col.meta.user_to_internal
        vecs = col.store.array
        # Every acked, not-deleted record must be present with identical vector and tags.
        for i in sorted(acked - deleted - in_flight):
            j = live.get(f"w{i}")
            if j is None:
                problems.append(f"acked w{i} missing")
            elif not np.array_equal(vecs[j], vector_for(i)):
                problems.append(f"w{i} vector differs")
            elif col.meta.attributes(j) != attrs_for(i):
                problems.append(f"w{i} attributes differ")
        # Every acked delete must really be gone.
        for i in sorted(deleted):
            if f"w{i}" in live:
                problems.append(f"deleted w{i} present")
        # Any record present at all must have its exact vector, so no half-applied write survived.
        for uid, j in live.items():  # includes unacked-but-durable records
            if not np.array_equal(vecs[j], vector_for(int(uid[1:]))):
                problems.append(f"{uid} (unacked) has a corrupt vector")
        # Graph size must match the record count.
        if col.hnsw is not None and col.hnsw.n != col.n:
            problems.append(f"graph has {col.hnsw.n} nodes, collection {col.n}")
        # A sample of acked vectors must be found by searching for themselves.
        found = 0
        sample = sorted(acked - deleted - in_flight)[:50]
        for i in sample:
            found += col.search(vector_for(i), k=1, ef=64).ids == [f"w{i}"]
        if sample and found < 0.95 * len(sample):
            problems.append(f"only {found}/{len(sample)} acked vectors findable")
        # Verdict for this run.
        return {
            "ok": not problems,
            "seed": seed,
            "acks": len(lines),
            "acked_records": len(acked),
            "recovered_records": len(live),
            "snapshot_every": snapshot_every,
            "in_flight_delete": sorted(in_flight),
            "problems": problems[:20],
        }
    finally:
        col.close()


# Experiment entry called by bench.run: repeat crash_once N times and save a pass count.
def run(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    import shutil
    import tempfile

    from bench.common import save_json, system_info

    runs, base_seed = int(cfg["runs"]), int(cfg.get("seed", 0))
    results = []
    workdir = Path(tempfile.mkdtemp(prefix="vectra-crash-"))
    t0 = time.perf_counter()
    # Run every seed in a temp folder, print progress, and always clean up the folder.
    try:
        for r in range(runs):
            res = crash_once(workdir, base_seed + r, int(cfg.get("max_acks", 150)))
            results.append(res)
            status = "ok" if res["ok"] else f"FAIL {res.get('problems') or res.get('error')}"
            print(f"[crash {r + 1}/{runs}] acks={res.get('acks', 0)} {status}", flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    # Summarise the runs and write bench/results/<name>.json.
    passed = sum(r["ok"] for r in results)
    out = {
        "experiment": "crash",
        "config": cfg,
        "system": system_info(1),
        "passed": passed,
        "runs": runs,
        "seconds": time.perf_counter() - t0,
        "total_acked_records": sum(r.get("acked_records", 0) for r in results),
        "results": results,
    }
    save_json(name, out)
    print(f"{passed}/{runs} crash-recovery runs passed")
    return out


if __name__ == "__main__":
    # Dispatch: "worker" runs the child writer; anything else points to bench.run.
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit("usage: python -m bench.run configs/bench/crash.yaml")
