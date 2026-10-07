"""Browser dashboard for demos: semantic + filtered search over real MS MARCO passages.

    uv sync --extra rest --extra data
    uv run python -m examples.dashboard            # http://127.0.0.1:8080
    uv run python -m examples.dashboard --n 50000  # more passages

First run streams the first ``--n`` passages from MS MARCO (no full download), embeds them with
MiniLM on this machine and loads them into a collection under data/dashboard/. Later runs just
reopen that collection, which is itself a recovery: newest snapshot + WAL replay.

The page lets you search by text, add a filter, compare the planner against each fixed filter
strategy, add and delete passages, and run a kill -9 crash-recovery cycle. It calls the engine
in-process (like api/rest.py); the gRPC path is shown by examples/demo.py.
"""

from __future__ import annotations

# Stdlib helpers, NumPy, FastAPI for the page and JSON routes, and the engine pieces.
import argparse
import contextlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from bench.datasets.msmarco import DATA, MODEL
from engine.collection import Collection, InvalidArgument
from examples.demo import passages

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("dashboard.html")
STRATEGIES = ["post_filter", "bitmap", "brute_force"]


# JSON bodies for the routes; 0 / "" mean "engine default", as in the gRPC and REST APIs.
class SearchBody(BaseModel):
    query: str
    k: int = 5
    filter: str = ""
    ef: int = 0


class AddBody(BaseModel):
    text: str


class DeleteBody(BaseModel):
    id: str


class CrashBody(BaseModel):
    seed: int = 7


# The two tags every passage carries; filters such as "words < 40 AND has_digits == true" use them.
def attributes(text: str) -> dict[str, Any]:
    return {"words": float(len(text.split())), "has_digits": any(c.isdigit() for c in text)}


# Passage embeddings: reuse the benchmark's base.npy if it is on disk, else embed here and cache.
def embeddings(texts: list[str], model) -> np.ndarray:
    base = DATA / "base.npy"
    if base.exists():
        emb = np.load(base, mmap_mode="r")
        if emb.shape[0] >= len(texts):
            return np.asarray(emb[: len(texts)])
    cache = DATA / f"dashboard_emb_{len(texts)}.npy"
    if not cache.exists():
        print(f"embedding {len(texts):,} passages with {MODEL} (once, cached to {cache})")
        emb = model.encode(texts, batch_size=128, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)  # fmt: skip
        np.save(cache, emb.astype(np.float32))
    return np.load(cache)


# Open the dashboard collection, or create and fill it on first run.
def load_collection(path: Path, texts: list[str], model) -> Collection:
    if (path / Collection.CONFIG_FILE).exists():
        t0 = time.perf_counter()
        col = Collection.open(path)
        ms = (time.perf_counter() - t0) * 1e3
        print(f"reopened {col.stats()['count']:,} records in {ms:.0f} ms (snapshot + WAL replay)")
        return col
    emb = embeddings(texts, model)
    col = Collection.create(path, "passages", emb.shape[1], {"metric": "cosine"})
    t0 = time.perf_counter()
    # Same write path as the API: every batch is WAL-logged and fsynced before it is indexed.
    for i in range(0, len(texts), 2000):
        j = min(i + 2000, len(texts))
        col.upsert([str(r) for r in range(i, j)], emb[i:j], [attributes(t) for t in texts[i:j]])
        print(f"\r  loaded {j:,}/{len(texts):,}", end="", flush=True)
    print(f"\n  {len(texts):,} records in {time.perf_counter() - t0:.1f} s")
    col.snapshot()
    return col


# Build the app: one collection, the MiniLM query encoder, and passage text looked up by id.
def create_app(n: int, root: Path) -> FastAPI:
    from sentence_transformers import SentenceTransformer

    DATA.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    texts = passages(n)
    model = SentenceTransformer(MODEL)
    col = load_collection(root / "passages", texts, model)
    # Passages added from the page are kept beside the collection so they survive restarts.
    user_file = root / "user_texts.json"
    user_texts: dict[str, str] = (
        json.loads(user_file.read_text(encoding="utf-8")) if user_file.exists() else {}
    )

    def text_of(pid: str) -> str:
        if pid in user_texts:
            return user_texts[pid]
        return texts[int(pid)] if pid.isdigit() and int(pid) < len(texts) else ""

    def embed(query: str) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        v = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float32), (time.perf_counter() - t0) * 1e3

    # Warm up the Numba kernels so the first search on stage is not a JIT compile.
    col.search(np.asarray(embed("warm up")[0]), k=5)

    # On shutdown close the collection, which writes a snapshot.
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        col.close()

    app = FastAPI(title="vectra dashboard", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    # Collection stats plus where the data came from.
    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return {**col.stats(), "model": MODEL, "corpus": f"MS MARCO, first {n:,} passages",
                "path": str(col.path)}  # fmt: skip

    # Text -> MiniLM vector -> engine search; returns hits, the chosen strategy and timings.
    @app.post("/api/search")
    def search(body: SearchBody) -> dict[str, Any]:
        v, embed_ms = embed(body.query)
        t0 = time.perf_counter()
        try:
            res = col.search(v, body.k, body.filter or None, body.ef or None, True)
        except InvalidArgument as e:
            raise HTTPException(400, str(e)) from e
        search_ms = (time.perf_counter() - t0) * 1e3
        hits = [
            {"id": i, "distance": float(d), "attributes": a, "text": text_of(i)}
            for i, d, a in zip(res.ids, res.distances, res.attributes or [], strict=True)
        ]
        return {"hits": hits, "strategy": res.strategy, "selectivity": res.selectivity,
                "ef": res.ef, "embed_ms": embed_ms, "search_ms": search_ms}  # fmt: skip

    # Same filtered query under the planner and under each forced strategy: median latency over
    # 5 runs, and recall@k against brute force (exact over the matching rows).
    @app.post("/api/compare")
    def compare(body: SearchBody) -> dict[str, Any]:
        if not body.filter:
            raise HTTPException(400, "compare needs a filter")
        v, _ = embed(body.query)
        rows = []
        try:
            exact = set(col.search(v, body.k, body.filter, strategy="brute_force").ids)
            for strategy in [None, *STRATEGIES]:
                times, res = [], None
                for _ in range(5):
                    t0 = time.perf_counter()
                    res = col.search(v, body.k, body.filter, body.ef or None, strategy=strategy)
                    times.append((time.perf_counter() - t0) * 1e3)
                assert res is not None
                rows.append({
                    "strategy": "planner" if strategy is None else strategy,
                    "picked": res.strategy,
                    "ms": float(np.median(times)),
                    "hits": len(res.ids),
                    "recall": len(exact & set(res.ids)) / max(len(exact), 1),
                })
        except InvalidArgument as e:
            raise HTTPException(400, str(e)) from e
        return {"rows": rows, "selectivity": res.selectivity, "matches": len(exact)}

    # Add one passage: same upsert path as a bulk load (WAL + fsync, then indexed), then searchable.
    @app.post("/api/add")
    def add(body: AddBody) -> dict[str, Any]:
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "empty text")
        pid = f"user-{len(user_texts) + 1}"
        v, _ = embed(text)
        col.upsert([pid], v[None, :], [attributes(text)])
        user_texts[pid] = text
        user_file.write_text(json.dumps(user_texts), encoding="utf-8")
        return {"id": pid, "attributes": attributes(text)}

    # Delete by id (a tombstone): the id stops appearing in results immediately.
    @app.post("/api/delete")
    def delete(body: DeleteBody) -> dict[str, Any]:
        return {"deleted": col.delete([body.id])}

    # One kill -9 cycle from bench/crash.py: a writer subprocess is killed at a random point,
    # the collection is reopened, and every acknowledged write is checked.
    @app.post("/api/crash")
    def crash(body: CrashBody) -> dict[str, Any]:
        from bench.crash import crash_once

        t0 = time.perf_counter()
        r = crash_once(Path(tempfile.mkdtemp(prefix="vectra-crash-")), body.seed)
        return {**r, "seconds": time.perf_counter() - t0}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=20_000, help="passages to load on first run")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "dashboard",
                    help="where the dashboard collection lives (delete it to reload)")
    args = ap.parse_args()
    import uvicorn

    app = create_app(args.n, args.root)
    print(f"\nvectra dashboard -> http://127.0.0.1:{args.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
