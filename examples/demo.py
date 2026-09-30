"""End-to-end demo: semantic search over MS MARCO passages through the gRPC API.

    uv sync --extra data                        # sentence-transformers for query embedding
    uv run python -m bench.datasets.msmarco     # once: passage embeddings (GPU if present)
    uv run python -m examples.demo

Starts a siftdb server, loads 50,000 real passages (their MiniLM embeddings plus
two attributes) through ``Upsert``, then runs natural-language queries, filtered
queries, a delete, and finally a kill -9 crash-recovery cycle. Query text is
embedded on the CPU; passage embeddings come from the dataset step.
"""

from __future__ import annotations

import argparse
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

from api.client import Client
from api.server import serve
from bench.datasets.msmarco import BASE_URL, DATA, MODEL, _stream_tsv

QUERIES = [
    "what causes the northern lights",
    "how long does it take to boil an egg",
    "symptoms of vitamin d deficiency",
    "who invented the telephone",
]


def passages(n: int) -> list[str]:
    cache = DATA / f"passages_{n}.txt"
    if not cache.exists():
        texts = _stream_tsv(f"{BASE_URL}/collection.tar.gz", "collection.tsv", n)
        cache.write_text("\n".join(t.replace("\n", " ") for t in texts), encoding="utf-8")
    return cache.read_text(encoding="utf-8").split("\n")


def banner(text: str) -> None:
    print(f"\n\033[1m{text}\033[0m")


def show(hits: list[dict], texts: list[str], ms: float, note: str = "") -> None:
    print(f"   {len(hits)} hits in {ms:.2f} ms round-trip {note}")
    for rank, h in enumerate(hits, 1):
        t = textwrap.shorten(texts[int(h["id"])], 110, placeholder=" ...")
        attrs = h.get("attributes", {})
        meta = f"  [words={attrs['words']:.0f}]" if "words" in attrs else ""
        print(f"   {rank}. d={h['distance']:.3f}{meta}  {t}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50_000, help="passages to load")
    args = ap.parse_args()

    banner("siftdb demo: 50k MS MARCO passages, MiniLM embeddings, gRPC")
    texts = passages(args.n)
    emb = np.load(DATA / "base.npy", mmap_mode="r")[: args.n]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cpu")
    qvecs = model.encode(QUERIES, normalize_embeddings=True)

    root = Path(tempfile.mkdtemp(prefix="siftdb-demo-"))
    server, servicer, port = serve(root, port=0)
    db = Client(f"127.0.0.1:{port}")
    print(f"server listening on 127.0.0.1:{port}, data in {root}")
    try:
        banner("1. CreateCollection  passages (dim=384, cosine, HNSW)")
        db.create_collection("passages", dim=384, metric="cosine")

        banner(f"2. Upsert {args.n:,} passages with attributes (words, has_digits)")
        attrs = [{"words": float(len(t.split())), "has_digits": any(c.isdigit() for c in t)}
                 for t in texts]  # fmt: skip
        t0 = time.perf_counter()
        n = db.upsert("passages", [str(i) for i in range(args.n)], emb, attrs, batch=2000)
        dt = time.perf_counter() - t0
        print(f"   {n:,} records in {dt:.1f} s ({n / dt:,.0f}/s, WAL-logged + HNSW-indexed)")

        banner("3. Semantic search (query text -> MiniLM on CPU -> gRPC Search, k=3)")
        for q, v in zip(QUERIES, qvecs, strict=True):
            db.search("passages", v, k=3)  # warm
            t0 = time.perf_counter()
            hits = db.search("passages", v, k=3, include_attributes=True)
            print(f'\n  "{q}"')
            show(hits, texts, (time.perf_counter() - t0) * 1e3)

        banner("4. Filtered search: same query, different filters (planner picks a strategy)")
        v = qvecs[0]
        stub = db.stub
        from proto import siftdb_pb2 as pb

        for flt in ["words < 40", "words >= 40 AND has_digits == true", "words == 17"]:
            t0 = time.perf_counter()
            rep = stub.Search(pb.SearchRequest(collection="passages", vector=v.tolist(), k=3,
                                               filter=flt, include_attributes=True))  # fmt: skip
            ms = (time.perf_counter() - t0) * 1e3
            hits = [
                {
                    "id": h.id,
                    "distance": h.distance,
                    "attributes": {
                        k: getattr(a, a.WhichOneof("value")) for k, a in h.attributes.items()
                    },
                }
                for h in rep.hits
            ]
            print(f"\n  filter: {flt}")
            show(hits, texts, ms, f"-> strategy={rep.strategy}, selectivity~{rep.selectivity:.4f}")

        banner("5. Delete the top hit, search again")
        top = db.search("passages", qvecs[0], k=1)[0]["id"]
        print(f"   delete id={top}: deleted={db.delete('passages', [top])}")
        t0 = time.perf_counter()
        hits = db.search("passages", qvecs[0], k=2)
        show(hits, texts, (time.perf_counter() - t0) * 1e3, "(after delete)")

        banner("6. Stats")
        for k, val in db.stats("passages").items():
            print(f"   {k:12s} {val}")

        banner("7. Error handling (structured gRPC status codes)")
        import grpc

        for label, call in [
            ("bad filter", lambda: db.search("passages", v, k=3, filter="words <")),
            ("wrong dim", lambda: db.search("passages", v[:10], k=3)),
            ("no such collection", lambda: db.search("nope", v, k=3)),
        ]:
            try:
                call()
            except grpc.RpcError as e:
                print(f"   {label:20s} -> {e.code().name}: {e.details()}")
    finally:
        db.close()
        server.stop(grace=None)
        servicer.close()

    banner("8. Crash recovery: kill -9 a writer mid-stream, reopen, verify every ack")
    from bench.crash import crash_once

    for seed in (7, 8, 9):
        r = crash_once(Path(tempfile.mkdtemp(prefix="siftdb-crash-")), seed)
        verdict = "0 acknowledged writes lost" if r["ok"] else f"FAILED {r['problems']}"
        print(f"   run {seed}: killed after {r['acks']} acked calls ({r['acked_records']} records "
              f"upserted, some later deleted); reopened with {r['recovered_records']} live "
              f"records -> {verdict}")  # fmt: skip


if __name__ == "__main__":
    main()
