"""Quick gRPC tour with random vectors: no dataset, no model, a few seconds.

    uv run python -m examples.grpc_quick

Starts an in-process gRPC server on a free port in a temp folder, then uses the Python
client exactly as the README's quick start does: create a collection, upsert 1,000 vectors
with attributes, a filtered search, stats, and two calls that fail with structured status
codes. The full demo over real passages is examples/demo.py.
"""

from __future__ import annotations

# Stdlib, NumPy and grpc for status codes, plus the client and the server entry point.
import tempfile
import time
from pathlib import Path

import grpc
import numpy as np

from api.client import Client
from api.server import serve


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="vectra-grpc-"))
    server, servicer, port = serve(root, port=0)
    db = Client(f"127.0.0.1:{port}")
    print(f"gRPC server on 127.0.0.1:{port}, data in {root}\n")
    try:
        # CreateCollection + Upsert: 1,000 random 128-d vectors, each with a price and category.
        db.create_collection("products", dim=128)
        rng = np.random.default_rng(0)
        vecs = rng.random((1000, 128), dtype=np.float32)
        attrs = [{"price": float(i), "category": "shoes" if i % 2 else "hats"} for i in range(1000)]
        print("Upsert  ->", db.upsert("products", [f"p{i}" for i in range(1000)], vecs, attrs))

        # Filtered Search: nearest neighbours of p0 among cheap shoes.
        flt = 'price < 500 AND category == "shoes"'
        t0 = time.perf_counter()
        hits = db.search("products", vecs[0], k=3, filter=flt, include_attributes=True)
        print(f"Search  -> filter {flt!r}, {(time.perf_counter() - t0) * 1e3:.2f} ms round-trip")
        for h in hits:
            print(f"           {h['id']:5s} distance {h['distance']:.2f}  {h['attributes']}")
        print("Stats   ->", db.stats("products"))

        # Bad input comes back as gRPC status codes, not crashes.
        for label, call in [
            ("wrong dim", lambda: db.search("products", vecs[0][:10], k=3)),
            ("bad filter", lambda: db.search("products", vecs[0], k=3, filter="price <")),
        ]:
            try:
                call()
            except grpc.RpcError as e:
                print(f"Error   -> {label}: {e.code().name}: {e.details()}")
    finally:
        db.close()
        server.stop(grace=None)
        servicer.close()


if __name__ == "__main__":
    main()
