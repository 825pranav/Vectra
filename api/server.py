"""gRPC server: thin translation layer over `engine.collection`.

All engine errors surface as structured gRPC status codes — never bare
exceptions to the client. The servicer holds a registry of open collections;
each collection enforces its own concurrency model (lock-free reads, single
writer lock), so the server-level lock only guards the registry itself.
"""

# Imports: gRPC + thread pool for the server, NumPy, the engine Collection and generated protobuf.
from __future__ import annotations

import argparse
import logging
import shutil
import threading
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

from engine.collection import Collection, InvalidArgument
from proto import vectra_pb2 as pb
from proto import vectra_pb2_grpc as pb_grpc
from storage.meta import AttrValue

# Module logger used for startup and "listening" messages.
log = logging.getLogger("vectra")


# Convert a protobuf Attribute oneof (num/txt/flag) into a Python value for the engine.
# An Attribute with nothing set is rejected as bad input.
def _attr_to_py(a: pb.Attribute) -> AttrValue:
    which = a.WhichOneof("value")
    if which == "num":
        return a.num
    if which == "txt":
        return a.txt
    if which == "flag":
        return a.flag
    raise InvalidArgument("attribute has no value")


# Convert a stored Python tag value back into a protobuf Attribute for the search reply.
def _attr_to_pb(v: AttrValue) -> pb.Attribute:
    if isinstance(v, bool):
        return pb.Attribute(flag=v)
    if isinstance(v, int | float):
        return pb.Attribute(num=float(v))
    return pb.Attribute(txt=str(v))


# gRPC front door: translates protobuf <-> NumPy and engine errors -> status codes.
# It holds no data logic; every real operation is delegated to a Collection.
class VectraServicer(pb_grpc.VectraServicer):
    # On start, open every collection folder under root that has a config file (this runs recovery).
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._registry_lock = threading.Lock()
        self.collections: dict[str, Collection] = {}
        for cfg in self.root.glob(f"*/{Collection.CONFIG_FILE}"):
            col = Collection.open(cfg.parent)
            self.collections[col.name] = col
            log.info("opened collection %r (%d records)", col.name, col.stats()["count"])

    # Look up an open collection by name, or abort the RPC with NOT_FOUND.
    def _get(self, name: str, context: grpc.ServicerContext) -> Collection:
        col = self.collections.get(name)
        if col is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"collection {name!r} does not exist")
        return col

    # ---- RPCs -----------------------------------------------------------

    # Create RPC: validate the name, then create the collection on disk and register it.
    def CreateCollection(self, request, context):
        # Reject empty names and names with slashes or a leading dot so they can't escape the root folder.
        name = request.name
        if not name or "/" in name or "\\" in name or name.startswith("."):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid collection name {name!r}")
        # Registry lock so two creates with the same name can't race.
        with self._registry_lock:
            if name in self.collections:
                context.abort(grpc.StatusCode.ALREADY_EXISTS, f"collection {name!r} exists")
            try:
                col = Collection.create(self.root / name, name, request.dim, dict(request.options))
            except (InvalidArgument, KeyError) as e:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            self.collections[name] = col
        return pb.CreateCollectionReply()

    # Drop RPC: unregister under the lock, then close the collection and delete its folder.
    def DropCollection(self, request, context):
        with self._registry_lock:
            col = self._get(request.name, context)
            del self.collections[request.name]
        col.close()
        shutil.rmtree(col.path, ignore_errors=True)
        return pb.DropCollectionReply()

    # Upsert RPC: protobuf records -> ids list, float32 matrix, tag dicts -> Collection.upsert().
    def Upsert(self, request, context):
        col = self._get(request.collection, context)
        # Empty request: nothing to write, reply 0 without touching the engine.
        if not request.records:
            return pb.UpsertReply(upserted=0)
        ids = [r.id for r in request.records]
        # Build the NumPy inputs; bad shapes or tags become INVALID_ARGUMENT instead of a crash.
        try:
            vecs = np.array([r.vector for r in request.records], dtype=np.float32)
            attrs = [{k: _attr_to_py(a) for k, a in r.attributes.items()} for r in request.records]
            n = col.upsert(ids, vecs, attrs)
        except InvalidArgument as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        return pb.UpsertReply(upserted=n)

    # Search RPC: query vector + k/filter/ef -> Collection.search() -> SearchReply.
    def Search(self, request, context):
        col = self._get(request.collection, context)
        # 0 / empty fields in protobuf mean "not set", so map them to defaults or None.
        try:
            res = col.search(
                np.asarray(request.vector, dtype=np.float32),
                k=request.k or 10,
                filter=request.filter or None,
                ef=request.ef or None,
                include_attributes=request.include_attributes,
            )
        except InvalidArgument as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        # Build one Hit per result (user id, distance, optional tags).
        hits = []
        for j, (uid, dist) in enumerate(zip(res.ids, res.distances, strict=True)):
            hit = pb.Hit(id=uid, distance=float(dist))
            if res.attributes is not None:
                for k, v in res.attributes[j].items():
                    hit.attributes[k].CopyFrom(_attr_to_pb(v))
            hits.append(hit)
        # The reply also says which strategy ran, the ef used and the estimated selectivity.
        return pb.SearchReply(
            hits=hits, strategy=res.strategy, ef=res.ef, selectivity=res.selectivity
        )

    # Delete RPC: pass user ids to the collection, reply with how many rows were found and deleted.
    def Delete(self, request, context):
        col = self._get(request.collection, context)
        return pb.DeleteReply(deleted=col.delete(list(request.ids)))

    # Stats RPC: copy the collection's stats dict into a StatsReply.
    def Stats(self, request, context):
        col = self._get(request.collection, context)
        s = col.stats()
        return pb.StatsReply(
            name=s["name"],
            dim=s["dim"],
            metric=s["metric"],
            index=s["index"],
            count=s["count"],
            deleted=s["deleted"],
            index_bytes=s["index_bytes"],
            lsn=s["lsn"],
        )

    # Close every open collection (used on shutdown).
    def close(self) -> None:
        for col in self.collections.values():
            col.close()


# Build the servicer (opens collections), a thread-pool gRPC server with 64 MB message caps,
# and start it. Returns (server, servicer, bound_port).
def serve(root: str | Path, host: str = "127.0.0.1", port: int = 50051, max_workers: int = 16):
    """Build and start a server; returns (server, servicer). Caller manages shutdown."""
    servicer = VectraServicer(Path(root))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    pb_grpc.add_VectraServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server, servicer, bound


# CLI entry point: parse flags, start the server, block until Ctrl+C, then shut down cleanly.
def main() -> None:
    ap = argparse.ArgumentParser(description="vectra gRPC server")
    ap.add_argument("--root", default="data/collections", help="collections directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server, servicer, bound = serve(args.root, args.host, args.port, args.workers)
    log.info("vectra listening on %s:%d", args.host, bound)
    # Block here serving requests; on Ctrl+C stop gRPC with a 2s grace, then close collections.
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2)
        servicer.close()


if __name__ == "__main__":
    main()
