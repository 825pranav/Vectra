"""Thin Python client over the gRPC stub.

from api.client import Client
db = Client("127.0.0.1:50051")
db.create_collection("docs", dim=384, metric="cosine")
db.upsert("docs", ["a", "b"], vectors, [{"price": 10.0}, {"price": 99.0}])
hits = db.search("docs", query, k=5, filter="price < 50")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import grpc
import numpy as np

from proto import siftdb_pb2 as pb
from proto import siftdb_pb2_grpc as pb_grpc


def _attr(v: Any) -> pb.Attribute:
    if isinstance(v, bool):
        return pb.Attribute(flag=v)
    if isinstance(v, int | float):
        return pb.Attribute(num=float(v))
    return pb.Attribute(txt=str(v))


def _py(a: pb.Attribute) -> Any:
    return getattr(a, a.WhichOneof("value"))


class Client:
    def __init__(self, target: str = "127.0.0.1:50051") -> None:
        self.channel = grpc.insecure_channel(
            target, options=[("grpc.max_send_message_length", 64 * 1024 * 1024)]
        )
        self.stub = pb_grpc.SiftDBStub(self.channel)

    def create_collection(self, name: str, dim: int, **options: Any) -> None:
        opts = {k.replace("__", "."): str(v) for k, v in options.items()}
        self.stub.CreateCollection(pb.CreateCollectionRequest(name=name, dim=dim, options=opts))

    def drop_collection(self, name: str) -> None:
        self.stub.DropCollection(pb.DropCollectionRequest(name=name))

    def upsert(
        self,
        collection: str,
        ids: Sequence[str],
        vectors: np.ndarray,
        attributes: Sequence[dict[str, Any]] | None = None,
        batch: int = 1000,
    ) -> int:
        vectors = np.asarray(vectors, dtype=np.float32)
        total = 0
        for s in range(0, len(ids), batch):
            recs = []
            for j in range(s, min(s + batch, len(ids))):
                r = pb.Record(id=str(ids[j]), vector=vectors[j].tolist())
                for k, v in (attributes[j] if attributes else {}).items():
                    r.attributes[k].CopyFrom(_attr(v))
                recs.append(r)
            rep = self.stub.Upsert(pb.UpsertRequest(collection=collection, records=recs))
            total += rep.upserted
        return total

    def search(
        self,
        collection: str,
        vector: np.ndarray,
        k: int = 10,
        filter: str = "",
        ef: int = 0,
        include_attributes: bool = False,
    ) -> list[dict[str, Any]]:
        rep = self.stub.Search(
            pb.SearchRequest(
                collection=collection,
                vector=np.asarray(vector, dtype=np.float32).tolist(),
                k=k,
                filter=filter,
                ef=ef,
                include_attributes=include_attributes,
            )
        )
        return [
            {
                "id": h.id,
                "distance": h.distance,
                **(
                    {"attributes": {k: _py(a) for k, a in h.attributes.items()}}
                    if include_attributes
                    else {}
                ),  # fmt: skip
            }
            for h in rep.hits
        ]

    def delete(self, collection: str, ids: Sequence[str]) -> int:
        return self.stub.Delete(pb.DeleteRequest(collection=collection, ids=list(ids))).deleted

    def stats(self, collection: str) -> dict[str, Any]:
        s = self.stub.Stats(pb.StatsRequest(collection=collection))
        return {f.name: v for f, v in s.ListFields()}

    def close(self) -> None:
        self.channel.close()
