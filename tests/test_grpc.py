"""End-to-end over real gRPC: in-process server, real client stub."""

import grpc
import numpy as np
import pytest

from api.server import serve
from proto import siftdb_pb2 as pb
from proto import siftdb_pb2_grpc as pb_grpc


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    root = tmp_path_factory.mktemp("grpc_root")
    server, servicer, port = serve(root, port=0)  # OS-assigned port
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield pb_grpc.SiftDBStub(channel)
    channel.close()
    server.stop(grace=None)
    servicer.close()


@pytest.fixture(scope="module")
def data(rng):
    return rng.normal(size=(1000, 16)).astype(np.float32)


def _records(ids, vecs, prices=None):
    recs = []
    for j, (i, v) in enumerate(zip(ids, vecs, strict=True)):
        r = pb.Record(id=i, vector=v.tolist())
        if prices is not None:
            r.attributes["price"].num = prices[j]
            r.attributes["cat"].txt = f"c{j % 3}"
        recs.append(r)
    return recs


def test_end_to_end(stub, data):
    stub.CreateCollection(pb.CreateCollectionRequest(name="e2e", dim=16))
    ids = [f"doc{i}" for i in range(1000)]
    r = stub.Upsert(
        pb.UpsertRequest(
            collection="e2e",
            records=_records(ids, data, prices=[float(i) for i in range(1000)]),
        )
    )
    assert r.upserted == 1000

    rep = stub.Search(
        pb.SearchRequest(collection="e2e", vector=data[321].tolist(), k=5, include_attributes=True)
    )
    assert rep.hits[0].id == "doc321"
    assert rep.hits[0].distance == 0.0
    assert rep.hits[0].attributes["price"].num == 321.0
    assert rep.hits[0].attributes["cat"].txt == "c0"
    assert len(rep.hits) == 5
    assert rep.strategy == "flat"

    stub.Delete(pb.DeleteRequest(collection="e2e", ids=["doc321"]))
    rep = stub.Search(pb.SearchRequest(collection="e2e", vector=data[321].tolist(), k=5))
    assert all(h.id != "doc321" for h in rep.hits)

    s = stub.Stats(pb.StatsRequest(collection="e2e"))
    assert s.count == 999 and s.deleted == 1 and s.dim == 16


def test_status_codes(stub, data):
    with pytest.raises(grpc.RpcError) as e:
        stub.Search(pb.SearchRequest(collection="missing", vector=[0.0] * 16, k=5))
    assert e.value.code() == grpc.StatusCode.NOT_FOUND

    stub.CreateCollection(pb.CreateCollectionRequest(name="dup", dim=4))
    with pytest.raises(grpc.RpcError) as e:
        stub.CreateCollection(pb.CreateCollectionRequest(name="dup", dim=4))
    assert e.value.code() == grpc.StatusCode.ALREADY_EXISTS

    with pytest.raises(grpc.RpcError) as e:
        stub.Upsert(
            pb.UpsertRequest(collection="dup", records=[pb.Record(id="a", vector=[1.0, 2.0])])
        )
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    with pytest.raises(grpc.RpcError) as e:
        stub.CreateCollection(
            pb.CreateCollectionRequest(name="badopt", dim=4, options={"hnsw.bogus": "1"})
        )
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_bad_name_rejected(stub):
    with pytest.raises(grpc.RpcError) as e:
        stub.CreateCollection(pb.CreateCollectionRequest(name="../evil", dim=4))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
