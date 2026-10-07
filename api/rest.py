"""Optional REST gateway for demos: JSON over HTTP -> the same collections.

    uv sync --extra rest
    uvicorn api.rest:create_app --factory --port 8080   # serves $VECTRA_ROOT

It talks to the engine in-process (not through gRPC) and maps engine errors to
HTTP 400/404/409, mirroring the gRPC status codes.
"""

# Imports: FastAPI + pydantic for the HTTP/JSON layer, and the same engine Collection gRPC uses.
from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine.collection import Collection, InvalidArgument

# Where collection folders live; overridable with the VECTRA_ROOT environment variable.
ROOT = Path(os.environ.get("VECTRA_ROOT", "data/collections"))


# JSON body for creating a collection: name, dimension and string options.
class CreateBody(BaseModel):
    name: str
    dim: int
    options: dict[str, str] = Field(default_factory=dict)


# One item in a JSON upsert: user id, vector and a flat dict of tags.
class RecordBody(BaseModel):
    id: str
    vector: list[float]
    attributes: dict[str, float | str | bool] = Field(default_factory=dict)


# JSON body for upsert: a list of records.
class UpsertBody(BaseModel):
    records: list[RecordBody]


# JSON body for search; 0 / "" mean "use the engine default", same as the gRPC fields.
class SearchBody(BaseModel):
    vector: list[float]
    k: int = 10
    filter: str = ""
    ef: int = 0
    include_attributes: bool = False


# JSON body for delete: user ids to remove.
class DeleteBody(BaseModel):
    ids: list[str]


# Build the FastAPI app: open every existing collection in-process, then define the routes.
# It skips gRPC entirely and calls Collection directly.
def create_app(root: Path = ROOT) -> FastAPI:
    root.mkdir(parents=True, exist_ok=True)
    cols: dict[str, Collection] = {}
    lock = threading.Lock()
    # Open each folder that has a config file (this runs recovery for each collection).
    for cfg in root.glob(f"*/{Collection.CONFIG_FILE}"):
        c = Collection.open(cfg.parent)
        cols[c.name] = c

    # App lifespan: nothing to do at startup; on shutdown close (and snapshot) every collection.
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for c in cols.values():  # clean shutdown snapshots each collection
            c.close()

    app = FastAPI(title="vectra", version="0.1.0", lifespan=lifespan)

    # Find a collection by name or return HTTP 404.
    def get(name: str) -> Collection:
        if name not in cols:
            raise HTTPException(404, f"collection {name!r} does not exist")
        return cols[name]

    # POST /collections: validate the name, then create the collection under a lock (409 if it
    # exists).
    @app.post("/collections", status_code=201)
    def create(body: CreateBody) -> dict[str, Any]:
        if not body.name.isidentifier():
            raise HTTPException(400, f"invalid collection name {body.name!r}")
        with lock:
            if body.name in cols:
                raise HTTPException(409, f"collection {body.name!r} exists")
            try:
                cols[body.name] = Collection.create(root / body.name, body.name, body.dim,
                                                    dict(body.options))  # fmt: skip
            except (InvalidArgument, KeyError) as e:
                raise HTTPException(400, str(e)) from e
        return {"name": body.name}

    # POST upsert: JSON records -> ids, float32 matrix, tag dicts -> Collection.upsert(); 400 if
    # bad.
    @app.post("/collections/{name}/upsert")
    def upsert(name: str, body: UpsertBody) -> dict[str, Any]:
        col = get(name)
        try:
            n = col.upsert(
                [r.id for r in body.records],
                np.array([r.vector for r in body.records], dtype=np.float32),
                [dict(r.attributes) for r in body.records],
            )
        except InvalidArgument as e:
            raise HTTPException(400, str(e)) from e
        return {"upserted": n}

    # POST search: run Collection.search() and turn the result into JSON hits plus
    # strategy/selectivity.
    @app.post("/collections/{name}/search")
    def search(name: str, body: SearchBody) -> dict[str, Any]:
        col = get(name)
        try:
            res = col.search(np.asarray(body.vector, dtype=np.float32), body.k,
                             body.filter or None, body.ef or None,
                             body.include_attributes)  # fmt: skip
        except InvalidArgument as e:
            raise HTTPException(400, str(e)) from e
        # Pair each user id with its distance, then attach tags if they were requested.
        pairs = zip(res.ids, res.distances, strict=True)
        hits: list[dict[str, Any]] = [{"id": i, "distance": float(d)} for i, d in pairs]
        if res.attributes is not None:
            for h, a in zip(hits, res.attributes, strict=True):
                h["attributes"] = a
        return {"hits": hits, "strategy": res.strategy, "selectivity": res.selectivity}

    # POST delete: remove by user id and return how many were found.
    @app.post("/collections/{name}/delete")
    def delete(name: str, body: DeleteBody) -> dict[str, Any]:
        return {"deleted": get(name).delete(body.ids)}

    # GET /collections/{name}: return the collection's stats dict.
    @app.get("/collections/{name}")
    def stats(name: str) -> dict[str, Any]:
        return get(name).stats()

    return app
