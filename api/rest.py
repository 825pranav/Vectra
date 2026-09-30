"""Optional REST gateway for demos: JSON over HTTP -> the same collections.

    uv sync --extra rest
    uvicorn api.rest:create_app --factory --port 8080   # serves $VECTRA_ROOT

It talks to the engine in-process (not through gRPC) and maps engine errors to
HTTP 400/404/409, mirroring the gRPC status codes.
"""

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

ROOT = Path(os.environ.get("VECTRA_ROOT", "data/collections"))


class CreateBody(BaseModel):
    name: str
    dim: int
    options: dict[str, str] = Field(default_factory=dict)


class RecordBody(BaseModel):
    id: str
    vector: list[float]
    attributes: dict[str, float | str | bool] = Field(default_factory=dict)


class UpsertBody(BaseModel):
    records: list[RecordBody]


class SearchBody(BaseModel):
    vector: list[float]
    k: int = 10
    filter: str = ""
    ef: int = 0
    include_attributes: bool = False


class DeleteBody(BaseModel):
    ids: list[str]


def create_app(root: Path = ROOT) -> FastAPI:
    root.mkdir(parents=True, exist_ok=True)
    cols: dict[str, Collection] = {}
    lock = threading.Lock()
    for cfg in root.glob(f"*/{Collection.CONFIG_FILE}"):
        c = Collection.open(cfg.parent)
        cols[c.name] = c

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for c in cols.values():  # clean shutdown snapshots each collection
            c.close()

    app = FastAPI(title="vectra", version="0.1.0", lifespan=lifespan)

    def get(name: str) -> Collection:
        if name not in cols:
            raise HTTPException(404, f"collection {name!r} does not exist")
        return cols[name]

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

    @app.post("/collections/{name}/search")
    def search(name: str, body: SearchBody) -> dict[str, Any]:
        col = get(name)
        try:
            res = col.search(np.asarray(body.vector, dtype=np.float32), body.k,
                             body.filter or None, body.ef or None,
                             body.include_attributes)  # fmt: skip
        except InvalidArgument as e:
            raise HTTPException(400, str(e)) from e
        pairs = zip(res.ids, res.distances, strict=True)
        hits: list[dict[str, Any]] = [{"id": i, "distance": float(d)} for i, d in pairs]
        if res.attributes is not None:
            for h, a in zip(hits, res.attributes, strict=True):
                h["attributes"] = a
        return {"hits": hits, "strategy": res.strategy, "selectivity": res.selectivity}

    @app.post("/collections/{name}/delete")
    def delete(name: str, body: DeleteBody) -> dict[str, Any]:
        return {"deleted": get(name).delete(body.ids)}

    @app.get("/collections/{name}")
    def stats(name: str) -> dict[str, Any]:
        return get(name).stats()

    return app
