"""Curio HTTP API (FastAPI).

Run locally:  uvicorn curio.api:app --reload --app-dir server
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import DEFAULT_ARTIFACTS, Engine, Rated

Kind = Literal["game", "accessory", "console", "other"]


class RatingIn(BaseModel):
    item_id: str = Field(min_length=1, max_length=20)
    rating: float = Field(ge=1, le=5)


class RecommendIn(BaseModel):
    persona: str | None = Field(default=None, max_length=32)
    ratings: list[RatingIn] = Field(default_factory=list, max_length=100)
    alpha: float | None = Field(default=None, ge=0, le=1)
    k: int = Field(default=12, ge=1, le=48)
    kinds: list[Kind] | None = None


class ExplainIn(BaseModel):
    item_id: str = Field(min_length=1, max_length=20)
    persona: str | None = Field(default=None, max_length=32)
    ratings: list[RatingIn] = Field(default_factory=list, max_length=100)
    alpha: float | None = Field(default=None, ge=0, le=1)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return Engine(Path(os.environ.get("CURIO_ARTIFACTS", DEFAULT_ARTIFACTS)))


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        get_engine()  # load the model once at boot, not on the first visitor's request
        yield

    app = FastAPI(title="Curio API", version="1.0.0", lifespan=lifespan,
                  description="Hybrid ALS + TF-IDF recommender over Amazon Video Games reviews.")
    origins = [o.strip() for o in os.environ.get("CLIENT_URL", "http://localhost:5173").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    def history(engine: Engine, persona: str | None, ratings: list[RatingIn]) -> list[Rated]:
        rated: list[Rated] = []
        if persona:
            try:
                rated = engine.persona_ratings(persona)
            except KeyError:
                raise HTTPException(404, f"Unknown persona '{persona}'")
        if ratings:
            try:
                extra = engine.to_rated([(r.item_id, r.rating) for r in ratings])
            except KeyError as e:
                raise HTTPException(404, f"Unknown product {e.args[0]}")
            merged = {r.index: r for r in rated}
            merged.update({r.index: r for r in extra})
            rated = list(merged.values())
        return rated

    def item_index(engine: Engine, item_id: str) -> int:
        i = engine.ix.get(item_id)
        if i is None:
            raise HTTPException(404, "Product not found")
        return i

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/stats")
    def stats(engine: Engine = Depends(get_engine)):
        return engine.stats()

    @app.get("/api/metrics")
    def metrics(engine: Engine = Depends(get_engine)):
        return engine.metrics

    @app.get("/api/personas")
    def personas(engine: Engine = Depends(get_engine)):
        out = []
        for p in engine.personas.values():
            out.append({
                "id": p["id"], "name": p["name"], "blurb": p["blurb"], "color": p["color"],
                "cold": not p["history"],
                "history": [{**engine.card(engine.ix[h["item_id"]]), "rating": h["rating"]} for h in p["history"]],
            })
        return out

    @app.post("/api/recommend")
    def recommend(body: RecommendIn, engine: Engine = Depends(get_engine)):
        rated = history(engine, body.persona, body.ratings)
        return engine.recommend(rated, body.alpha, body.k, set(body.kinds) if body.kinds else None)

    @app.post("/api/explain")
    def explain(body: ExplainIn, engine: Engine = Depends(get_engine)):
        i = item_index(engine, body.item_id)
        return engine.explain(i, history(engine, body.persona, body.ratings), body.alpha)

    @app.get("/api/items/{item_id}")
    def item(item_id: str, engine: Engine = Depends(get_engine)):
        return engine.detail(item_index(engine, item_id))

    @app.get("/api/items/{item_id}/similar")
    def similar(item_id: str, k: int = Query(8, ge=1, le=24), engine: Engine = Depends(get_engine)):
        return engine.similar(item_index(engine, item_id), k)

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=2, max_length=80), limit: int = Query(12, ge=1, le=40),
               engine: Engine = Depends(get_engine)):
        return engine.search(q, limit)

    @app.get("/api/onboarding")
    def onboarding(k: int = Query(24, ge=4, le=60), engine: Engine = Depends(get_engine)):
        return engine.onboarding(k)

    return app


app = create_app()
