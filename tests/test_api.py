"""API tests against a tiny synthetic model, so they run without the real data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

N = 12


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    art = tmp_path_factory.mktemp("artifacts")
    rng = np.random.default_rng(0)
    titles = [f"Space Shooter {i}" if i < 6 else f"Farm Life {i}" for i in range(N)]
    pd.DataFrame({
        "item_id": [f"B{i:09d}" for i in range(N)],
        "title": titles,
        "store": ["Studio"] * N,
        "categories": [["PC", "Games"]] * 10 + [["PC", "Accessories", "Headsets"]] * 2,
        "price": [9.99] * N,
        "average_rating": [4.5] * N,
        "rating_number": list(range(N)),
        "image": [None] * N,
        "description": ["space lasers"] * 6 + ["cozy farming"] * 6,
        "train_count": list(range(N)),
    }).to_parquet(art / "items.parquet")
    Y = np.vstack([np.tile([1.0, 0.0], (6, 1)), np.tile([0.0, 1.0], (6, 1))]) + rng.normal(0, 0.05, (N, 2))
    np.save(art / "als_item_factors.npy", Y.astype(np.float32))
    X = sp.csr_matrix(np.vstack([np.tile([0.8, 0.6, 0, 0], (6, 1)), np.tile([0, 0, 0.6, 0.8], (6, 1))]).astype(np.float32))
    sp.save_npz(art / "tfidf.npz", X)
    (art / "tfidf_terms.json").write_text(json.dumps(["space", "lasers", "cozy", "farming"]))
    np.save(art / "popularity.npy", np.arange(N, dtype=np.float32))
    (art / "metrics.json").write_text(json.dumps({
        "k": 10, "alpha": 0.8, "dataset": {"users": 3}, "als_params": {"regularization": 0.1, "alpha": 10.0},
        "test": {m: {"hit_rate@10": h, "coverage@10": c} for m, h, c in
                 [("hybrid", 0.08, 0.19), ("als", 0.079, 0.137), ("tfidf", 0.02, 0.7), ("popularity", 0.027, 0.001)]},
    }))
    (art / "personas_history.json").write_text(json.dumps({
        "retro": [{"item_id": "B000000000", "rating": 5}, {"item_id": "B000000001", "rating": 4}],
        "sports": [{"item_id": "B000000007", "rating": 5}],
        "new": [],
    }))

    import os
    os.environ["CURIO_ARTIFACTS"] = str(art)
    from fastapi.testclient import TestClient
    from curio import api
    api.get_engine.cache_clear()
    with TestClient(api.create_app()) as c:
        yield c
    api.get_engine.cache_clear()


def test_health_and_stats(client):
    assert client.get("/api/health").json() == {"ok": True}
    s = client.get("/api/stats").json()
    assert s["alpha"] == 0.8
    assert s["lift_vs_popularity"] == pytest.approx(2.96, abs=0.01)


def test_personas_include_history_cards(client):
    ps = {p["id"]: p for p in client.get("/api/personas").json()}
    assert ps["new"]["cold"] is True
    assert [h["title"] for h in ps["retro"]["history"]] == ["Space Shooter 0", "Space Shooter 1"]


def test_persona_recommendations_follow_taste_and_skip_rated(client):
    res = client.post("/api/recommend", json={"persona": "retro", "k": 4}).json()
    assert res["mode"] == "hybrid"
    titles = [r["title"] for r in res["items"]]
    assert all(t.startswith("Space Shooter") for t in titles)
    assert "Space Shooter 0" not in titles and "Space Shooter 1" not in titles
    top = res["items"][0]
    assert top["score"] == pytest.approx(top["als"] + top["tfidf"], abs=1e-3)
    assert top["because"]["title"].startswith("Space Shooter")


def test_alpha_zero_is_pure_content(client):
    res = client.post("/api/recommend", json={"persona": "retro", "alpha": 0, "k": 3}).json()
    assert all(r["als"] == 0 for r in res["items"])
    assert "space" in res["items"][0]["terms"]


def test_cold_start_uses_popularity_and_kind_filter(client):
    res = client.post("/api/recommend", json={"k": 3, "kinds": ["game"]}).json()
    assert res["mode"] == "popularity"
    assert [r["title"] for r in res["items"]] == ["Farm Life 9", "Farm Life 8", "Farm Life 7"]


def test_onboarding_ratings_personalise(client):
    res = client.post("/api/recommend", json={"ratings": [{"item_id": "B000000008", "rating": 5}], "k": 3}).json()
    assert res["mode"] == "hybrid"
    assert all(r["title"].startswith("Farm Life") for r in res["items"])


def test_explain_and_similar(client):
    ex = client.post("/api/explain", json={"item_id": "B000000002", "persona": "retro"}).json()
    assert ex["rank"] <= 4
    assert {t["term"] for t in ex["shared_terms"]} == {"space", "lasers"}
    sim = client.get("/api/items/B000000002/similar?k=3").json()
    assert all(x["title"].startswith("Space") for x in sim["also_bought"] + sim["similar_content"])


def test_item_detail_search_onboarding(client):
    d = client.get("/api/items/B000000003").json()
    assert d["kind"] == "game" and d["top_terms"][0]["term"] in {"space", "lasers"}
    assert [x["title"] for x in client.get("/api/search?q=farm life&limit=2").json()] == ["Farm Life 11", "Farm Life 10"]
    ob = client.get("/api/onboarding?k=4").json()
    assert all(x["kind"] == "game" for x in ob)


def test_errors(client):
    assert client.get("/api/items/NOPE").status_code == 404
    assert client.post("/api/recommend", json={"persona": "ghost"}).status_code == 404
    assert client.post("/api/recommend", json={"ratings": [{"item_id": "X", "rating": 9}]}).status_code == 422
    assert client.post("/api/recommend", json={"ratings": [{"item_id": "NOPE", "rating": 4}]}).status_code == 404
    assert client.get("/api/search?q=a").status_code == 422


def test_cards_never_contain_nan(client):
    # Products with no categories have no platform; the card must say null, not NaN.
    import math
    for x in client.get("/api/onboarding?k=12").json() + client.get("/api/search?q=space&limit=12").json():
        assert all(not (isinstance(v, float) and not math.isfinite(v)) for v in x.values())
