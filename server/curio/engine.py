"""Serving engine: loads the trained files and answers recommendation queries.

Training lives in scripts/train.py. This module only needs numpy, scipy and
pandas, so the API image stays small (no implicit or scikit-learn at runtime).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = HERE.parent / "artifacts"
PERSONAS_FILE = HERE / "personas.json"
PERSONAS_HISTORY_FILE = "personas_history.json"
BORING_WORDS = {"games", "game", "legacy", "systems", "video", "accessories", "pc", "features", "product", "new"}


def rating_weight(r):
    return np.asarray(r, dtype=np.float32) / 5.0


def profile_weight(r):
    return np.clip(np.asarray(r, dtype=np.float32) - 2.5, 0.1, None)


def fold_in(Y: np.ndarray, YtY: np.ndarray, items: np.ndarray, values: np.ndarray,
            reg: float, conf_alpha: float) -> np.ndarray:
    """Implicit-ALS user vector (Hu, Koren & Volinsky 2008) for a user the model never saw.

    Solves (YᵀY + Yᵢᵀ(Cᵢ − I)Yᵢ + λI) x = YᵢᵀCᵢ·1, the same step the trainer uses.
    """
    f = Y.shape[1]
    if len(items) == 0:
        return np.zeros(f, dtype=np.float32)
    c = conf_alpha * np.asarray(values, dtype=np.float64)
    Yi = Y[items].astype(np.float64)
    A = YtY + reg * np.eye(f) + (Yi * (c - 1.0)[:, None]).T @ Yi
    b = Yi.T @ c
    return np.linalg.solve(A, b).astype(np.float32)


def minmax(s: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    s = s.astype(np.float32, copy=True)
    keep = np.ones(len(s), dtype=bool)
    keep[exclude] = False
    if not keep.any():
        return np.zeros_like(s)
    lo, hi = s[keep].min(), s[keep].max()
    out = (s - lo) / (hi - lo if hi - lo > 1e-9 else 1.0)
    out[exclude] = -1.0
    return out


def platform_of(categories: list[str]) -> str | None:
    if not categories:
        return None
    if categories[0] == "Legacy Systems" and len(categories) >= 3:
        return categories[2]
    return categories[0]


def kind_of(categories: list[str]) -> str:
    cats = set(categories)
    if "Games" in cats:
        return "game"
    if "Consoles" in cats or any(c.endswith("Consoles") for c in cats):
        return "console"
    if cats:
        return "accessory"
    return "other"


@dataclass
class Rated:
    index: int
    rating: float


class Engine:
    def __init__(self, artifacts: Path = DEFAULT_ARTIFACTS):
        artifacts = Path(artifacts)
        self.metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
        p = self.metrics["als_params"]
        self.reg, self.conf_alpha = float(p["regularization"]), float(p["alpha"])
        self.default_alpha = float(self.metrics["alpha"])

        items = pd.read_parquet(artifacts / "items.parquet")
        items["categories"] = items.categories.map(lambda c: [str(x) for x in c])
        items["platform"] = items.categories.map(platform_of)
        items["kind"] = items.categories.map(kind_of)
        self.items = items.reset_index(drop=True)
        self.ix = {iid: i for i, iid in enumerate(self.items.item_id)}
        self.n = len(self.items)

        self.Y = np.load(artifacts / "als_item_factors.npy").astype(np.float32)
        self.YtY = (self.Y.T.astype(np.float64) @ self.Y.astype(np.float64))
        norms = np.linalg.norm(self.Y, axis=1, keepdims=True)
        self.Y_unit = self.Y / np.where(norms > 0, norms, 1.0)

        self.X = sp.load_npz(artifacts / "tfidf.npz").tocsr().astype(np.float32)
        self.XT = self.X.T.tocsr()
        self.terms = np.array(json.loads((artifacts / "tfidf_terms.json").read_text(encoding="utf-8")))
        # Category scaffolding ("legacy systems", "pc games") is real signal but a poor explanation.
        self.explainable = np.array([not (set(t.split()) & BORING_WORDS) for t in self.terms])

        counts = np.load(artifacts / "popularity.npy").astype(np.float32)
        self.pop_counts = counts
        pop = np.log1p(counts)
        self.pop_score = pop / max(pop.max(), 1e-9)
        self.titles_lower = self.items.title.str.lower().to_numpy()

        self.personas = self._load_personas(artifacts)

    # ---------- personas ----------

    def _load_personas(self, artifacts: Path) -> dict[str, dict]:
        meta = json.loads(PERSONAS_FILE.read_text(encoding="utf-8"))
        hist_path = artifacts / PERSONAS_HISTORY_FILE
        histories = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {}
        out = {}
        for p in meta:
            history = [h for h in histories.get(p["id"], []) if h["item_id"] in self.ix]
            out[p["id"]] = {**p, "history": history}
        return out

    def persona_ratings(self, persona_id: str) -> list[Rated]:
        p = self.personas.get(persona_id)
        if p is None:
            raise KeyError(persona_id)
        return [Rated(self.ix[h["item_id"]], float(h["rating"])) for h in p["history"]]

    def to_rated(self, ratings: list[tuple[str, float]]) -> list[Rated]:
        out: dict[int, Rated] = {}
        for item_id, rating in ratings:
            i = self.ix.get(item_id)
            if i is None:
                raise KeyError(item_id)
            out[i] = Rated(i, float(rating))
        return list(out.values())

    # ---------- cards ----------

    def card(self, i: int) -> dict:
        r = self.items.iloc[i]
        return {
            "id": r.item_id,
            "title": r.title,
            "store": present(r.store),
            "platform": present(r.platform),
            "kind": r.kind,
            "image": present(r.image),
            "price": None if pd.isna(r.price) else float(r.price),
            "average_rating": None if pd.isna(r.average_rating) else float(r.average_rating),
            "rating_count": int(r.rating_number),
        }

    def detail(self, i: int) -> dict:
        r = self.items.iloc[i]
        return {
            **self.card(i),
            "categories": list(r.categories),
            "description": r.description,
            "top_terms": self.item_terms(i, 8),
            "popularity_rank": int((self.pop_counts > self.pop_counts[i]).sum()) + 1,
        }

    # ---------- signals ----------

    def signals(self, rated: list[Rated]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (als_norm, tfidf_norm, profile, user_vec) for a rating history."""
        idx = np.array([r.index for r in rated], dtype=np.int64)
        vals = np.array([r.rating for r in rated], dtype=np.float32)
        u = fold_in(self.Y, self.YtY, idx, rating_weight(vals), self.reg, self.conf_alpha)
        als = minmax(self.Y @ u, idx)
        w = sp.csr_matrix((profile_weight(vals), (np.zeros(len(idx), dtype=np.int64), idx)), shape=(1, self.n))
        prof = w @ self.X
        norm = np.sqrt(prof.multiply(prof).sum())
        prof = prof / norm if norm > 0 else prof
        tf = minmax(np.asarray((self.X @ prof.T).todense()).ravel(), idx)
        return als, tf, sp.csr_matrix(prof), u

    def recommend(self, rated: list[Rated], alpha: float | None = None, k: int = 12,
                  kinds: set[str] | None = None) -> dict:
        alpha = self.default_alpha if alpha is None else float(alpha)
        exclude = np.array([r.index for r in rated], dtype=np.int64)
        allowed = None
        if kinds:
            allowed = self.items.kind.isin(kinds).to_numpy()

        if not rated:
            score = self.pop_score.copy()
            if allowed is not None:
                score[~allowed] = -1
            top = np.argsort(-score)[:k]
            return {"mode": "popularity", "alpha": alpha, "items": [
                {**self.card(i), "score": round(float(score[i]), 4), "als": None, "tfidf": None,
                 "reason": "Popular with shoppers right now"} for i in top]}

        als, tf, prof, _ = self.signals(rated)
        score = alpha * als + (1 - alpha) * tf
        if allowed is not None:
            score[~allowed] = -2
        top = np.argpartition(-score, k)[:k]
        top = top[np.argsort(-score[top])]
        out = []
        for i in top:
            terms = self.shared_terms(prof, i, 4)
            because = self.nearest_rated(i, rated)
            out.append({
                **self.card(i),
                "score": round(float(score[i]), 4),
                "als": round(float(alpha * als[i]), 4),
                "tfidf": round(float((1 - alpha) * tf[i]), 4),
                "because": because,
                "terms": [t["term"] for t in terms],
                "reason": self.reason(als[i], tf[i], alpha, because, terms),
            })
        return {"mode": "hybrid", "alpha": alpha, "items": out}

    # ---------- explanations ----------

    def item_terms(self, i: int, n: int) -> list[dict]:
        row = self.X[i]
        keep = self.explainable[row.indices]
        row = sp.csr_matrix((row.data[keep], row.indices[keep], [0, int(keep.sum())]), shape=row.shape)
        order = np.argsort(-row.data)[:n]
        return [{"term": str(self.terms[row.indices[j]]), "weight": round(float(row.data[j]), 4)} for j in order]

    def shared_terms(self, prof: sp.csr_matrix, i: int, n: int) -> list[dict]:
        overlap = prof.multiply(self.X[i]).tocsr()
        if overlap.nnz == 0:
            return []
        overlap.data[~self.explainable[overlap.indices]] = 0
        overlap.eliminate_zeros()
        if overlap.nnz == 0:
            return []
        order = np.argsort(-overlap.data)[:n]
        return [{"term": str(self.terms[overlap.indices[j]]), "weight": round(float(overlap.data[j]), 5)} for j in order]

    def nearest_rated(self, i: int, rated: list[Rated]) -> dict | None:
        liked = [r for r in rated if r.rating >= 4] or rated
        if not liked:
            return None
        idx = np.array([r.index for r in liked])
        sims = self.Y_unit[idx] @ self.Y_unit[i]
        j = int(idx[int(np.argmax(sims))])
        return {"id": self.items.item_id[j], "title": self.items.title[j]}

    @staticmethod
    def reason(als: float, tf: float, alpha: float, because, terms) -> str:
        a, t = alpha * als, (1 - alpha) * tf
        if a >= t and because:
            return f"Shoppers who liked {short(because['title'])} also bought this"
        if terms:
            return "Matches what you like: " + ", ".join(x["term"] for x in terms[:3])
        return "Strong overall match for your history"

    def explain(self, i: int, rated: list[Rated], alpha: float | None = None) -> dict:
        alpha = self.default_alpha if alpha is None else float(alpha)
        if not rated:
            return {"mode": "popularity", "alpha": alpha, "popularity_rank": self.detail(i)["popularity_rank"]}
        als, tf, prof, _ = self.signals(rated)
        score = alpha * als + (1 - alpha) * tf
        rank = int((score > score[i]).sum()) + 1
        return {
            "mode": "hybrid",
            "alpha": alpha,
            "rank": rank,
            "of": int((score > -1).sum()),
            "score": round(float(score[i]), 4),
            "als": {"raw": round(float(als[i]), 4), "weighted": round(float(alpha * als[i]), 4)},
            "tfidf": {"raw": round(float(tf[i]), 4), "weighted": round(float((1 - alpha) * tf[i]), 4)},
            "because": self.nearest_rated(i, rated),
            "shared_terms": self.shared_terms(prof, i, 8),
        }

    def similar(self, i: int, k: int = 8) -> dict:
        als = self.Y_unit @ self.Y_unit[i]
        als[i] = -np.inf
        top_a = np.argpartition(-als, k)[:k]
        top_a = top_a[np.argsort(-als[top_a])]
        content = np.asarray((self.X @ self.X[i].T).todense()).ravel()
        content[i] = -np.inf
        top_c = np.argpartition(-content, k)[:k]
        top_c = top_c[np.argsort(-content[top_c])]
        return {
            "also_bought": [{**self.card(j), "similarity": round(float(als[j]), 4)} for j in top_a],
            "similar_content": [{**self.card(j), "similarity": round(float(content[j]), 4)} for j in top_c],
        }

    # ---------- discovery ----------

    def search(self, q: str, limit: int = 12) -> list[dict]:
        words = [w for w in re.split(r"\s+", q.lower().strip()) if w]
        if not words:
            return []
        mask = np.ones(self.n, dtype=bool)
        for w in words:
            mask &= np.char.find(self.titles_lower.astype(str), w) >= 0
        hits = np.flatnonzero(mask)
        hits = hits[np.argsort(-self.pop_counts[hits])][:limit]
        return [self.card(i) for i in hits]

    def onboarding(self, k: int = 24) -> list[dict]:
        """Well-known games to rate first: the most-liked game per platform, round-robin."""
        games = self.items[self.items.kind == "game"].copy()
        games["pop"] = self.pop_counts[games.index]
        games = games.sort_values("pop", ascending=False)
        buckets = [g.index.tolist() for _, g in games.groupby("platform", sort=False)]
        picks, depth = [], 0
        while len(picks) < k and any(depth < len(b) for b in buckets):
            for b in buckets:
                if depth < len(b) and len(picks) < k:
                    picks.append(b[depth])
            depth += 1
        return [self.card(i) for i in picks]

    def stats(self) -> dict:
        m = self.metrics
        return {
            "dataset": m["dataset"],
            "alpha": m["alpha"],
            "k": m["k"],
            "test": m["test"],
            "lift_vs_popularity": round(m["test"]["hybrid"][f"hit_rate@{m['k']}"] / m["test"]["popularity"][f"hit_rate@{m['k']}"], 2),
            "coverage_gain_vs_als": round(m["test"]["hybrid"][f"coverage@{m['k']}"] / m["test"]["als"][f"coverage@{m['k']}"] - 1, 3),
        }


def present(value):
    """pandas stores missing strings as NaN, which JSON cannot encode."""
    return None if value is None or (isinstance(value, float) and value != value) else value


def short(title: str, n: int = 42) -> str:
    title = re.split(r"\s+[-–(\[]\s*", title)[0]
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"
