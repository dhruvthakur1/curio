"""Phase 2: train ALS + TF-IDF, tune on validation, evaluate on test, save model files.

    1. Fit TF-IDF on product text.
    2. Grid-search ALS on train, scored on validation users (ALS alone).
    3. Sweep alpha for the hybrid on validation users.
    4. Refit ALS on train + validation, evaluate every model on held-out test users.
    5. Save the files the API needs to server/artifacts/.

Run:  python scripts/train.py            (full, ~10 min)
      python scripts/train.py --quick    (small samples, for a smoke test)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from curio.metrics import ranking_metrics, summarise  # noqa: E402
from curio.recommender import (  # noqa: E402
    ALSModel, Popularity, TfidfModel, fuse, interaction_matrix, minmax_rows,
    profile_weight, rating_weight, top_k,
)

DATA = ROOT / "data" / "processed"
ART = ROOT / "server" / "artifacts"
K = 10
SEGMENTS = [("3–4 ratings", 3, 4), ("5–9 ratings", 5, 9), ("10+ ratings", 10, 10**9)]


class Data:
    def __init__(self):
        self.items = pd.read_parquet(DATA / "items.parquet")
        self.train = pd.read_parquet(DATA / "train.parquet")
        self.valid = pd.read_parquet(DATA / "valid.parquet")
        self.test = pd.read_parquet(DATA / "test.parquet")
        self.item_ix = pd.Index(self.items.item_id)
        users = pd.concat([self.train.user_id, self.valid.user_id, self.test.user_id]).unique()
        self.user_ix = pd.Index(users)
        self.n_users, self.n_items = len(self.user_ix), len(self.item_ix)

    def matrices(self, history: pd.DataFrame) -> dict[str, sp.csr_matrix]:
        u = self.user_ix.get_indexer(history.user_id)
        i = self.item_ix.get_indexer(history.item_id)
        r = history.rating.to_numpy(np.float32)
        mk = lambda v: interaction_matrix(u, i, v, self.n_users, self.n_items)
        return {"als": mk(rating_weight(r)), "profile": mk(profile_weight(r)), "rating": mk(r)}

    def targets(self, df: pd.DataFrame, n: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if n and n < len(df):
            df = df.sample(n=n, random_state=seed)
        return self.user_ix.get_indexer(df.user_id), self.item_ix.get_indexer(df.item_id)


def signal_scores(rows, mats, als: ALSModel | None, tfidf: TfidfModel | None):
    seen = mats["rating"][rows]
    a = minmax_rows(als.score(als.user_vectors(rows)), seen) if als else None
    t = minmax_rows(tfidf.score(tfidf.profiles(mats["profile"][rows])), seen) if tfidf else None
    return a, t, seen


def evaluate(rows, targets, mats, n_items, als=None, tfidf=None, alphas=(1.0,), pop=None, batch=1000):
    """Returns {name: {"recs": users×K, "per_user": {...}}} for each alpha (and popularity)."""
    recs = {a: [] for a in alphas}
    if pop is not None:
        recs["popularity"] = []
    for start in range(0, len(rows), batch):
        r = rows[start:start + batch]
        a_s, t_s, seen = signal_scores(r, mats, als, tfidf)
        for alpha in alphas:
            if a_s is None:
                s = t_s
            elif t_s is None:
                s = a_s
            else:
                s = fuse(a_s, t_s, alpha)
            recs[alpha].append(top_k(s, K))
        if pop is not None:
            recs["popularity"].append(top_k(minmax_rows(pop.score(len(r)), seen), K))
    out = {}
    for name, parts in recs.items():
        rr = np.vstack(parts)
        out[name] = {"recs": rr, "per_user": ranking_metrics(rr, targets, K)}
    return out


def report(res, n_items):
    return summarise(res["per_user"], res["recs"], K, n_items)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    n_grid, n_alpha, n_test = (1500, 2000, 3000) if args.quick else (6000, 10000, 25000)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    d = Data()
    log(f"{d.n_users:,} users · {d.n_items:,} items")
    train_m = d.matrices(d.train)

    tfidf = TfidfModel().fit(d.items.text)
    log(f"TF-IDF: {tfidf.X.shape[1]:,} terms, {tfidf.X.nnz / d.n_items:.0f} per item")

    # ---- 2. ALS grid on validation ----
    v_rows, v_tgt = d.targets(d.valid, n_grid, seed=1)
    grid = [dict(factors=f, regularization=r, alpha=c) for f, r, c in
            itertools.product([64, 128], [0.05, 0.5], [10.0, 40.0])]
    if args.quick:
        grid = grid[:2]
    grid_results = []
    for g in grid:
        als = ALSModel(iterations=15, **g).fit(train_m["als"])
        m = report(evaluate(v_rows, v_tgt, train_m, d.n_items, als=als)[1.0], d.n_items)
        grid_results.append({**g, **m})
        log(f"ALS {g} → ndcg@10 {m['ndcg@10']:.4f} hit@10 {m['hit_rate@10']:.4f}")
    best_g = max(grid_results, key=lambda x: x["ndcg@10"])
    best_params = {k: best_g[k] for k in ("factors", "regularization", "alpha")}
    log(f"best ALS: {best_params}")

    # ---- 3. alpha sweep on validation ----
    als = ALSModel(iterations=20, **best_params).fit(train_m["als"])
    a_rows, a_tgt = d.targets(d.valid, n_alpha, seed=2)
    alphas = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
    sweep = evaluate(a_rows, a_tgt, train_m, d.n_items, als=als, tfidf=tfidf, alphas=alphas)
    curve = [{"alpha": a, **report(sweep[a], d.n_items)} for a in alphas]
    best_alpha = max(curve, key=lambda x: x["ndcg@10"])["alpha"]
    for c in curve:
        log(f"α={c['alpha']:.2f} ndcg@10 {c['ndcg@10']:.4f} hit@10 {c['hit_rate@10']:.4f} cov {c['coverage@10']:.3f}")
    log(f"best α = {best_alpha}")

    # ---- 4. refit on train+valid, evaluate on test ----
    full = pd.concat([d.train, d.valid], ignore_index=True)
    full_m = d.matrices(full)
    als = ALSModel(iterations=20, **best_params).fit(full_m["als"])
    pop = Popularity.fit(full_m["rating"])
    t_rows, t_tgt = d.targets(d.test, n_test, seed=3)

    both = evaluate(t_rows, t_tgt, full_m, d.n_items, als=als, tfidf=tfidf,
                    alphas=(1.0, 0.0, best_alpha), pop=pop)
    models = {"hybrid": both[best_alpha], "als": both[1.0], "tfidf": both[0.0], "popularity": both["popularity"]}
    test_metrics = {name: report(res, d.n_items) for name, res in models.items()}
    for name, m in test_metrics.items():
        log(f"TEST {name:10s} {m}")

    hist_len = np.asarray((full_m["rating"][t_rows] > 0).sum(axis=1)).ravel()
    segments = []
    for label, lo, hi in SEGMENTS:
        mask = (hist_len >= lo) & (hist_len <= hi)
        seg = {"segment": label, "users": int(mask.sum())}
        for name, res in models.items():
            seg[name] = {
                "hit_rate@10": round(float(res["per_user"]["hit"][mask].mean()), 5),
                "ndcg@10": round(float(res["per_user"]["ndcg"][mask].mean()), 5),
            }
        segments.append(seg)
        log(f"segment {label}: {seg}")

    # ---- 5. save ----
    if args.quick:
        log("quick run: not saving artifacts")
        return test_metrics
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "als_item_factors.npy", als.item_factors.astype(np.float32))
    sp.save_npz(ART / "tfidf.npz", tfidf.X.astype(np.float32))
    (ART / "tfidf_terms.json").write_text(json.dumps(tfidf.terms.tolist()))
    np.save(ART / "popularity.npy", pop.counts)
    keep = ["item_id", "title", "store", "categories", "price", "average_rating", "rating_number", "image", "description"]
    items = d.items[keep].copy()
    items["description"] = items.description.str.slice(0, 600)
    items["train_count"] = np.asarray((full_m["rating"] > 0).sum(axis=0)).ravel()
    items.to_parquet(ART / "items.parquet", index=False)
    summary = json.loads((DATA / "summary.json").read_text())
    metrics = {
        "k": K,
        "dataset": summary,
        "protocol": {
            "split": "leave-last-out by time; validation = 2nd-to-last rating, test = last rating",
            "validation_users_grid": len(v_rows), "validation_users_alpha": len(a_rows), "test_users": len(t_rows),
            "candidates": "all products the user has not rated",
        },
        "als_params": {**best_params, "iterations": 20},
        "als_grid": grid_results,
        "alpha": best_alpha,
        "alpha_curve": curve,
        "test": test_metrics,
        "segments": segments,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in ART.iterdir()}
    log(f"saved {sizes}")
    return test_metrics


if __name__ == "__main__":
    with threadpool_limits(1, "blas"):
        main()
