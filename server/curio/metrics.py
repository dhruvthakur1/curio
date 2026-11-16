"""Ranking metrics for leave-one-out evaluation (one held-out item per user)."""

from __future__ import annotations

import numpy as np


def ranking_metrics(recs: np.ndarray, targets: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """recs: users × ≥k ranked item indices. targets: held-out item per user.

    Returns per-user arrays so callers can average over any subset of users.
    With a single relevant item, Recall@K equals Hit Rate@K and Precision@K is hit / K.
    """
    recs = recs[:, :k]
    match = recs == targets[:, None]
    hit = match.any(axis=1).astype(np.float32)
    rank = np.where(match.any(axis=1), match.argmax(axis=1), -1)
    ndcg = np.where(rank >= 0, 1.0 / np.log2(np.maximum(rank, 0) + 2.0), 0.0).astype(np.float32)
    return {"hit": hit, "precision": hit / k, "recall": hit, "ndcg": ndcg}


def summarise(per_user: dict[str, np.ndarray], recs: np.ndarray, k: int, n_items: int) -> dict[str, float]:
    out = {f"{m}@{k}": round(float(v.mean()), 5) for m, v in per_user.items() if m != "hit"}
    out[f"hit_rate@{k}"] = out.pop(f"recall@{k}")
    out[f"recall@{k}"] = out[f"hit_rate@{k}"]
    out[f"coverage@{k}"] = round(len(np.unique(recs[:, :k])) / n_items, 5)
    return out
