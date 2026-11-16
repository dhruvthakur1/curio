"""Curio's hybrid recommender: ALS collaborative filtering + TF-IDF content similarity.

    score(u, i) = alpha * ALS(u, i) + (1 - alpha) * TFIDF(u, i)

Both signals are min-max scaled per user before they are blended, so alpha
really is "how much to trust each signal" rather than a unit conversion.
Users with no history get a popularity ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ---------- interaction weights ----------

def rating_weight(ratings: np.ndarray) -> np.ndarray:
    """Confidence that a rating means 'liked'. 5 stars -> 1.0, 1 star -> 0.2."""
    return np.asarray(ratings, dtype=np.float32) / 5.0


def profile_weight(ratings: np.ndarray) -> np.ndarray:
    """How much a rated item shapes the content profile. Disliked items barely count."""
    return np.clip(np.asarray(ratings, dtype=np.float32) - 2.5, 0.1, None)


def interaction_matrix(users: np.ndarray, items: np.ndarray, values: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix((np.asarray(values, dtype=np.float32), (users, items)), shape=(n_users, n_items))
    m.sum_duplicates()
    return m


# ---------- the two signals ----------

class ALSModel:
    def __init__(self, factors: int = 64, regularization: float = 0.05, iterations: int = 20,
                 alpha: float = 20.0, seed: int = 42):
        self.params = dict(factors=factors, regularization=regularization, iterations=iterations, alpha=alpha)
        self.model = AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations,
            alpha=alpha, random_state=seed, use_gpu=False, calculate_training_loss=False,
        )

    def fit(self, user_items: sp.csr_matrix) -> "ALSModel":
        self.model.fit(user_items, show_progress=False)
        return self

    @property
    def item_factors(self) -> np.ndarray:
        return np.asarray(self.model.item_factors)

    def user_vectors(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.user_factors)[rows]

    def fold_in(self, user_row: sp.csr_matrix) -> np.ndarray:
        """Vector for a user the model never saw (one ALS half-step)."""
        return np.asarray(self.model.recalculate_user(0, user_row)).ravel()

    def score(self, user_vecs: np.ndarray) -> np.ndarray:
        return user_vecs @ self.item_factors.T


class TfidfModel:
    def __init__(self, max_features: int = 30000, min_df: int = 3, max_df: float = 0.3):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=min_df, max_df=max_df, ngram_range=(1, 2),
            stop_words="english", sublinear_tf=True, dtype=np.float32,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
        )
        self.X: sp.csr_matrix | None = None  # items x terms, rows L2-normalised

    def fit(self, texts) -> "TfidfModel":
        self.X = self.vectorizer.fit_transform(texts).tocsr()
        return self

    @property
    def terms(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()

    def profiles(self, weights: sp.csr_matrix) -> sp.csr_matrix:
        """users x items weights -> users x terms profiles (L2-normalised)."""
        return normalize(weights @ self.X)

    def score(self, profiles: sp.csr_matrix) -> np.ndarray:
        return (profiles @ self.X.T).toarray().astype(np.float32)


# ---------- fusion ----------

def minmax_rows(scores: np.ndarray, mask: sp.csr_matrix | None = None) -> np.ndarray:
    """Scale each row to [0, 1], ignoring masked (already seen) items, which sink to -1."""
    s = np.array(scores, dtype=np.float32, copy=True)
    if mask is not None:
        s[mask.nonzero()] = np.nan
    lo = np.nanmin(s, axis=1, keepdims=True)
    hi = np.nanmax(s, axis=1, keepdims=True)
    out = (s - lo) / np.where(hi - lo > 1e-9, hi - lo, 1.0)
    return np.nan_to_num(out, nan=-1.0)


def fuse(als_norm: np.ndarray, tfidf_norm: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * als_norm + (1.0 - alpha) * tfidf_norm


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, idx, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, axis=1)


@dataclass
class Popularity:
    counts: np.ndarray  # liked (4+ star) interactions per item

    @classmethod
    def fit(cls, ratings: sp.csr_matrix) -> "Popularity":
        liked = ratings.multiply(ratings >= 4.0) > 0
        return cls(np.asarray(liked.sum(axis=0)).ravel().astype(np.float32))

    def score(self, n_users: int) -> np.ndarray:
        s = np.log1p(self.counts)
        return np.tile(s / max(s.max(), 1e-9), (n_users, 1)).astype(np.float32)
