import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from curio.metrics import ranking_metrics, summarise  # noqa: E402
from curio.recommender import (  # noqa: E402
    ALSModel, Popularity, TfidfModel, fuse, interaction_matrix, minmax_rows, top_k,
)


def test_minmax_rows_scales_and_sinks_seen_items():
    s = np.array([[1.0, 3.0, 5.0, 9.0]])
    seen = sp.csr_matrix(np.array([[0, 0, 0, 1]]))
    out = minmax_rows(s, seen)
    assert np.allclose(out[0, :3], [0.0, 0.5, 1.0])
    assert out[0, 3] == -1.0


def test_fuse_endpoints():
    a, t = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert np.allclose(fuse(a, t, 1.0), a)
    assert np.allclose(fuse(a, t, 0.0), t)
    assert np.allclose(fuse(a, t, 0.7), [[0.7, 0.3]])


def test_top_k_is_sorted_descending():
    s = np.array([[0.1, 0.9, 0.5, 0.7, 0.3]])
    assert top_k(s, 3).tolist() == [[1, 3, 2]]


def test_ranking_metrics_leave_one_out():
    recs = np.array([[5, 2, 9], [1, 3, 4]])
    m = ranking_metrics(recs, np.array([2, 7]), k=3)
    assert m["hit"].tolist() == [1.0, 0.0]
    assert np.isclose(m["ndcg"][0], 1 / np.log2(3))
    s = summarise(m, recs, 3, n_items=10)
    assert s["hit_rate@3"] == s["recall@3"] == 0.5
    assert np.isclose(s["precision@3"], 0.5 / 3, atol=1e-5)
    assert s["coverage@3"] == 0.6


def test_tfidf_prefers_items_with_shared_words():
    texts = ["dark souls hard action rpg", "hard action rpg with bosses", "cute farming life sim", "farming sim cozy"]
    tf = TfidfModel(min_df=1, max_df=1.0).fit(texts)
    profile = tf.profiles(sp.csr_matrix(np.array([[1.0, 0, 0, 0]], dtype=np.float32)))
    scores = tf.score(profile)[0]
    assert scores[1] > scores[2] and scores[1] > scores[3]


def test_als_learns_co_occurrence_and_folds_in_new_users():
    rng = np.random.default_rng(0)
    # Two taste groups: users 0-49 like items 0-4, users 50-99 like items 5-9.
    users, items = [], []
    for u in range(100):
        group = range(0, 5) if u < 50 else range(5, 10)
        for i in rng.choice(list(group), size=3, replace=False):
            users.append(u)
            items.append(i)
    m = interaction_matrix(np.array(users), np.array(items), np.ones(len(users)), 100, 10)
    als = ALSModel(factors=8, iterations=15, alpha=10.0, regularization=0.01).fit(m)
    new_user = sp.csr_matrix(([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, 10))
    scores = als.score(als.fold_in(new_user)[None, :])[0]
    assert scores[2:5].mean() > scores[5:10].mean()


def test_popularity_counts_only_liked_ratings():
    r = sp.csr_matrix(np.array([[5, 2, 0], [4, 0, 1], [0, 5, 0]], dtype=np.float32))
    pop = Popularity.fit(r)
    assert pop.counts.tolist() == [2, 1, 0]
    assert pop.score(2).shape == (2, 3)
