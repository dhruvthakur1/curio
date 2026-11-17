# Curio — hybrid product recommender

[![CI](https://github.com/dhruvthakur1/curio/actions/workflows/ci.yml/badge.svg)](https://github.com/dhruvthakur1/curio/actions/workflows/ci.yml)

Curio recommends video games and accessories by blending two signals:

- **ALS collaborative filtering** — "shoppers like you also bought" (teal)
- **TF-IDF content similarity** — "what the product is about" (amber)

`score = α · ALS + (1 − α) · TF-IDF`, with a popularity fallback for brand-new users.

## Data

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UC San Diego), Video Games category:

| File | Size | Used for |
|---|---|---|
| `benchmark/5core/rating_only/Video_Games.csv.gz` | 12.8 MB | ALS training and evaluation |
| `raw/meta_categories/meta_Video_Games.jsonl.gz` | 103 MB | TF-IDF text, titles, images |

After cleaning: **94,762 users · 25,611 products · 814,533 ratings** (density 0.034%).
Split is leave-last-out by time: each user's last rating is the test item, the one before it is used to tune α.

```bash
pip install -r requirements-dev.txt
python scripts/prepare_data.py   # reads data/raw, writes data/processed
python scripts/train.py          # tunes, evaluates, writes server/artifacts (~7 min)
python scripts/build_personas.py # demo shoppers' real histories
python -m pytest tests
uvicorn curio.api:app --app-dir server --port 8000   # API docs at /docs
cd client && npm install && npm run dev               # http://localhost:5174 (proxies /api)
```

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/stats` | Dataset size and headline test metrics |
| GET | `/api/metrics` | Full evaluation: ALS grid, α curve, per-segment results |
| GET | `/api/personas` | Demo shoppers (real anonymised users) with their rating history |
| POST | `/api/recommend` | `{persona?, ratings?, alpha?, k?, kinds?}` → ranked products with the ALS / TF-IDF split and a reason |
| POST | `/api/explain` | Why a product ranks where it does for a user: rank, both signal scores, shared TF-IDF terms, closest rated product |
| GET | `/api/items/{id}` | Product detail with its most distinctive TF-IDF terms |
| GET | `/api/items/{id}/similar` | "Shoppers also bought" (ALS) and "Similar products" (TF-IDF) rails |
| GET | `/api/search?q=` | Title search for the cold-start picker |
| GET | `/api/onboarding` | Well-known games across platforms for new shoppers to rate |

New ratings are folded into the trained ALS model at request time (one least-squares solve, ~1 ms), so a brand-new shopper gets personalised results after a single rating. The API needs only numpy, scipy and pandas; training libraries are dev-only.

## Results (test set, 25,000 held-out users, top 10 out of all unrated products)

| Model | Hit Rate@10 | NDCG@10 | Precision@10 | Coverage@10 |
|---|---|---|---|---|
| **Hybrid (α = 0.8)** | **8.04%** | **0.0441** | **0.80%** | **18.9%** |
| ALS only | 7.89% | 0.0429 | 0.79% | 13.7% |
| TF-IDF only | 2.33% | 0.0116 | 0.23% | 76.3% |
| Popularity | 2.70% | 0.0134 | 0.27% | 0.07% |

- The hybrid finds the user's real next purchase in its top 10 **~3× as often as popularity**, and matches or beats ALS on accuracy while recommending **39% more of the catalogue** (coverage 13.7% → 18.9%).
- α was tuned on validation users only (α sweep 0 → 1 in steps of 0.05); the test set was used once.
- ALS: 128 factors, regularisation 0.5, confidence α 40, 20 iterations (picked from an 8-config grid on validation).
- Recall@10 equals Hit Rate@10 here because each user has one held-out item.
- Gains are largest for users with short histories (3–4 ratings: hit rate 9.54% hybrid vs 9.32% ALS).

## Project structure

```
scripts/         data cleaning, training + evaluation, persona histories
server/curio/    recommender (training), engine (serving), FastAPI app
server/artifacts trained model files the API loads (committed, ~30 MB)
client/          React + Vite web app (landing, recommendations, product, model lab)
tests/           pytest: split, models, metrics, API (synthetic model, no data needed)
```

## Roadmap

1. ✅ Data download, cleaning and split
2. ✅ ALS + TF-IDF training, fusion, evaluation
3. ✅ FastAPI backend
4. ✅ React landing and recommendations
5. ✅ Product page and model lab
6. ✅ Tests, Docker and deploy config — see [DEPLOY.md](DEPLOY.md)
