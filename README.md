# Curio — hybrid product recommender

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
python scripts/prepare_data.py   # reads data/raw, writes data/processed
python -m pytest tests
```

## Roadmap

1. ✅ Data download, cleaning and split
2. ALS + TF-IDF training, fusion, evaluation
3. FastAPI backend
4. React landing and recommendations
5. Product page and model lab
6. Tests, Docker and deploy
