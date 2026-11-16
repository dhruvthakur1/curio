"""Phase 1: clean the Amazon Reviews 2023 Video Games data and split it.

Inputs (downloaded once, git-ignored):
  data/raw/Video_Games.csv.gz          5-core ratings: user_id, parent_asin, rating, timestamp
  data/raw/meta_Video_Games.jsonl.gz   product metadata

Outputs (data/processed/):
  items.parquet   one row per product that appears in the ratings
  train.parquet   all interactions except each user's last two
  valid.parquet   each user's second-to-last interaction (used to tune alpha)
  test.parquet    each user's last interaction (used only for the final report)
  summary.json    counts shown on the landing page

Split: leave-last-out by time, the standard protocol for this benchmark.
Users keep at least 3 training interactions, otherwise they go to train only.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

MIN_TRAIN = 3
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def parse_listish(value) -> list[str]:
    """Metadata fields are lists, or strings that look like Python lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        value = value.strip()
        if not value or value == "None":
            return []
        if value.startswith("["):
            try:
                parsed = ast.literal_eval(value)
                return [str(v) for v in parsed if v]
            except (ValueError, SyntaxError):
                pass
        return [value]
    return [str(value)]


def clean_text(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def first_image(images) -> str | None:
    if isinstance(images, str):
        try:
            images = ast.literal_eval(images)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(images, list):
        return None
    for img in images:
        if isinstance(img, dict):
            url = img.get("large") or img.get("hi_res") or img.get("thumb")
            if url:
                return url
    return None


def to_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"user_id": "string", "parent_asin": "string", "rating": "float32", "timestamp": "int64"},
    )
    df = df.rename(columns={"parent_asin": "item_id"})
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    # A user can rate the same product more than once; keep the latest rating.
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    return df.reset_index(drop=True)


def load_items(path: Path, wanted: set[str]) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            asin = d.get("parent_asin")
            if asin not in wanted:
                continue
            title = clean_text(str(d.get("title") or ""))
            if not title:
                continue
            categories = [c for c in parse_listish(d.get("categories")) if c != "Video Games"]
            features = [clean_text(x) for x in parse_listish(d.get("features"))]
            description = clean_text(" ".join(parse_listish(d.get("description"))))
            rows.append(
                {
                    "item_id": asin,
                    "title": title,
                    "store": clean_text(str(d.get("store") or "")) or None,
                    "categories": categories,
                    "features": features,
                    "description": description[:2000],
                    "price": to_float(d.get("price")),
                    "average_rating": to_float(d.get("average_rating")),
                    "rating_number": int(to_float(d.get("rating_number")) or 0),
                    "image": first_image(d.get("images")),
                }
            )
    items = pd.DataFrame(rows).drop_duplicates("item_id")
    # Text that TF-IDF reads. Title and categories are repeated to weight them up.
    items["text"] = (
        items["title"] + " " + items["title"] + " "
        + items["categories"].map(" ".join) + " " + items["categories"].map(" ".join) + " "
        + items["store"].fillna("") + " "
        + items["features"].map(" ".join) + " "
        + items["description"]
    ).map(clean_text)
    return items.reset_index(drop=True)


def split_leave_last_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    pos_from_end = df.groupby("user_id").cumcount(ascending=False)
    n = df.groupby("user_id")["item_id"].transform("size")
    eligible = n >= MIN_TRAIN + 2
    test = df[eligible & (pos_from_end == 0)]
    valid = df[eligible & (pos_from_end == 1)]
    train = df[~eligible | (pos_from_end >= 2)]
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    ratings = load_ratings(args.raw / "Video_Games.csv.gz")
    print(f"ratings: {len(ratings):,} rows, {ratings.user_id.nunique():,} users, {ratings.item_id.nunique():,} items")

    items = load_items(args.raw / "meta_Video_Games.jsonl.gz", set(ratings.item_id.unique()))
    print(f"items with metadata: {len(items):,}")

    # Keep only interactions whose product we can show and describe.
    ratings = ratings[ratings.item_id.isin(set(items.item_id))]
    items = items[items.item_id.isin(set(ratings.item_id))].reset_index(drop=True)

    train, valid, test = split_leave_last_out(ratings)
    # Evaluation items must exist in training, otherwise no model can rank them.
    known = set(train.item_id)
    valid = valid[valid.item_id.isin(known)]
    test = test[test.item_id.isin(known)]

    args.out.mkdir(parents=True, exist_ok=True)
    items.to_parquet(args.out / "items.parquet", index=False)
    train.to_parquet(args.out / "train.parquet", index=False)
    valid.to_parquet(args.out / "valid.parquet", index=False)
    test.to_parquet(args.out / "test.parquet", index=False)

    summary = {
        "dataset": "Amazon Reviews 2023 — Video Games (5-core)",
        "source": "McAuley Lab, UC San Diego",
        "users": int(ratings.user_id.nunique()),
        "items": int(len(items)),
        "interactions": int(len(ratings)),
        "train": int(len(train)),
        "valid": int(len(valid)),
        "test": int(len(test)),
        "density": round(len(ratings) / (ratings.user_id.nunique() * len(items)), 6),
        "mean_rating": round(float(ratings.rating.mean()), 3),
        "positive_share": round(float((ratings.rating >= 4).mean()), 3),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
