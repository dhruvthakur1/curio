import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import parse_listish, split_leave_last_out, first_image  # noqa: E402


def make(user, n):
    return [{"user_id": user, "item_id": f"{user}-i{k}", "rating": 5.0, "timestamp": k} for k in range(n)]


def test_leave_last_out_uses_time_order():
    df = pd.DataFrame(make("a", 6) + make("b", 4))
    train, valid, test = split_leave_last_out(df.sample(frac=1, random_state=0))
    assert test[test.user_id == "a"].item_id.tolist() == ["a-i5"]
    assert valid[valid.user_id == "a"].item_id.tolist() == ["a-i4"]
    assert len(train[train.user_id == "a"]) == 4
    # b has too few interactions to hold any out
    assert "b" not in set(test.user_id) | set(valid.user_id)
    assert len(train[train.user_id == "b"]) == 4


def test_no_overlap_between_splits():
    df = pd.DataFrame(make("a", 8) + make("c", 5))
    train, valid, test = split_leave_last_out(df)
    keys = lambda d: set(zip(d.user_id, d.item_id))
    assert not keys(train) & keys(test)
    assert not keys(train) & keys(valid)
    assert len(train) + len(valid) + len(test) == len(df)


def test_parse_listish_handles_strings_and_none():
    assert parse_listish("['PC', 'Games']") == ["PC", "Games"]
    assert parse_listish("None") == []
    assert parse_listish(None) == []
    assert parse_listish("plain text") == ["plain text"]


def test_first_image_prefers_large():
    assert first_image("[{'thumb': 't.jpg', 'large': 'l.jpg'}]") == "l.jpg"
    assert first_image("[]") is None
