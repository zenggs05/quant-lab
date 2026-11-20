"""Leakage tests: batch features must equal streaming features computed through the mock API."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv import EMBARGO_DAYS, holdout_fold, selection_folds  # noqa: E402
from features import build_features, revealed_from_train  # noqa: E402
from mock_api import iter_time_steps  # noqa: E402

DATA = ROOT / "data" / "processed" / "train.parquet"
DATES = [401, 402]
STOCKS = list(range(0, 200, 7))  # subset for speed; cross-sectional features use the same subset in both modes


import json  # noqa: E402

W = {int(k): v for k, v in json.loads((ROOT / "reports/eda_day2.json").read_text())["index_weights"]["weights_by_stock"].items()}


@pytest.fixture(scope="module")
def sample() -> pl.DataFrame:
    return pl.read_parquet(DATA).filter(
        pl.col("date_id").is_in([min(DATES) - 1] + DATES) & pl.col("stock_id").is_in(STOCKS)
    )


def test_folds_are_ordered_and_embargoed() -> None:
    folds = selection_folds() + [holdout_fold()]
    for f in folds:
        assert f.train_end + EMBARGO_DAYS < f.valid_start
    for f in selection_folds():
        assert f.valid_end < holdout_fold().valid_start


def test_batch_equals_streaming(sample: pl.DataFrame) -> None:
    day = sample.filter(pl.col("date_id").is_in(DATES))
    batch = build_features(day.drop("target"), revealed_from_train(sample), W).sort("row_id")

    streamed = []
    hist: list[pl.DataFrame] = []
    for test, revealed in iter_time_steps(sample, DATES):
        hist.append(test)
        d = test["date_id"][0]
        past = pl.concat(hist).filter(pl.col("date_id") == d)
        feats = build_features(past, revealed, W)
        streamed.append(feats.filter(pl.col("time_id") == test["time_id"][0]))
    stream = pl.concat(streamed).sort("row_id")

    assert batch.columns == stream.columns
    assert batch.height == stream.height == day.height
    for c in batch.columns:
        a, b = batch[c], stream[c]
        if a.dtype.is_float():
            diff = (a.fill_null(-9e9).fill_nan(-8e9) - b.fill_null(-9e9).fill_nan(-8e9)).abs().max()
            assert diff is None or diff < 1e-9, f"feature {c} differs between batch and streaming: {diff}"
        else:
            assert a.equals(b), f"feature {c} differs"


def test_no_target_in_features(sample: pl.DataFrame) -> None:
    feats = build_features(sample.drop("target"), revealed_from_train(sample))
    assert "target" not in feats.columns
