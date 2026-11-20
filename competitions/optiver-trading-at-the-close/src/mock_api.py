"""Local replay of the optiver2023 time-series API (written from the documented behaviour;
the official API binary is never imported locally).

At each time_id (date d, seconds s) the caller receives:
- `test`: all stocks' order-book/auction rows for (d, s), without `target`;
- `revealed`: targets of date d-1 for seconds_in_bucket <= s (conservative reading of the
  example revealed_targets.csv, which serves day d-1 / second s at day d / second s).
"""
from __future__ import annotations

from collections.abc import Iterator

import polars as pl

FEATURE_INPUT_COLS = [
    "stock_id", "date_id", "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
    "reference_price", "matched_size", "far_price", "near_price", "bid_price", "bid_size",
    "ask_price", "ask_size", "wap", "time_id", "row_id",
]


def iter_time_steps(df: pl.DataFrame, dates: list[int]) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    df = df.sort(["time_id", "stock_id"])
    for d in dates:
        day = df.filter(pl.col("date_id") == d)
        prev = df.filter(pl.col("date_id") == d - 1).select(
            "stock_id", "seconds_in_bucket", pl.col("target").alias("revealed_target"),
            pl.col("date_id").alias("revealed_date_id"),
        )
        for s in sorted(day["seconds_in_bucket"].unique().to_list()):
            test = day.filter(pl.col("seconds_in_bucket") == s).select(FEATURE_INPUT_COLS)
            revealed = prev.filter(pl.col("seconds_in_bucket") <= s)
            yield test, revealed
