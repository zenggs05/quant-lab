"""Feature engineering for Optiver Trading at the Close.

Contract (enforced by tests/test_leakage.py): features for row (stock, date d, second s)
may depend only on
  * rows of date d with seconds_in_bucket <= s (any stock), and
  * revealed targets of date d-1 with seconds_in_bucket <= s.
`build_features` must give identical results when run on a full day in batch and when
run in streaming mode on the history available at each time step.
"""
from __future__ import annotations

import polars as pl

RAW_COLS = [
    "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag", "reference_price",
    "matched_size", "far_price", "near_price", "bid_price", "bid_size", "ask_price",
    "ask_size", "wap",
]

LAG_COLS = ["wap", "reference_price", "imbalance_size", "matched_size", "bid_size", "ask_size"]
LAGS = (1, 2, 3, 6)


def _ratio(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a / (b + 1e-9)


def build_features(day_rows: pl.DataFrame, revealed: pl.DataFrame | None = None,
                   index_weights: dict[int, float] | None = None) -> pl.DataFrame:
    """Compute features for all rows in `day_rows`.

    day_rows: auction rows (no target) for one or more dates, possibly only up to some second.
    revealed: columns [stock_id, seconds_in_bucket, revealed_target, revealed_date_id] with the
              previous-day targets visible so far (or None).
    """
    g = ["stock_id", "date_id"]
    df = day_rows.sort(["stock_id", "date_id", "seconds_in_bucket"])

    df = df.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        _ratio(pl.col("bid_size") - pl.col("ask_size"), pl.col("bid_size") + pl.col("ask_size")).alias("book_imb"),
        _ratio(pl.col("imbalance_size"), pl.col("matched_size")).alias("imb_ratio"),
        (pl.col("imbalance_size") * pl.col("imbalance_buy_sell_flag")).alias("signed_imb"),
        _ratio(pl.col("imbalance_size") * pl.col("imbalance_buy_sell_flag"), pl.col("matched_size")).alias("signed_imb_ratio"),
        (pl.col("reference_price") - pl.col("wap")).alias("ref_minus_wap"),
        (pl.col("far_price") - pl.col("near_price")).alias("far_minus_near"),
        (pl.col("near_price") - pl.col("reference_price")).alias("near_minus_ref"),
        (pl.col("bid_price") + pl.col("ask_price")).truediv(2).alias("mid"),
        _ratio(pl.col("imbalance_size"), pl.col("bid_size") + pl.col("ask_size")).alias("imb_over_depth"),
    )
    df = df.with_columns((pl.col("wap") - pl.col("mid")).alias("wap_minus_mid"))

    # Within-day lags / changes (strictly backward-looking).
    lag_exprs = []
    for c in LAG_COLS:
        for k in LAGS:
            lag_exprs.append((pl.col(c) / pl.col(c).shift(k).over(g) - 1).alias(f"{c}_ret_{k}"))
    lag_exprs += [
        (pl.col("signed_imb_ratio") - pl.col("signed_imb_ratio").shift(1).over(g)).alias("signed_imb_ratio_d1"),
        (pl.col("book_imb") - pl.col("book_imb").shift(1).over(g)).alias("book_imb_d1"),
        (pl.col("wap") / pl.col("wap").first().over(g) - 1).alias("wap_ret_day"),
        pl.col("imbalance_buy_sell_flag").diff().over(g).alias("flag_change"),
    ]
    df = df.with_columns(lag_exprs)

    # Cross-sectional features at the same time_id (same-time information only).
    cs_cols = ["wap_ret_1", "wap_ret_6", "wap_ret_day", "signed_imb_ratio", "book_imb", "imb_ratio"]
    df = df.with_columns(
        [(pl.col(c) - pl.col(c).mean().over("time_id")).alias(f"{c}_cs") for c in cs_cols]
        + [pl.col("wap_ret_1").mean().over("time_id").alias("mkt_wap_ret_1"),
           pl.col("wap_ret_6").mean().over("time_id").alias("mkt_wap_ret_6")]
    )

    # v2: index-relative features using the recovered synthetic-index weights.
    if index_weights is not None:
        df = df.with_columns(pl.col("stock_id").replace_strict(index_weights, default=0.0).alias("_w"))
        idx_cols = ["wap_ret_1", "wap_ret_3", "wap_ret_6", "wap_ret_day", "signed_imb_ratio", "book_imb"]
        df = df.with_columns(
            [((pl.col(c).fill_null(0) * pl.col("_w")).sum().over("time_id")
              / (pl.col("_w") * pl.col(c).is_not_null()).sum().over("time_id")).alias(f"idx_{c}")
             for c in idx_cols]
        )
        df = df.with_columns([(pl.col(c) - pl.col(f"idx_{c}")).alias(f"{c}_rel") for c in idx_cols])
        df = df.with_columns(pl.col("_w").alias("index_weight")).drop("_w")

    # Previous-day revealed targets (same second and day-level summary of revealed so far).
    if revealed is not None and revealed.height:
        rv = revealed.select("stock_id", "seconds_in_bucket", "revealed_target", "revealed_date_id")
        df = df.with_columns((pl.col("date_id") - 1).alias("_prev_date"))
        same_s = rv.rename({"revealed_target": "prev_day_target_same_s"})
        df = df.join(same_s, left_on=["stock_id", "_prev_date", "seconds_in_bucket"],
                     right_on=["stock_id", "revealed_date_id", "seconds_in_bucket"], how="left")
        # mean of previous-day targets with sib <= current s
        rv_sorted = rv.sort(["stock_id", "revealed_date_id", "seconds_in_bucket"]).with_columns(
            pl.col("revealed_target").cum_sum().over(["stock_id", "revealed_date_id"]).alias("_cs"),
            pl.col("revealed_target").cum_count().over(["stock_id", "revealed_date_id"]).alias("_cn"),
        ).select("stock_id", "revealed_date_id", "seconds_in_bucket",
                 (pl.col("_cs") / pl.col("_cn")).alias("prev_day_target_cummean"))
        df = df.join(rv_sorted, left_on=["stock_id", "_prev_date", "seconds_in_bucket"],
                     right_on=["stock_id", "revealed_date_id", "seconds_in_bucket"], how="left")
        df = df.drop("_prev_date")
    else:
        df = df.with_columns(pl.lit(None, pl.Float64).alias("prev_day_target_same_s"),
                             pl.lit(None, pl.Float64).alias("prev_day_target_cummean"))
    return df


def feature_columns(df: pl.DataFrame) -> list[str]:
    drop = {"date_id", "time_id", "row_id", "target", "currently_scored", "mid"}
    return [c for c in df.columns if c not in drop]


def revealed_from_train(train: pl.DataFrame) -> pl.DataFrame:
    """Full previous-day target table, as it would be revealed (filtering by s happens in the join:
    a row at second s only joins previous-day rows at seconds <= s)."""
    return train.select(
        "stock_id", "seconds_in_bucket", pl.col("target").alias("revealed_target"),
        pl.col("date_id").alias("revealed_date_id"),
    )
