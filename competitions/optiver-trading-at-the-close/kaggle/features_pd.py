"""pandas/numpy port of src/features.py for the Kaggle notebook (Python 3.10, 2023 image).

Must match src/features.py value-for-value (checked by tests/test_parity.py).
Avoids APIs newer than pandas 1.5 / 2.0.
"""
import numpy as np
import pandas as pd

LAG_COLS = ["wap", "reference_price", "imbalance_size", "matched_size", "bid_size", "ask_size"]
LAGS = (1, 2, 3, 6)
CS_COLS = ["wap_ret_1", "wap_ret_6", "wap_ret_day", "signed_imb_ratio", "book_imb", "imb_ratio"]
IDX_COLS = ["wap_ret_1", "wap_ret_3", "wap_ret_6", "wap_ret_day", "signed_imb_ratio", "book_imb"]
DROP = {"date_id", "time_id", "row_id", "target", "currently_scored", "mid"}


def _ratio(a, b):
    return a / (b + 1e-9)


def build_features_pd(day_rows, revealed=None, index_weights=None):
    """day_rows: auction rows (no target). revealed: [stock_id, seconds_in_bucket, revealed_target,
    revealed_date_id] previous-day targets visible so far. Returns a new DataFrame sorted by
    (stock_id, date_id, seconds_in_bucket)."""
    g = ["stock_id", "date_id"]
    df = day_rows.sort_values(["stock_id", "date_id", "seconds_in_bucket"], kind="mergesort").reset_index(drop=True)
    f = {}
    f["spread"] = df["ask_price"] - df["bid_price"]
    f["book_imb"] = _ratio(df["bid_size"] - df["ask_size"], df["bid_size"] + df["ask_size"])
    f["imb_ratio"] = _ratio(df["imbalance_size"], df["matched_size"])
    f["signed_imb"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    f["signed_imb_ratio"] = _ratio(df["imbalance_size"] * df["imbalance_buy_sell_flag"], df["matched_size"])
    f["ref_minus_wap"] = df["reference_price"] - df["wap"]
    f["far_minus_near"] = df["far_price"] - df["near_price"]
    f["near_minus_ref"] = df["near_price"] - df["reference_price"]
    f["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    f["imb_over_depth"] = _ratio(df["imbalance_size"], df["bid_size"] + df["ask_size"])
    f["wap_minus_mid"] = df["wap"] - f["mid"]

    grp = df.groupby(g, sort=False)
    for c in LAG_COLS:
        gc = grp[c]
        for k in LAGS:
            f[f"{c}_ret_{k}"] = df[c] / gc.shift(k) - 1
    tmp = pd.DataFrame({"stock_id": df["stock_id"], "date_id": df["date_id"],
                        "sir": f["signed_imb_ratio"], "bi": f["book_imb"]})
    tg = tmp.groupby(g, sort=False)
    f["signed_imb_ratio_d1"] = tmp["sir"] - tg["sir"].shift(1)
    f["book_imb_d1"] = tmp["bi"] - tg["bi"].shift(1)
    first_pos = grp.cumcount() == 0
    first_wap = df["wap"].where(first_pos)
    # forward-fill the (possibly NaN) first value within group by position, not skipping NaN
    gid = grp.ngroup()
    first_by_gid = pd.Series(df["wap"].values[first_pos.values], index=gid.values[first_pos.values])
    f["wap_ret_day"] = df["wap"] / gid.map(first_by_gid).values - 1
    del first_wap
    f["flag_change"] = df["imbalance_buy_sell_flag"] - grp["imbalance_buy_sell_flag"].shift(1)

    out = pd.concat([df, pd.DataFrame(f)], axis=1)

    tgrp = out.groupby("time_id", sort=False)
    for c in CS_COLS:
        out[f"{c}_cs"] = out[c] - tgrp[c].transform("mean")
    out["mkt_wap_ret_1"] = tgrp["wap_ret_1"].transform("mean")
    out["mkt_wap_ret_6"] = tgrp["wap_ret_6"].transform("mean")

    if index_weights is not None:
        w = out["stock_id"].map(index_weights).fillna(0.0).astype("float64")
        tmp = pd.DataFrame({"time_id": out["time_id"]})
        for c in IDX_COLS:
            tmp[f"n_{c}"] = out[c].fillna(0) * w
            tmp[f"d_{c}"] = w * out[c].notna()
        sums = tmp.groupby("time_id", sort=False).transform("sum")
        for c in IDX_COLS:
            out[f"idx_{c}"] = sums[f"n_{c}"] / sums[f"d_{c}"]
        for c in IDX_COLS:
            out[f"{c}_rel"] = out[c] - out[f"idx_{c}"]
        out["index_weight"] = w

    if revealed is not None and len(revealed):
        rv = revealed[["stock_id", "seconds_in_bucket", "revealed_target", "revealed_date_id"]].copy()
        rv = rv.sort_values(["stock_id", "revealed_date_id", "seconds_in_bucket"], kind="mergesort")
        rg = rv.groupby(["stock_id", "revealed_date_id"], sort=False)
        cs = rg["revealed_target"].cumsum()
        cn = rv["revealed_target"].notna().astype("int64").groupby([rv["stock_id"], rv["revealed_date_id"]]).cumsum()
        rv["prev_day_target_cummean"] = (cs / cn).where(rv["revealed_target"].notna())
        rv = rv.rename(columns={"revealed_target": "prev_day_target_same_s", "revealed_date_id": "_prev_date"})
        out["_prev_date"] = out["date_id"] - 1
        out = out.merge(rv, on=["stock_id", "_prev_date", "seconds_in_bucket"], how="left").drop(columns="_prev_date")
    else:
        out["prev_day_target_same_s"] = np.nan
        out["prev_day_target_cummean"] = np.nan
    return out


def feature_columns_pd(df):
    return [c for c in df.columns if c not in DROP]
