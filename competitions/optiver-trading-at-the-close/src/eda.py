"""Day-2 EDA: missingness, target structure, and recovery of synthetic-index weights.

Usage: .venv/Scripts/python.exe -I src/eda.py
Writes reports/eda_day2.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tr = pl.read_parquet(ROOT / "data/processed/train.parquet")
    out: dict = {}

    # far/near price availability by second
    miss = (tr.group_by("seconds_in_bucket")
            .agg(pl.col("far_price").is_null().mean().alias("far_null"),
                 pl.col("near_price").is_null().mean().alias("near_null"))
            .sort("seconds_in_bucket"))
    out["far_near_null_by_s"] = {int(r[0]): [round(r[1], 4), round(r[2], 4)] for r in miss.iter_rows()}

    # target abs mean by second
    t = tr.drop_nulls("target")
    by_s = t.group_by("seconds_in_bucket").agg(pl.col("target").abs().mean().alias("mae0")).sort("seconds_in_bucket")
    out["zero_mae_by_s"] = {int(a): round(b, 3) for a, b in by_s.iter_rows()}

    # Cross-sectional structure: does the stock-weighted target sum to ~0 per time_id?
    # target_i = 1e4 * (r_i - r_index), r_index = sum w_j r_j. Recover r_i from 60s-ahead WAP.
    t = t.sort(["stock_id", "date_id", "seconds_in_bucket"]).with_columns(
        (pl.col("wap").shift(-6).over(["stock_id", "date_id"]) / pl.col("wap") - 1).alias("r60")
    ).drop_nulls("r60")
    piv_r = t.pivot(on="stock_id", index="time_id", values="r60").sort("time_id")
    piv_t = t.pivot(on="stock_id", index="time_id", values="target").sort("time_id")
    cols = [c for c in piv_r.columns if c != "time_id"]
    R = piv_r.select(cols).to_numpy()
    T = piv_t.select(cols).to_numpy()
    # r_index_t estimated per (t, i) as r_i - target_i/1e4; average over stocks per time
    idx_est = np.nanmedian(R - T / 1e4, axis=1)
    ok = np.isfinite(idx_est) & np.isfinite(R).all(axis=1)
    w, *_ = np.linalg.lstsq(np.nan_to_num(R[ok]), idx_est[ok], rcond=None)
    resid = idx_est[ok] - np.nan_to_num(R[ok]) @ w
    out["index_weights"] = {
        "n_times_used": int(ok.sum()),
        "sum_w": float(w.sum()),
        "r2": float(1 - resid.var() / idx_est[ok].var()),
        "weights_by_stock": {int(c): round(float(x), 5) for c, x in zip(cols, w)},
    }
    out["index_consistency_note"] = "idx_est = median_i(r60_i - target_i/1e4); spread across stocks measures definition fit"
    spread = np.nanstd(R - T / 1e4, axis=1)
    out["index_est_crossstock_std_median"] = float(np.nanmedian(spread))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/eda_day2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "index_weights"}, indent=1)[:3000])
    iw = out["index_weights"]
    ws = sorted(iw["weights_by_stock"].values())
    print("index weights: n", iw["n_times_used"], "sum", round(iw["sum_w"], 4), "r2", round(iw["r2"], 6),
          "min/max", ws[0], ws[-1], "distinct(rounded 3)", len(set(round(x, 3) for x in ws)))


if __name__ == "__main__":
    main()
