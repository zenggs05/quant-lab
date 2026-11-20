"""One-shot holdout evaluation (date_id 400-480) of configs frozen on selection folds.

Pre-declared configs (2026-11-19, before looking at holdout):
  - zeros
  - lgb_v1_static: v1 features, 600 rounds, index zero-sum, trained on 0..398
  - lgb_v1_online_k10: same, refit every 10 days on dates <= d-2
Usage: .venv/Scripts/python.exe -I src/holdout_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import holdout_fold  # noqa: E402
from experiments_week1 import fit, load  # noqa: E402
from train_baseline import index_weights, log_row, mae, zero_sum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    w = index_weights()
    f = holdout_fold()
    feats, cols = load("v1")
    val = feats.filter(pl.col("date_id").is_between(f.valid_start, f.valid_end))
    y = val["target"].to_numpy()
    zs = lambda p: zero_sum(val.select("time_id", "stock_id").with_columns(pl.Series("p", p)), "p", w).to_numpy()
    res = {"zeros": mae(y, 0 * y)}
    t0 = time.time()
    static = fit(feats.filter(pl.col("date_id").is_between(f.train_start, f.train_end)), cols, 600)
    p_static = static.predict(val.select(cols).to_numpy())
    res["lgb_v1_static"] = mae(y, zs(p_static))
    res["lgb_v1_static_no_zerosum"] = mae(y, p_static)
    print(res, f"{time.time()-t0:.0f}s", flush=True)
    dates = val["date_id"].to_numpy()
    p_on = np.empty(val.height)
    model = static
    for start in range(f.valid_start, f.valid_end + 1, 10):
        if start > f.valid_start:
            model = fit(feats.filter(pl.col("date_id").is_between(f.train_start, start - 2)), cols, 600)
        m = (dates >= start) & (dates < start + 10)
        p_on[m] = model.predict(val.filter(pl.Series(m)).select(cols).to_numpy())
    res["lgb_v1_online_k10"] = mae(y, zs(p_on))
    # per-second-bucket breakdown for the static model
    br = (val.select("seconds_in_bucket").with_columns(pl.Series("e", np.abs(y - zs(p_static))), pl.Series("e0", np.abs(y)))
          .with_columns((pl.col("seconds_in_bucket") // 60 * 60).alias("bucket"))
          .group_by("bucket").agg(pl.col("e").mean(), pl.col("e0").mean()).sort("bucket"))
    res["static_by_minute"] = {int(b): [round(e, 4), round(e0, 4)] for b, e, e0 in br.iter_rows()}
    print(json.dumps(res, indent=1))
    (ROOT / "reports/holdout_once.json").write_text(json.dumps(res, indent=1))
    for k in ["zeros", "lgb_v1_static", "lgb_v1_online_k10"]:
        log_row({"time": time.strftime("%Y-%m-%d %H:%M"), "experiment": f"HOLDOUT_{k}", "rounds": 600, "n_features": len(cols),
                 "mae_f0": "", "mae_f1": "", "mae_f2": "", "mae_f3": "", "mae_mean": round(res[k], 5), "mae_std": "",
                 "note": "one-shot holdout date_id 400-480"})


if __name__ == "__main__":
    main()
