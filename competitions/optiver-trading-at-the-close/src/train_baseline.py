"""Day-2 baselines on the selection folds (holdout untouched).

Usage: .venv/Scripts/python.exe -I src/train_baseline.py [--rounds 800]
Appends rows to reports/experiments.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from cv import selection_folds
from features import RAW_COLS, build_features, feature_columns, revealed_from_train

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "reports/experiments.csv"


def index_weights() -> dict[int, float]:
    w = json.loads((ROOT / "reports/eda_day2.json").read_text())["index_weights"]["weights_by_stock"]
    return {int(k): v for k, v in w.items()}


def zero_sum(df: pl.DataFrame, pred_col: str, w: dict[int, float]) -> pl.Series:
    """Subtract the index-weighted mean prediction per time_id (weights renormalised over present stocks)."""
    t = df.select("time_id", "stock_id", pred_col).with_columns(
        pl.col("stock_id").replace_strict(w, default=0.0).alias("_w"))
    t = t.with_columns(((pl.col(pred_col) * pl.col("_w")).sum().over("time_id")
                        / pl.col("_w").sum().over("time_id")).alias("_m"))
    return t[pred_col] - t["_m"]


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def log_row(row: dict) -> None:
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)


def main(rounds: int) -> None:
    tr = pl.read_parquet(ROOT / "data/processed/train.parquet")
    t0 = time.time()
    feats = build_features(tr.drop("target"), revealed_from_train(tr)).join(
        tr.select("row_id", "target"), on="row_id").filter(pl.col("target").is_not_null())
    fcols = feature_columns(feats)
    print(f"features built: {feats.shape}, {len(fcols)} features, {time.time()-t0:.0f}s", flush=True)
    w = index_weights()

    params = dict(objective="l1", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=20, verbose=-1, seed=42)
    results = {k: [] for k in ["zeros", "stock_median", "lgb_raw", "lgb_v1", "lgb_v1_zerosum"]}
    curves = {}
    for f in selection_folds():
        trn = feats.filter(pl.col("date_id").is_between(f.train_start, f.train_end))
        val = feats.filter(pl.col("date_id").is_between(f.valid_start, f.valid_end))
        y = val["target"].to_numpy()
        results["zeros"].append(mae(y, 0 * y))
        med = trn.group_by("stock_id").agg(pl.col("target").median().alias("m"))
        pm = val.join(med, on="stock_id", how="left")["m"].fill_null(0).to_numpy()
        results["stock_median"].append(mae(y, pm))

        for name, cols in [("lgb_raw", ["stock_id"] + RAW_COLS), ("lgb_v1", fcols)]:
            t1 = time.time()
            dtr = lgb.Dataset(trn.select(cols).to_numpy(), trn["target"].to_numpy(), feature_name=cols,
                              categorical_feature=[], free_raw_data=True)
            ev = {}
            dva = lgb.Dataset(val.select(cols).to_numpy(), y, reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["v"],
                          callbacks=[lgb.record_evaluation(ev)])
            p = m.predict(val.select(cols).to_numpy())
            results[name].append(mae(y, p))
            curves[(name, f.name)] = ev["v"]["l1"]
            if name == "lgb_v1":
                vp = val.select("time_id", "stock_id").with_columns(pl.Series("p", p))
                results["lgb_v1_zerosum"].append(mae(y, zero_sum(vp, "p", w).to_numpy()))
                imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)[:15]
                print("  top gain:", [c for _, c in imp], flush=True)
            print(f"{f.name} {name}: {results[name][-1]:.4f} ({time.time()-t1:.0f}s)", flush=True)
        print(f"{f.name} summary:", {k: round(v[-1], 4) for k, v in results.items()}, flush=True)

    stamp = time.strftime("%Y-%m-%d %H:%M")
    for k, v in results.items():
        row = {"time": stamp, "experiment": k, "rounds": rounds if k.startswith("lgb") else "",
               "n_features": len(fcols) if k.startswith("lgb_v1") else (len(RAW_COLS) + 1 if k == "lgb_raw" else 0),
               **{f"mae_{f.name}": round(x, 5) for f, x in zip(selection_folds(), v)},
               "mae_mean": round(float(np.mean(v)), 5), "mae_std": round(float(np.std(v)), 5),
               "note": "selection folds only; fixed rounds, no early stopping"}
        log_row(row)
    # Fixed-rounds check: where would the curve have been best (for information only, not used).
    for (name, fold), c in curves.items():
        print(f"curve {name} {fold}: best_iter={int(np.argmin(c))+1} best={min(c):.4f} final={c[-1]:.4f}")
    print(json.dumps({k: [round(np.mean(v), 4), round(np.std(v), 4)] for k, v in results.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=800)
    main(ap.parse_args().rounds)
