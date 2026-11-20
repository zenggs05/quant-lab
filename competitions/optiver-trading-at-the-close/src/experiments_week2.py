"""Week-2 experiments on selection folds f0-f3 (holdout untouched):
  lgb_static, lgb_online_k10 (refit every 10 days on dates <= d-2), cb_static (CatBoost MAE).
Saves per-fold out-of-fold predictions to reports/oof_week2/<fold>.parquet for blend analysis.

Usage: .venv/Scripts/python.exe -I src/experiments_week2.py {lgb|cb} [--threads N] [--cb-iters N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import selection_folds  # noqa: E402
from experiments_week1 import PARAMS, load  # noqa: E402

import lightgbm as lgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "oof_week2"


def fit_lgb(df: pl.DataFrame, cols: list[str], threads: int) -> lgb.Booster:
    p = dict(PARAMS, num_threads=threads)
    d = lgb.Dataset(df.select(cols).to_numpy(), df["target"].to_numpy(), feature_name=cols, free_raw_data=True)
    return lgb.train(p, d, num_boost_round=600)


def run_lgb(threads: int) -> None:
    feats, cols = load("v1")
    for f in selection_folds():
        t0 = time.time()
        val = feats.filter(pl.col("date_id").is_between(f.valid_start, f.valid_end))
        static = fit_lgb(feats.filter(pl.col("date_id").is_between(f.train_start, f.train_end)), cols, threads)
        X = val.select(cols).to_numpy()
        p_static = static.predict(X)
        dates = val["date_id"].to_numpy()
        p_on = np.empty(val.height)
        model = static
        for start in range(f.valid_start, f.valid_end + 1, 10):
            if start > f.valid_start:
                model = fit_lgb(feats.filter(pl.col("date_id").is_between(f.train_start, start - 2)), cols, threads)
            m = (dates >= start) & (dates < start + 10)
            p_on[m] = model.predict(X[m])
        out = val.select("row_id", "time_id", "stock_id", "date_id", "seconds_in_bucket", "target").with_columns(
            pl.Series("lgb_static", p_static), pl.Series("lgb_online", p_on))
        OUT.mkdir(parents=True, exist_ok=True)
        out.write_parquet(OUT / f"{f.name}_lgb.parquet")
        print(f"{f.name} lgb done ({time.time()-t0:.0f}s)", flush=True)


def run_cb(threads: int, iters: int) -> None:
    from catboost import CatBoostRegressor

    feats, cols = load("v1")
    for f in selection_folds():
        t0 = time.time()
        trn = feats.filter(pl.col("date_id").is_between(f.train_start, f.train_end))
        val = feats.filter(pl.col("date_id").is_between(f.valid_start, f.valid_end))
        m = CatBoostRegressor(loss_function="MAE", iterations=iters, learning_rate=0.1, depth=8,
                              border_count=128, thread_count=threads, random_seed=42, verbose=0,
                              allow_writing_files=False)
        m.fit(trn.select(cols).to_numpy().astype(np.float32), trn["target"].to_numpy())
        p = m.predict(val.select(cols).to_numpy().astype(np.float32))
        OUT.mkdir(parents=True, exist_ok=True)
        val.select("row_id").with_columns(pl.Series("cb_static", p)).write_parquet(OUT / f"{f.name}_cb.parquet")
        print(f"{f.name} cb done ({time.time()-t0:.0f}s, {threads} threads, {iters} iters)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["lgb", "cb"])
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--cb-iters", type=int, default=1500)
    a = ap.parse_args()
    run_lgb(a.threads) if a.which == "lgb" else run_cb(a.threads, a.cb_iters)
