"""Week-1 experiments on selection folds (holdout untouched):
  A) feature set v1 vs v2 (index-relative features), static model;
  B) online refit every K days inside the validation block (uses revealed targets of past days only).

Usage: .venv/Scripts/python.exe -I src/experiments_week1.py {features|online} [--rounds N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import EMBARGO_DAYS, selection_folds  # noqa: E402
from features import build_features, feature_columns, revealed_from_train  # noqa: E402
from train_baseline import index_weights, log_row, mae, zero_sum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=20, verbose=-1, seed=42)


def load(version: str) -> tuple[pl.DataFrame, list[str]]:
    tr = pl.read_parquet(ROOT / "data/processed/train.parquet")
    w = index_weights() if version == "v2" else None
    f = build_features(tr.drop("target"), revealed_from_train(tr), w).join(
        tr.select("row_id", "target"), on="row_id").filter(pl.col("target").is_not_null())
    return f, feature_columns(f)


def fit(df: pl.DataFrame, cols: list[str], rounds: int) -> lgb.Booster:
    d = lgb.Dataset(df.select(cols).to_numpy(), df["target"].to_numpy(), feature_name=cols, free_raw_data=True)
    return lgb.train(PARAMS, d, num_boost_round=rounds)


def run_features(rounds: int) -> None:
    w = index_weights()
    res = {}
    for v in ["v1", "v2"]:
        feats, cols = load(v)
        res[v] = []
        for f in selection_folds():
            t0 = time.time()
            trn = feats.filter(pl.col("date_id").is_between(f.train_start, f.train_end))
            val = feats.filter(pl.col("date_id").is_between(f.valid_start, f.valid_end))
            p = fit(trn, cols, rounds).predict(val.select(cols).to_numpy())
            vp = val.select("time_id", "stock_id").with_columns(pl.Series("p", p))
            res[v].append(mae(val["target"].to_numpy(), zero_sum(vp, "p", w).to_numpy()))
            print(f"{v} {f.name}: {res[v][-1]:.4f} ({len(cols)} feats, {time.time()-t0:.0f}s)", flush=True)
        log_row({"time": time.strftime("%Y-%m-%d %H:%M"), "experiment": f"lgb_{v}_zerosum", "rounds": rounds,
                 "n_features": len(cols), **{f"mae_{f.name}": round(x, 5) for f, x in zip(selection_folds(), res[v])},
                 "mae_mean": round(float(np.mean(res[v])), 5), "mae_std": round(float(np.std(res[v])), 5),
                 "note": "week1 feature comparison; zero-sum applied"})
    d = np.array(res["v2"]) - np.array(res["v1"])
    print(json.dumps({"v1": res["v1"], "v2": res["v2"], "v2_minus_v1": d.tolist(), "mean_diff": float(d.mean())}))


def run_online(rounds: int, version: str, k_days: int, folds: list[str]) -> None:
    """Static model vs refit every k_days within the validation block.

    At validation date d, a refit may use all targets of dates <= d-1 (revealed by the API by the
    end of day d-1); we additionally keep the 1-day embargo used offline, so refit data ends at d-1-EMBARGO... no:
    the API reveals d-1 targets during day d, so a refit performed at the start of day d can use dates <= d-2
    fully plus part of d-1. We conservatively use dates <= d-2.
    """
    w = index_weights()
    feats, cols = load(version)
    out = {}
    for f in [x for x in selection_folds() if x.name in folds]:
        val = feats.filter(pl.col("date_id").is_between(f.valid_start, f.valid_end))
        t0 = time.time()
        static = fit(feats.filter(pl.col("date_id").is_between(f.train_start, f.train_end)), cols, rounds)
        p_static = static.predict(val.select(cols).to_numpy())
        p_online = np.empty(val.height)
        dates = val["date_id"].to_numpy()
        model = static
        for start in range(f.valid_start, f.valid_end + 1, k_days):
            if start > f.valid_start:
                model = fit(feats.filter(pl.col("date_id").is_between(f.train_start, start - 2)), cols, rounds)
            m = (dates >= start) & (dates < start + k_days)
            p_online[m] = model.predict(val.filter(pl.Series(m)).select(cols).to_numpy())
        y = val["target"].to_numpy()
        r = {}
        for name, p in [("static", p_static), ("online", p_online)]:
            vp = val.select("time_id", "stock_id").with_columns(pl.Series("p", p))
            r[name] = mae(y, zero_sum(vp, "p", w).to_numpy())
        out[f.name] = r
        print(f"{f.name} k={k_days}: static {r['static']:.4f} online {r['online']:.4f} "
              f"diff {r['online']-r['static']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(json.dumps(out))
    log_row({"time": time.strftime("%Y-%m-%d %H:%M"), "experiment": f"online_{version}_k{k_days}", "rounds": rounds,
             "n_features": len(cols), **{f"mae_{k}": round(v["online"], 5) for k, v in out.items()},
             "mae_mean": round(float(np.mean([v["online"] for v in out.values()])), 5),
             "mae_std": round(float(np.mean([v["online"] - v["static"] for v in out.values()])), 5),
             "note": "online refit; mae_std column holds mean(online-static) for this row"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["features", "online"])
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--version", default="v2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", default="f2,f3")
    a = ap.parse_args()
    if a.mode == "features":
        run_features(a.rounds)
    else:
        run_online(a.rounds, a.version, a.k, a.folds.split(","))
