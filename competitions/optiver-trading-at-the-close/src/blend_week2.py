"""Blend analysis on week-2 OOF predictions (selection folds only).

Blend weight w for CatBoost in p = (1-w)*lgb + w*cb is chosen by leave-one-fold-out:
for each fold, w is picked on the other three folds and applied to the held-out fold.
Zero-sum post-processing is applied after blending.

Usage: .venv/Scripts/python.exe -I src/blend_week2.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import index_weights, log_row, mae, zero_sum  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "reports" / "oof_week2"
FOLDS = ["f0", "f1", "f2", "f3"]
GRID = np.round(np.arange(0, 1.01, 0.1), 2)


def load(fold: str) -> pl.DataFrame:
    a = pl.read_parquet(OOF / f"{fold}_lgb.parquet")
    b = pl.read_parquet(OOF / f"{fold}_cb.parquet")
    return a.join(b, on="row_id", how="inner")


def main() -> None:
    w_idx = index_weights()
    data = {f: load(f) for f in FOLDS}

    def score(df: pl.DataFrame, col: str) -> float:
        zs = zero_sum(df.select("time_id", "stock_id", col), col, w_idx).to_numpy()
        return mae(df["target"].to_numpy(), zs)

    def blend(df: pl.DataFrame, lgb_col: str, w: float) -> pl.DataFrame:
        return df.with_columns(((1 - w) * pl.col(lgb_col) + w * pl.col("cb_static")).alias("blend"))

    res: dict = {"single": {}, "blend_lofo": {}}
    for col in ["lgb_static", "lgb_online", "cb_static"]:
        res["single"][col] = [score(data[f], col) for f in FOLDS]

    for base in ["lgb_static", "lgb_online"]:
        grid_scores = {f: [score(blend(data[f], base, w), "blend") for w in GRID] for f in FOLDS}
        picked, held = [], []
        for f in FOLDS:
            others = np.mean([grid_scores[g] for g in FOLDS if g != f], axis=0)
            wi = int(np.argmin(others))
            picked.append(float(GRID[wi]))
            held.append(grid_scores[f][wi])
        res["blend_lofo"][base] = {"weights_picked": picked, "heldout_mae": held,
                                   "grid_mean": dict(zip(map(float, GRID), np.mean([grid_scores[f] for f in FOLDS], axis=0).round(5).tolist()))}

    summary = {
        k: round(float(np.mean(v)), 5) for k, v in res["single"].items()
    } | {f"blend_{k}": round(float(np.mean(v["heldout_mae"])), 5) for k, v in res["blend_lofo"].items()}
    res["summary_mean"] = summary
    print(json.dumps(res, indent=1))
    (ROOT / "reports" / "blend_week2.json").write_text(json.dumps(res, indent=1))
    stamp = time.strftime("%Y-%m-%d %H:%M")
    rows = [(k, v) for k, v in res["single"].items()] + [(f"blend_{k}_lofo", v["heldout_mae"]) for k, v in res["blend_lofo"].items()]
    for name, v in rows:
        log_row({"time": stamp, "experiment": f"w2_{name}", "rounds": "", "n_features": 61,
                 **{f"mae_{f}": round(x, 5) for f, x in zip(FOLDS, v)},
                 "mae_mean": round(float(np.mean(v)), 5), "mae_std": round(float(np.std(v)), 5),
                 "note": "week2; zero-sum applied; blends use leave-one-fold-out weights"})


if __name__ == "__main__":
    main()
