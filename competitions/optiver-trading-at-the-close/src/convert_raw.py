"""Convert raw Optiver CSVs to parquet and print basic integrity checks.

Usage:
    .venv/Scripts/python.exe -I src/convert_raw.py <extracted_dir> <out_dir>

The extracted directory is treated as untrusted data: files are only read as CSV,
nothing from it is imported or executed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    train = pl.read_csv(src / "train.csv", infer_schema_length=100_000)
    # The CSV stores some missing targets as the literal "nan"; normalise all float NaN to null.
    train = train.with_columns(pl.col(pl.Float64).fill_nan(None))
    train.write_parquet(out / "train.parquet", compression="zstd")

    for name in ("test", "revealed_targets", "sample_submission"):
        df = pl.read_csv(src / "example_test_files" / f"{name}.csv", infer_schema_length=100_000)
        df.write_parquet(out / f"example_{name}.parquet", compression="zstd")

    per_day = train.group_by("date_id").agg(pl.col("stock_id").n_unique().alias("n_stocks"))
    snaps = train.group_by(["stock_id", "date_id"]).len()
    dup = train.select(["stock_id", "date_id", "seconds_in_bucket"]).is_duplicated().sum()
    summary = {
        "rows": train.height,
        "columns": train.columns,
        "dtypes": {c: str(t) for c, t in train.schema.items()},
        "n_stocks": train["stock_id"].n_unique(),
        "date_id_range": [int(train["date_id"].min()), int(train["date_id"].max())],
        "n_dates": train["date_id"].n_unique(),
        "stocks_per_day_min_max": [int(per_day["n_stocks"].min()), int(per_day["n_stocks"].max())],
        "snapshots_per_stock_day_min_max": [int(snaps["len"].min()), int(snaps["len"].max())],
        "seconds_in_bucket_values": sorted(train["seconds_in_bucket"].unique().to_list()),
        "duplicate_keys": int(dup),
        "null_counts": {c: int(v) for c, v in zip(train.columns, train.null_count().row(0)) if v},
        "target_describe": {
            k: float(v)
            for k, v in zip(
                ["mean", "std", "min", "p01", "p50", "p99", "max"],
                train.select(
                    pl.col("target").mean().alias("mean"),
                    pl.col("target").std().alias("std"),
                    pl.col("target").min().alias("min"),
                    pl.col("target").quantile(0.01).alias("p01"),
                    pl.col("target").median().alias("p50"),
                    pl.col("target").quantile(0.99).alias("p99"),
                    pl.col("target").max().alias("max"),
                ).row(0),
            )
        },
        "zero_pred_mae_all": float(train["target"].abs().mean()),
        "zero_pred_mae_date_ge_385_last20pct": float(train.filter(pl.col("date_id") >= 385)["target"].abs().mean()),
        "zero_pred_mae_date_ge_400": float(train.filter(pl.col("date_id") >= 400)["target"].abs().mean()),
    }
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
