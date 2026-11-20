"""The pandas port used in the Kaggle notebook must match the polars reference features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle"))

from features import build_features, feature_columns, revealed_from_train  # noqa: E402
from features_pd import build_features_pd, feature_columns_pd  # noqa: E402

W = {int(k): v for k, v in json.loads((ROOT / "reports/eda_day2.json").read_text())["index_weights"]["weights_by_stock"].items()}


def test_pandas_port_matches_polars() -> None:
    # include days with missing wap rows (stock 131 day 35) and missing targets
    tr = pl.read_parquet(ROOT / "data/processed/train.parquet").filter(pl.col("date_id").is_in([34, 35, 36, 400, 401]))
    ref = build_features(tr.drop("target"), revealed_from_train(tr), W).sort("row_id")
    pdf = tr.drop("target").to_pandas()
    rv = revealed_from_train(tr).to_pandas()
    got = build_features_pd(pdf, rv, W).sort_values("row_id").reset_index(drop=True)
    assert sorted(feature_columns(ref)) == sorted(feature_columns_pd(got))
    assert len(got) == ref.height
    bad = []
    for c in feature_columns(ref):
        a = ref[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
        b = got[c].astype("float64").to_numpy()
        same_nan = np.isnan(a) == np.isnan(b)
        close = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        if not (same_nan.all() and close.all()):
            bad.append((c, int((~close).sum())))
    assert not bad, f"mismatching features: {bad}"
