"""End-to-end check: the notebook's streaming predictions through the local API replay must equal
offline batch predictions from the same model on the same rows (both reveal modes).

Run after kaggle/build_notebook.py. Slow-ish (~1 min)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "kaggle" / "build"


@pytest.mark.parametrize("has_time_id", ["0", "1"])
@pytest.mark.parametrize("reveal_mode", ["per_second", "full_day"])
def test_streaming_equals_batch(reveal_mode: str, has_time_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "OPTIVER_MODE": "model", "OPTIVER_DATA_DIR": str(ROOT / "data/raw/extracted"), "OPTIVER_ROUNDS": "30",
        "OPTIVER_TRAIN_MIN_DATE": "440", "OPTIVER_TRAIN_MAX_DATE": "477", "OPTIVER_LOCAL_DATES": "478-479",
        "OPTIVER_REVEAL_MODE": reveal_mode, "OPTIVER_TEST_HAS_TIME_ID": has_time_id, "OPTIVER_THREADS": "8", "OPTIVER_REFIT_EVERY": "0", "OPTIVER_CB_ITERS": "20",
        "OPTIVER_LOCAL_SUBMISSION": str(tmp_path / "sub.csv"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.path.insert(0, str(ROOT / "kaggle" / "local_api"))
    spec = importlib.util.spec_from_file_location("nb", BUILD / "submission_model.py")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    import optiver2023

    captured = {}
    orig_train = nb.train_model

    def train_and_capture(tr=None, with_cb=False):
        m, cols, cb = orig_train(tr, with_cb=with_cb)
        captured["model"], captured["cols"], captured["cb"] = m, cols, cb
        return m, cols, cb

    nb.train_model = train_and_capture
    nb.main(optiver2023)
    sub = pd.read_csv(tmp_path / "sub.csv")

    tr = pd.read_parquet(ROOT / "data/processed/train.parquet")
    tr = tr[tr["date_id"].isin([477, 478, 479])]
    rv = tr[["stock_id", "seconds_in_bucket", "target", "date_id"]].rename(
        columns={"target": "revealed_target", "date_id": "revealed_date_id"})
    feats = nb.build_features_pd(tr.drop(columns=["target"]), rv, nb.INDEX_WEIGHTS)
    feats = feats[feats["date_id"].isin([478, 479])]
    Xb = feats[captured["cols"]].to_numpy(dtype=np.float32)
    p = captured["model"].predict(Xb)
    if captured["cb"] is not None:
        p = (1 - nb.CB_WEIGHT) * p + nb.CB_WEIGHT * captured["cb"].predict(Xb)
    batch = pd.DataFrame({"row_id": feats["row_id"].to_numpy(), "time_id": feats["time_id"].to_numpy(),
                          "stock_id": feats["stock_id"].to_numpy(), "p": p})
    batch["p"] = batch.groupby("time_id", group_keys=False).apply(
        lambda g: pd.Series(nb.zero_sum(g["stock_id"].to_numpy(), g["p"].to_numpy()), index=g.index))
    m = sub.merge(batch, on="row_id")
    assert len(m) == len(sub) == len(batch)
    assert np.abs(m["target"] - m["p"]).max() < 1e-6
