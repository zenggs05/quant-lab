"""Online refit inside the notebook must train on exactly the rows offline CV would use:
train rows plus test days with date <= d-2 (targets taken from the API's revealed_targets)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "kaggle" / "build"
KEYS = ["stock_id", "date_id", "seconds_in_bucket"]


@pytest.mark.parametrize("reveal_mode", ["per_second", "full_day"])
def test_refit_frames_match_offline(reveal_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "OPTIVER_MODE": "model", "OPTIVER_DATA_DIR": str(ROOT / "data/raw/extracted"), "OPTIVER_ROUNDS": "5",
        "OPTIVER_TRAIN_MIN_DATE": "455", "OPTIVER_TRAIN_MAX_DATE": "466", "OPTIVER_LOCAL_DATES": "467-476",
        "OPTIVER_REVEAL_MODE": reveal_mode, "OPTIVER_THREADS": "8", "OPTIVER_REFIT_EVERY": "3", "OPTIVER_CB_WEIGHT": "0",
        "OPTIVER_LOCAL_SUBMISSION": str(tmp_path / "sub.csv"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.path.insert(0, str(ROOT / "kaggle" / "local_api"))
    spec = importlib.util.spec_from_file_location("nb_refit", BUILD / "submission_model.py")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    import optiver2023

    frames = []
    orig = nb.train_model

    def spy(tr=None, with_cb=False):
        if tr is not None:
            frames.append(tr.copy())
        return orig(tr, with_cb=with_cb)

    nb.train_model = spy
    nb.main(optiver2023)

    full = pd.read_parquet(ROOT / "data/processed/train.parquet")
    # first call = initial training (train 455..466); refits at test days 4,7,10 -> dates 470,473,476
    assert len(frames) == 4, f"expected initial fit + 3 refits, got {len(frames)}"
    for tr, d in zip(frames[1:], [470, 473, 476]):
        expect = full[(full["date_id"] >= 455) & (full["date_id"] <= d - 2)]
        assert tr["date_id"].max() == d - 2, (d, tr["date_id"].max())
        assert len(tr) == len(expect), (d, len(tr), len(expect))
        m = tr.merge(expect, on=KEYS, suffixes=("", "_x"))
        assert len(m) == len(expect)
        for c in ["target", "wap", "imbalance_size", "far_price", "row_id"]:
            a, b = m[c], m[f"{c}_x"]
            if a.dtype == object:
                assert (a == b).all(), c
            else:
                assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), c


def test_refit_cadence_ignores_replayed_train_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live API first replays the last train dates; refit cadence must count only new dates."""
    env = {
        "OPTIVER_MODE": "model", "OPTIVER_DATA_DIR": str(ROOT / "data/raw/extracted"), "OPTIVER_ROUNDS": "5",
        "OPTIVER_TRAIN_MIN_DATE": "455", "OPTIVER_TRAIN_MAX_DATE": "466", "OPTIVER_LOCAL_DATES": "465-476",
        "OPTIVER_REVEAL_MODE": "per_second", "OPTIVER_THREADS": "8", "OPTIVER_REFIT_EVERY": "3",
        "OPTIVER_CB_WEIGHT": "0", "OPTIVER_LOCAL_SUBMISSION": str(tmp_path / "sub.csv"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.path.insert(0, str(ROOT / "kaggle" / "local_api"))
    spec = importlib.util.spec_from_file_location("nb_cadence", BUILD / "submission_model.py")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    import optiver2023

    maxdates = []
    orig = nb.train_model

    def spy(tr=None, with_cb=False):
        if tr is not None:
            maxdates.append(int(tr["date_id"].max()))
        return orig(tr, with_cb=with_cb)

    nb.train_model = spy
    nb.main(optiver2023)
    # 2 replayed train days (465-466) then new days 467..476. Refits at new days 4/7/10 = dates 470/473/476
    # using data up to d-2. (Counting replayed days too would shift refits to 471/474 - caught by this test.)
    assert maxdates == [466, 468, 471, 474], maxdates


def test_training_prev_day_targets_follow_actual_predecessor() -> None:
    """With a gap in dates, training features must take previous-day targets from the previous present
    date (as serving does), not from the missing literal d-1."""
    spec = importlib.util.spec_from_file_location("nb_gap", BUILD / "submission_model.py")
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    full = pd.read_parquet(ROOT / "data/processed/train.parquet")
    tr = full[full["date_id"].isin([300, 302])][nb.RAW_SCHEMA].reset_index(drop=True)  # 301 missing
    seen = {}
    orig = nb.build_features_pd

    def spy(day, rv, w):
        out = orig(day, rv, w)
        seen["f"] = out
        return out

    nb.build_features_pd = spy
    nb.build_training_matrix(tr)
    f = seen["f"]
    d302 = f[f["date_id"] == 302].merge(
        full[full["date_id"] == 300][["stock_id", "seconds_in_bucket", "target"]], on=["stock_id", "seconds_in_bucket"])
    assert len(d302) > 0
    assert np.allclose(d302["prev_day_target_same_s"], d302["target"], equal_nan=True)
