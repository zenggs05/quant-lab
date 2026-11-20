"""Local stand-in for the competition's optiver2023 API, used only for end-to-end testing.

Written from the documented behaviour and from example_test_files (the official binary is never
imported). Serves (test, revealed_targets, sample_prediction) per time_id for the dates in
OPTIVER_LOCAL_DATES (e.g. "478-480"), using data/processed/train.parquet.

REVEAL_MODE ("per_second" default, or "full_day"): per_second reveals day d-1 second s at day d
second s (matches example files); full_day reveals all of day d-1 at s=0 (other reading of the docs).
In both modes the `seconds_in_bucket` column of revealed_targets is the *current* second, as in the
example file, so consumers must derive the revealed second from revealed_time_id.
"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class _Env:
    def __init__(self):
        a, b = os.environ.get("OPTIVER_LOCAL_DATES", "478-480").split("-")
        self.dates = list(range(int(a), int(b) + 1))
        self.reveal_mode = os.environ.get("OPTIVER_REVEAL_MODE", "per_second")
        self.data = pd.read_parquet(os.path.join(_ROOT, "data", "processed", "train.parquet"),
                                    filters=[("date_id", ">=", self.dates[0] - 1), ("date_id", "<=", self.dates[-1])]
                                    ).reset_index(drop=True)
        self.predictions = []
        self._served = None
        self._used = False

    def iter_test(self):
        if self._used:
            raise RuntimeError("iter_test can only be called once")
        self._used = True
        cols = [c for c in self.data.columns if c != "target"]
        for d in self.dates:
            day = self.data[self.data["date_id"] == d]
            earlier = self.data[self.data["date_id"] < d]["date_id"]
            prev = self.data[self.data["date_id"] == (earlier.max() if len(earlier) else -1)]
            for s in sorted(day["seconds_in_bucket"].unique()):
                test = day[day["seconds_in_bucket"] == s][cols].copy()
                test["currently_scored"] = False
                tid = int(test["time_id"].iloc[0])
                if os.environ.get("OPTIVER_TEST_HAS_TIME_ID", "0") != "1":
                    test = test.drop(columns=["time_id"])  # the live Kaggle API omits time_id from `test`
                if self.reveal_mode == "full_day":
                    src = prev if s == 0 else prev.iloc[0:0]
                else:
                    src = prev[prev["seconds_in_bucket"] == s]
                rv = pd.DataFrame({
                    "stock_id": src["stock_id"].to_numpy(),
                    "date_id": d,
                    "seconds_in_bucket": s,
                    "time_id": tid,
                    "revealed_target": src["target"].to_numpy(),
                    "revealed_date_id": src["date_id"].to_numpy(),
                    "revealed_time_id": src["time_id"].to_numpy(),
                })
                if s != 0 or self.reveal_mode == "per_second":
                    # example file appends one all-null row per non-zero second
                    if s != 0:
                        rv = pd.concat([rv, pd.DataFrame({"stock_id": [np.nan], "date_id": [d], "seconds_in_bucket": [s],
                                                          "time_id": [tid], "revealed_target": [np.nan],
                                                          "revealed_date_id": [np.nan], "revealed_time_id": [np.nan]})],
                                       ignore_index=True)
                sample = pd.DataFrame({"time_id": tid, "row_id": test["row_id"].to_numpy(), "target": 0})
                self._served = sample
                yield test.reset_index(drop=True), rv, sample
                if self._served is not None:
                    raise RuntimeError("predict() must be called before the next iteration")
        sub = pd.concat(self.predictions, ignore_index=True)
        sub.to_csv(os.environ.get("OPTIVER_LOCAL_SUBMISSION", "submission.csv"), index=False)

    def predict(self, df):
        if self._served is None:
            raise RuntimeError("call iter_test first")
        if not isinstance(df, pd.DataFrame) or list(df.columns) != ["time_id", "row_id", "target"]:
            raise RuntimeError(f"bad prediction frame: {type(df)} {getattr(df, 'columns', None)}")
        if len(df) != len(self._served) or df["target"].isna().any():
            raise RuntimeError("prediction length mismatch or NaN")
        self.predictions.append(df.copy())
        self._served = None


def make_env():
    return _Env()
