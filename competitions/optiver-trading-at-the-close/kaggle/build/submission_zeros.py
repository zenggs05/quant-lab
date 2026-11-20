# Optiver - Trading at the Close: late-submission notebook (single cell).
# Train LightGBM on all train data inside the notebook, then predict through the optiver2023 API.
# Paste this whole file into one cell of a notebook copied from a 2023 notebook
# (keep "Pin to original environment": the API binary needs Python 3.10). Internet must be off.
# Features: identical to the local research code (checked by tests/test_parity.py).
import gc
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

MODE = os.environ.get("OPTIVER_MODE", "zeros")  # "model" or "zeros" (zeros = API/runtime/scoring probe)
DATA_DIR = os.environ.get("OPTIVER_DATA_DIR", "/kaggle/input/optiver-trading-at-the-close")
TRAIN_ROUNDS = int(os.environ.get("OPTIVER_ROUNDS", "600"))
TRAIN_MAX_DATE = int(os.environ.get("OPTIVER_TRAIN_MAX_DATE", "480"))
TRAIN_MIN_DATE = int(os.environ.get("OPTIVER_TRAIN_MIN_DATE", "0"))
FEATURE_VERSION = os.environ.get("OPTIVER_FEATURES", "v1")  # v1 chosen: v2 index-relative feats gave no CV gain
CHUNK_DAYS = 40
# Online refit: every REFIT_EVERY test days, retrain on train + test days whose targets are fully
# revealed (dates <= d-2 at the start of test day d; matches the CV protocol). 0 disables.
REFIT_EVERY = int(os.environ.get("OPTIVER_REFIT_EVERY", "10"))
# Safety budget: skip further refits if elapsed wall time exceeds this many seconds.
# CatBoost blend: trained once (no refit), weight chosen by leave-one-fold-out CV (0.3 on every fold).
CB_WEIGHT = float(os.environ.get("OPTIVER_CB_WEIGHT", "0.3"))
CB_ITERS = int(os.environ.get("OPTIVER_CB_ITERS", "1500"))
REFIT_TIME_BUDGET = float(os.environ.get("OPTIVER_REFIT_TIME_BUDGET", str(7 * 3600)))
NUM_THREADS = int(os.environ.get("OPTIVER_THREADS", str(os.cpu_count() or 4)))

# ---- features (inlined copy of kaggle/features_pd.py; keep in sync) ----
LAG_COLS = ["wap", "reference_price", "imbalance_size", "matched_size", "bid_size", "ask_size"]
LAGS = (1, 2, 3, 6)
CS_COLS = ["wap_ret_1", "wap_ret_6", "wap_ret_day", "signed_imb_ratio", "book_imb", "imb_ratio"]
IDX_COLS = ["wap_ret_1", "wap_ret_3", "wap_ret_6", "wap_ret_day", "signed_imb_ratio", "book_imb"]
DROP = {"date_id", "time_id", "row_id", "target", "currently_scored", "mid"}


def _ratio(a, b):
    return a / (b + 1e-9)


def build_features_pd(day_rows, revealed=None, index_weights=None):
    """day_rows: auction rows (no target). revealed: [stock_id, seconds_in_bucket, revealed_target,
    revealed_date_id] previous-day targets visible so far. Returns a new DataFrame sorted by
    (stock_id, date_id, seconds_in_bucket)."""
    g = ["stock_id", "date_id"]
    df = day_rows.sort_values(["stock_id", "date_id", "seconds_in_bucket"], kind="mergesort").reset_index(drop=True)
    f = {}
    f["spread"] = df["ask_price"] - df["bid_price"]
    f["book_imb"] = _ratio(df["bid_size"] - df["ask_size"], df["bid_size"] + df["ask_size"])
    f["imb_ratio"] = _ratio(df["imbalance_size"], df["matched_size"])
    f["signed_imb"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    f["signed_imb_ratio"] = _ratio(df["imbalance_size"] * df["imbalance_buy_sell_flag"], df["matched_size"])
    f["ref_minus_wap"] = df["reference_price"] - df["wap"]
    f["far_minus_near"] = df["far_price"] - df["near_price"]
    f["near_minus_ref"] = df["near_price"] - df["reference_price"]
    f["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    f["imb_over_depth"] = _ratio(df["imbalance_size"], df["bid_size"] + df["ask_size"])
    f["wap_minus_mid"] = df["wap"] - f["mid"]

    grp = df.groupby(g, sort=False)
    for c in LAG_COLS:
        gc = grp[c]
        for k in LAGS:
            f[f"{c}_ret_{k}"] = df[c] / gc.shift(k) - 1
    tmp = pd.DataFrame({"stock_id": df["stock_id"], "date_id": df["date_id"],
                        "sir": f["signed_imb_ratio"], "bi": f["book_imb"]})
    tg = tmp.groupby(g, sort=False)
    f["signed_imb_ratio_d1"] = tmp["sir"] - tg["sir"].shift(1)
    f["book_imb_d1"] = tmp["bi"] - tg["bi"].shift(1)
    first_pos = grp.cumcount() == 0
    first_wap = df["wap"].where(first_pos)
    # forward-fill the (possibly NaN) first value within group by position, not skipping NaN
    gid = grp.ngroup()
    first_by_gid = pd.Series(df["wap"].values[first_pos.values], index=gid.values[first_pos.values])
    f["wap_ret_day"] = df["wap"] / gid.map(first_by_gid).values - 1
    del first_wap
    f["flag_change"] = df["imbalance_buy_sell_flag"] - grp["imbalance_buy_sell_flag"].shift(1)

    out = pd.concat([df, pd.DataFrame(f)], axis=1)

    tgrp = out.groupby("time_id", sort=False)
    for c in CS_COLS:
        out[f"{c}_cs"] = out[c] - tgrp[c].transform("mean")
    out["mkt_wap_ret_1"] = tgrp["wap_ret_1"].transform("mean")
    out["mkt_wap_ret_6"] = tgrp["wap_ret_6"].transform("mean")

    if index_weights is not None:
        w = out["stock_id"].map(index_weights).fillna(0.0).astype("float64")
        tmp = pd.DataFrame({"time_id": out["time_id"]})
        for c in IDX_COLS:
            tmp[f"n_{c}"] = out[c].fillna(0) * w
            tmp[f"d_{c}"] = w * out[c].notna()
        sums = tmp.groupby("time_id", sort=False).transform("sum")
        for c in IDX_COLS:
            out[f"idx_{c}"] = sums[f"n_{c}"] / sums[f"d_{c}"]
        for c in IDX_COLS:
            out[f"{c}_rel"] = out[c] - out[f"idx_{c}"]
        out["index_weight"] = w

    if revealed is not None and len(revealed):
        rv = revealed[["stock_id", "seconds_in_bucket", "revealed_target", "revealed_date_id"]].copy()
        rv = rv.sort_values(["stock_id", "revealed_date_id", "seconds_in_bucket"], kind="mergesort")
        rg = rv.groupby(["stock_id", "revealed_date_id"], sort=False)
        cs = rg["revealed_target"].cumsum()
        cn = rv["revealed_target"].notna().astype("int64").groupby([rv["stock_id"], rv["revealed_date_id"]]).cumsum()
        rv["prev_day_target_cummean"] = (cs / cn).where(rv["revealed_target"].notna())
        rv = rv.rename(columns={"revealed_target": "prev_day_target_same_s", "revealed_date_id": "_prev_date"})
        out["_prev_date"] = out["date_id"] - 1
        out = out.merge(rv, on=["stock_id", "_prev_date", "seconds_in_bucket"], how="left").drop(columns="_prev_date")
    else:
        out["prev_day_target_same_s"] = np.nan
        out["prev_day_target_cummean"] = np.nan
    return out


def feature_columns_pd(df):
    return [c for c in df.columns if c not in DROP]


# Synthetic-index weights recovered from train (reports/eda_day2.json); R^2 = 1.0, sum = 1.0.
INDEX_WEIGHTS = {0: 0.004, 1: 0.001, 2: 0.002, 3: 0.006, 4: 0.004, 5: 0.004, 6: 0.002, 7: 0.006, 8: 0.006, 9: 0.002, 10: 0.002, 11: 0.008, 12: 0.006, 13: 0.002, 14: 0.008, 15: 0.006, 16: 0.002, 17: 0.006, 18: 0.004, 19: 0.002, 20: 0.004, 21: 0.001, 22: 0.006, 23: 0.004, 24: 0.002, 25: 0.002, 26: 0.004, 27: 0.002, 28: 0.004, 29: 0.004, 30: 0.001, 31: 0.001, 32: 0.002, 33: 0.002, 34: 0.006, 35: 0.004, 36: 0.004, 37: 0.004, 38: 0.006, 39: 0.002, 40: 0.002, 41: 0.04, 42: 0.002, 43: 0.002, 44: 0.004, 45: 0.04, 46: 0.002, 47: 0.001, 48: 0.006, 49: 0.004, 50: 0.004, 51: 0.006, 52: 0.001, 53: 0.004, 54: 0.004, 55: 0.002, 56: 0.006, 57: 0.004, 58: 0.006, 59: 0.004, 60: 0.006, 61: 0.004, 62: 0.002, 63: 0.001, 64: 0.002, 65: 0.004, 66: 0.002, 67: 0.008, 68: 0.004, 69: 0.004, 70: 0.002, 71: 0.004, 72: 0.006, 73: 0.002, 74: 0.004, 75: 0.004, 76: 0.002, 77: 0.004, 78: 0.004, 79: 0.004, 80: 0.001, 81: 0.002, 82: 0.002, 83: 0.008, 84: 0.02, 85: 0.004, 86: 0.006, 87: 0.002, 88: 0.02, 89: 0.002, 90: 0.002, 91: 0.006, 92: 0.004, 93: 0.002, 94: 0.001, 95: 0.02, 96: 0.006, 97: 0.001, 98: 0.002, 99: 0.004, 100: 0.001, 101: 0.002, 102: 0.006, 103: 0.006, 104: 0.004, 105: 0.006, 106: 0.001, 107: 0.002, 108: 0.004, 109: 0.006, 110: 0.006, 111: 0.001, 112: 0.04, 113: 0.006, 114: 0.002, 115: 0.004, 116: 0.002, 117: 0.002, 118: 0.006, 119: 0.002, 120: 0.002, 121: 0.004, 122: 0.006, 123: 0.006, 124: 0.002, 125: 0.002, 126: 0.008, 127: 0.006, 128: 0.004, 129: 0.002, 130: 0.006, 131: 0.002, 132: 0.004, 133: 0.006, 134: 0.002, 135: 0.004, 136: 0.001, 137: 0.004, 138: 0.002, 139: 0.004, 140: 0.008, 141: 0.006, 142: 0.008, 143: 0.002, 144: 0.004, 145: 0.002, 146: 0.001, 147: 0.004, 148: 0.004, 149: 0.004, 150: 0.006, 151: 0.008, 152: 0.004, 153: 0.001, 154: 0.001, 155: 0.002, 156: 0.006, 157: 0.004, 158: 0.001, 159: 0.002, 160: 0.006, 161: 0.004, 162: 0.006, 163: 0.008, 164: 0.002, 165: 0.002, 166: 0.004, 167: 0.002, 168: 0.04, 169: 0.002, 170: 0.002, 171: 0.004, 172: 0.002, 173: 0.002, 174: 0.006, 175: 0.02, 176: 0.004, 177: 0.002, 178: 0.006, 179: 0.02, 180: 0.001, 181: 0.002, 182: 0.006, 183: 0.004, 184: 0.006, 185: 0.004, 186: 0.004, 187: 0.004, 188: 0.004, 189: 0.002, 190: 0.004, 191: 0.04, 192: 0.002, 193: 0.008, 194: 0.002, 195: 0.004, 196: 0.001, 197: 0.004, 198: 0.006, 199: 0.004}

FEAT_WEIGHTS = INDEX_WEIGHTS if FEATURE_VERSION == "v2" else None

RAW_SCHEMA = ["stock_id", "date_id", "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag",
              "reference_price", "matched_size", "far_price", "near_price", "bid_price", "bid_size",
              "ask_price", "ask_size", "wap", "target", "time_id", "row_id"]

PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=NUM_THREADS, verbose=-1, seed=42)


def zero_sum(stock_ids, preds):
    w = pd.Series(stock_ids).map(INDEX_WEIGHTS).fillna(0.0).to_numpy()
    if w.sum() <= 0:
        return preds
    return preds - (preds * w).sum() / w.sum()


def build_training_matrix(tr):
    """tr: raw rows with train.csv schema (incl. target, time_id). Returns X (float32), y, cols.
    Features are built in date chunks; features depend only on the same date and the previous
    date's targets, so chunking (with the previous date included for reveals) is exact."""
    dates = np.sort(tr["date_id"].unique())
    # Previous-day targets come from the previous date actually present (relabelled as d-1), exactly as
    # serving relabels the most recent revealed date; identical to d-1 when dates are contiguous.
    pred_of = {int(b): int(a) for a, b in zip(dates[:-1], dates[1:])}
    Xs, ys, cols = [], [], None
    for i in range(0, len(dates), CHUNK_DAYS):
        ch = dates[i:i + CHUNK_DAYS]
        part = tr[tr["date_id"].isin(ch)]
        rv_parts = []
        for dd in ch:
            pd_ = pred_of.get(int(dd))
            if pd_ is None:
                continue
            r = tr.loc[tr["date_id"] == pd_, ["stock_id", "seconds_in_bucket", "target"]]
            rv_parts.append(pd.DataFrame({"stock_id": r["stock_id"].to_numpy(),
                                          "seconds_in_bucket": r["seconds_in_bucket"].to_numpy(),
                                          "revealed_target": r["target"].to_numpy(),
                                          "revealed_date_id": np.full(len(r), int(dd) - 1, dtype="int64")}))
        rv = pd.concat(rv_parts, ignore_index=True) if rv_parts else None
        f = build_features_pd(part.drop(columns=["target"]), rv, FEAT_WEIGHTS)
        f = f.merge(part[["row_id", "target"]], on="row_id", how="left")
        f = f[f["target"].notna()]
        if cols is None:
            cols = feature_columns_pd(f)
        Xs.append(f[cols].to_numpy(dtype=np.float32))
        ys.append(f["target"].to_numpy(dtype=np.float64))
        del part, rv, rv_parts, f
        gc.collect()
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    gc.collect()
    return X, y, cols


def fit_lgb(X, y, cols):
    return lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=cols, free_raw_data=True), num_boost_round=TRAIN_ROUNDS)


def load_train():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    tr = tr[(tr["date_id"] >= TRAIN_MIN_DATE) & (tr["date_id"] <= TRAIN_MAX_DATE)].reset_index(drop=True)
    tr["time_id"] = tr["date_id"].astype("int64") * 55 + tr["seconds_in_bucket"].astype("int64") // 10
    return tr[RAW_SCHEMA]


def fit_cb(X, y):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function="MAE", iterations=CB_ITERS, learning_rate=0.1, depth=8,
                          border_count=128, thread_count=NUM_THREADS, random_seed=42, verbose=0,
                          allow_writing_files=False)
    m.fit(X, y)
    return m


def train_model(tr=None, with_cb=False):
    """Returns (lgb_model, cols, cb_model_or_None)."""
    t0 = time.time()
    if tr is None:
        tr = load_train()
    X, y, cols = build_training_matrix(tr)
    print(f"train matrix {X.shape}, features built in {time.time()-t0:.0f}s", flush=True)
    model = fit_lgb(X, y, cols)
    print(f"lgb trained in {time.time()-t0:.0f}s", flush=True)
    cb = None
    if with_cb:
        cb = fit_cb(X, y)
        print(f"catboost trained in {time.time()-t0:.0f}s", flush=True)
    del X, y
    gc.collect()
    return model, cols, cb


def _parse_revealed(revealed_targets, d):
    """Return revealed rows of the most recent date before d, re-labelled as date d-1, with the revealed
    row's own second derived from revealed_time_id (the `seconds_in_bucket` column is the current second)."""
    if revealed_targets is None or not len(revealed_targets):
        return None
    r = revealed_targets.dropna(subset=["stock_id", "revealed_date_id", "revealed_time_id"])
    r = r[r["revealed_date_id"] < d]
    if not len(r):
        return None
    r = r[r["revealed_date_id"] == r["revealed_date_id"].max()]
    true_date = r["revealed_date_id"].astype("int64").to_numpy()
    sec = ((r["revealed_time_id"].astype("int64").to_numpy() - true_date * 55) * 10).astype("int64")
    return pd.DataFrame({
        "stock_id": r["stock_id"].astype("int64").to_numpy(),
        "seconds_in_bucket": sec,
        "revealed_target": r["revealed_target"].astype("float64").to_numpy(),
        "revealed_date_id": np.full(len(r), d - 1, dtype="int64"),
        "_true_date": true_date,
    })


def main(env_module):
    t_start = time.time()
    train_raw, cb_model = None, None
    if MODE == "model":
        train_raw = load_train()
        model, cols, cb_model = train_model(train_raw, with_cb=CB_WEIGHT > 0)
        if REFIT_EVERY <= 0:
            train_raw = None
            gc.collect()
    env = env_module.make_env()
    day_hist, revealed_hist = [], []
    cur_date = None
    test_days = {}      # true date_id -> raw rows of that test day (train.csv schema, target NaN)
    test_targets = []   # revealed target rows: [stock_id, date_id(true), seconds_in_bucket, target]
    n_test_days, n_new_days, n_refits = 0, 0, 0
    max_train = int(train_raw["date_id"].max()) if train_raw is not None else TRAIN_MAX_DATE
    n_iter, t0 = 0, time.time()
    for test, revealed_targets, sample_prediction in env.iter_test():
        if MODE == "zeros":
            sample_prediction["target"] = 0.0
            env.predict(sample_prediction)
            n_iter += 1
            continue
        test = test.drop(columns=[c for c in ("currently_scored",) if c in test.columns]).copy()
        # The live API's `test` frame lacks time_id; always derive it.
        test["time_id"] = test["date_id"].astype("int64") * 55 + test["seconds_in_bucket"].astype("int64") // 10
        d = int(test["date_id"].iloc[0])
        if d != cur_date:
            if cur_date is not None and day_hist:
                test_days[cur_date] = pd.concat(day_hist, ignore_index=True)
            cur_date, day_hist, revealed_hist = d, [], []
            n_test_days += 1
            if d > max_train:
                n_new_days += 1
            # Online refit at the start of a test day: data with fully revealed targets only.
            if (REFIT_EVERY > 0 and d > max_train and n_new_days > 1 and (n_new_days - 1) % REFIT_EVERY == 0
                    and time.time() - t_start < REFIT_TIME_BUDGET):
                aug = build_refit_frame(train_raw, test_days, test_targets)
                if aug is not None:
                    t1 = time.time()
                    model, cols, _ = train_model(aug)
                    n_refits += 1
                    print(f"refit {n_refits} at date {d}: {len(aug)} rows, {time.time()-t1:.0f}s", flush=True)
                    del aug
                    gc.collect()
        day_hist.append(test)
        r = _parse_revealed(revealed_targets, d)
        if r is not None:
            revealed_hist.append(r.drop(columns="_true_date"))
            if REFIT_EVERY > 0:
                test_targets.append(pd.DataFrame({"stock_id": r["stock_id"].to_numpy(), "date_id": r["_true_date"].to_numpy(),
                                                  "seconds_in_bucket": r["seconds_in_bucket"].to_numpy(),
                                                  "target": r["revealed_target"].to_numpy()}))
        day = pd.concat(day_hist, ignore_index=True)
        rv = (pd.concat(revealed_hist, ignore_index=True)
              .drop_duplicates(["stock_id", "revealed_date_id", "seconds_in_bucket"], keep="last")
              if revealed_hist else None)
        feats = build_features_pd(day, rv, FEAT_WEIGHTS)
        now = feats[feats["time_id"] == test["time_id"].iloc[0]]
        Xn = now[cols].to_numpy(dtype=np.float32)
        p = model.predict(Xn, num_threads=NUM_THREADS)
        if cb_model is not None:
            p = (1 - CB_WEIGHT) * p + CB_WEIGHT * cb_model.predict(Xn)
        p = zero_sum(now["stock_id"].to_numpy(), p)
        pred = pd.Series(p, index=now["row_id"].to_numpy())
        sample_prediction["target"] = sample_prediction["row_id"].map(pred).fillna(0.0).astype("float64").to_numpy()
        env.predict(sample_prediction)
        n_iter += 1
        if n_iter % 550 == 0:
            print(f"iter {n_iter}, date {d}, {time.time()-t0:.0f}s", flush=True)
    print(f"done: {n_iter} iterations, {n_test_days} test days ({n_new_days} new), {n_refits} refits in {time.time()-t0:.0f}s", flush=True)


def build_refit_frame(train_raw, test_days, test_targets):
    """Train rows + completed new test days except the most recently completed one (its targets are only
    being revealed during the current day). With contiguous dates this is exactly 'dates <= d-2'.
    Days overlapping train (the API replays the last train days first) are skipped. Called at the start
    of a new day, before that day is added to test_days."""
    if not test_targets:
        return None
    tt = pd.concat(test_targets, ignore_index=True).drop_duplicates(
        ["stock_id", "date_id", "seconds_in_bucket"], keep="last")
    max_train = int(train_raw["date_id"].max())
    served = sorted(test_days)
    keep = [k for k in served[:-1] if k > max_train]
    have = set(tt["date_id"].unique().tolist())
    missing = [k for k in keep if k not in have]
    if missing:
        print(f"warning: no revealed targets for days {missing}; excluded from refit", flush=True)
        keep = [k for k in keep if k in have]
    if not keep:
        return None
    raw = pd.concat([test_days[k] for k in keep], ignore_index=True)
    raw = raw.merge(tt, on=["stock_id", "date_id", "seconds_in_bucket"], how="left")
    for c in RAW_SCHEMA:
        if c not in raw.columns:
            raw[c] = np.nan
    return pd.concat([train_raw, raw[RAW_SCHEMA]], ignore_index=True)


if __name__ == "__main__":
    import optiver2023  # provided by the competition (Kaggle) or by the local replay harness

    main(optiver2023)
