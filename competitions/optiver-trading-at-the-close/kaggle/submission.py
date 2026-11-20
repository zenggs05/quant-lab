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

MODE = os.environ.get("OPTIVER_MODE", "model")  # "model" or "zeros" (zeros = API/runtime/scoring probe)
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
# FEATURES_PD_BEGIN
# FEATURES_PD_END

# Synthetic-index weights recovered from train (reports/eda_day2.json); R^2 = 1.0, sum = 1.0.
# INDEX_WEIGHTS_BEGIN
INDEX_WEIGHTS = None
# INDEX_WEIGHTS_END

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
