# Day 2 notes (2026-11-19)

## API serving semantics (from example_test_files, verified locally)
- `time_id = date_id * 55 + seconds_in_bucket / 10`.
- In the example files, `revealed_targets` at (date d, second s) carries the targets of (d-1, s). These match train targets exactly (33,000 rows, max abs diff 0).
- Our mock is conservative: at (d, s) it reveals only targets of d-1 with seconds_in_bucket <= s. This is leak-free under either reading (per-second reveal or full-day reveal at s=0).

## EDA
- far_price / near_price are 100% null before s=300. After s=300 near_price is complete and far_price is about 2% null.
- The zero-prediction MAE depends strongly on s: about 8.0 at s=0, 9.3–9.5 at s=240–280, and about 5.0 near s=480.
- Synthetic-index weights are recovered exactly: regressing the implied index return on per-stock 60-second WAP returns gives R²=1.0 and weights summing to 1.0. Weights take 7 distinct values between 0.001 and 0.04. Saved in `reports/eda_day2.json`.

## Leakage tests (`tests/test_leakage.py`)
- Folds are ordered and separated by a 1-day embargo.
- Batch features equal features streamed through the mock API, for every feature column.
- Mutation check: injecting `wap.shift(-1)` makes the test fail. The test catches future leakage.

## Baselines (selection folds f0–f3; holdout 400–480 untouched)
| model | f0 | f1 | f2 | f3 | mean |
|---|---|---|---|---|---|
| zeros | 5.8804 | 6.4262 | 6.4222 | 6.4755 | 6.3011 |
| per-stock median | 5.8848 | 6.4285 | 6.4225 | 6.4764 | 6.3030 |
| LGBM raw (13 features) | 5.8194 | 6.3445 | 6.3257 | 6.3798 | 6.2174 |
| LGBM v1 (61 features) | 5.7700 | 6.2933 | 6.2737 | 6.3288 | 6.1665 |
| LGBM v1 + index zero-sum | 5.7649 | 6.2867 | 6.2721 | 6.3253 | 6.1622 |

- The ranking is the same on all 4 folds. Feature set v1 gains about 0.051 over raw. Zero-sum post-processing gains about 0.004, consistently.
- Gain over zeros is about 0.139 on average, in line with the public reference (about 0.14 on the last 20% of dates).
- Fixed 800 rounds, no early stopping. In-fold best iterations were 289–712 and the curves are flat after that, so the fixed round count costs at most about 0.005.
- Top gain features: wap_minus_mid, seconds_in_bucket, matched_size, ref_minus_wap, wap_ret_day_cs, signed_imb_ratio_cs, spread, prev_day_target_cummean.

## Next
- Online daily refit using revealed targets. Richer cross-sectional and index-relative features (stock return minus weighted index return). CatBoost or an NN for diversity.
- Day 6: build the Kaggle notebook (Python 3.10 pinned env) and submit an all-zeros run to confirm scoring semantics.
