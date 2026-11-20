# Optiver – Trading at the Close: a late-submission study where CV, holdout and leaderboard agreed

*AI-assistance disclosure: all design decisions, validation protocol and final sign-off are mine; Claude wrote most of the code and the first draft of this post, which I edited.*

## TL;DR
- As a late submission this is not officially ranked, but 5.4571 would have placed **31st of 3,225 teams** on the final private leaderboard (30th scored 5.4567, 31st 5.4575).
- A compact pipeline — 61 hand-built features, LightGBM refit every 10 test days blended 0.7/0.3 with a one-shot CatBoost, plus index-weighted zero-sum post-processing — scores **5.4571 private MAE** as a late submission. A static single LightGBM scores 5.4750, an all-zeros probe 5.6036, and 1st place 5.4030. The gap from zeros to 1st spans 0.2006, so this submission closes about 73% of it.
- Local validation predicted the leaderboard well. For the one configuration measured at every stage — static LightGBM + zero-sum — gain over zeros was **0.140 in CV, 0.139 on a one-shot holdout, and 0.1286 on the leaderboard**. Every change that helped in CV also helped on the holdout; every change that was also submitted helped on the leaderboard. Nothing flipped direction.
- The leaderboard's main lesson was about the API, not the model. The first model submission failed after training because the example files differ from the live API in one undocumented way.

## 1. Problem
Each row is a 10-second snapshot (seconds 0–540) of one of 200 Nasdaq stocks during the closing auction. The target is the stock's 60-second-ahead weighted-average price (WAP, roughly the order-size-weighted mid-price) minus that of a synthetic index built from the same 200 stocks, in basis points (1 bp = 0.01%). The metric is MAE. Predictions are made online through a time-series API that serves one time step at a time and also streams the previous day's targets. In the example files shipped with the competition, the target for (day d−1, second s) is revealed at (day d, second s); the documentation can also be read as revealing all of day d−1 at the start of day d.

## 2. Validation design
**Splits.** All splits are by `date_id`. Rows of one date share the index move, so random row splits leak cross-sectional information.
- Model selection uses four expanding-window folds. Validation blocks are 240–279, 280–319, 320–359 and 360–399, with a 1-day embargo before each (we drop the trading day immediately before each validation block so that overnight information cannot leak in through revealed-targets features).
- The holdout is date_id 400–480. Three configurations (zeros, static LightGBM, LightGBM refit every 10 days) were written into the evaluation script's docstring before it was run, evaluated exactly once, and never retuned; later model choices (the CatBoost blend) used the folds only. This is an honour-system seal — the project was not under version control at the time, so there is no commit timestamp to prove it.
- The 1-day embargo is strictly stronger than needed (the 60-second target is intraday-only), and is kept as a safety margin for the revealed-targets features that touch d−1.

**What a feature may see.** A feature for (stock, date d, second s) may use (a) rows of date d at seconds ≤ s, for any stock, and (b) targets of date d−1 at seconds ≤ s. Rule (b) is leak-free under either reading of the reveal schedule above, and the end-to-end test is run under both readings.

**Tests that enforce it (`tests/`, 12 tests total).**
1. **Batch = streaming.** Features computed on a whole day at once equal features streamed step by step through a local replay of the API, column by column. A mutation test that injects `wap.shift(-1)` makes this test fail, so it can see look-ahead.
2. **Cross-library equality.** Features are developed in polars offline for speed, then reimplemented in pandas for the Kaggle 2023 image. A test asserts the two implementations agree to 1e-9 on every column of a 50 000-row sample, including days with missing prices and missing targets.
3. **End-to-end.** Notebook predictions streamed through the replay equal offline batch predictions to < 1e-6. Checked under both plausible reveal semantics (per-second and full-day), with and without `time_id` in the served frame, and under a stress test (shuffled rows/columns, extra columns, float id columns).
4. **Online refit protocol.** Each in-notebook refit must train on exactly the row set an offline run would use: original training data plus any test day whose targets have been fully revealed by the API, excluding the most recently served day. Two regression tests cover this; both were confirmed to fail on the pre-fix code described in Section 6.

## 3. Features and post-processing
**61 columns:** the 13 raw inputs (including `stock_id` and `seconds_in_bucket`) plus 48 derived features in four families:
- **Book and auction state (10):** spread, book imbalance, imbalance-to-matched ratio, signed imbalance (size × buy/sell flag), signed imbalance ratio, reference − WAP, far − near, near − reference, imbalance over depth, WAP − mid.
- **Intraday dynamics (30):** returns over 1/2/3/6 steps for WAP, reference price, imbalance size, matched size, bid size, ask size; first differences of signed imbalance ratio and book imbalance; WAP return since the first second of the day; flag change.
- **Cross-section at the same time step (8):** de-meaned versions of six key signals, plus the market-average 1- and 6-step WAP returns.
- **Revealed previous-day targets (2):** the previous day's target at the same second, and its running mean up to that second.

**Synthetic-index zero-sum.** The target construction `target_i = 1e4 · (r_i − r_index)` means the implied index return is `r_index = r_i − target_i/1e4` for every stock. Pooling across stocks and times, we OLS-regress that implied series on per-stock 60-second WAP returns and recover the index weights with R² = 1.0, weights summing to exactly 1 and taking 7 distinct values — this confirms we have the right functional form, it does not merely restate the construction. At serving time we subtract the weighted-mean prediction from every stock's prediction at each time step, producing a zero-sum prediction under the recovered weights. CV gain 0.0043 (same-run comparison, all 4 folds improved), holdout gain 0.0035.

## 4. Results
| | CV mean (4 folds) | Holdout 400–480 | Private LB |
|---|---|---|---|
| zeros | 6.3011 | 5.9936 | 5.6036 |
| LGBM raw (13 input features) | 6.2174 | – | – |
| LGBM v1 (61 features) | 6.1665 | 5.8581 | – |
| LGBM v1 + zero-sum | 6.1611 | 5.8546 | **5.4750** |
| CatBoost v1 + zero-sum (alone) | 6.1659 | – | – |
| LGBM + refit every 10d + zero-sum | 6.1575 | 5.8478 | – |
| 0.7 × refit LGBM + 0.3 × CatBoost + zero-sum | **6.1546** | – | **5.4571** |

*Legend.* LGBM v1 is one LightGBM boosted on all 61 features. Dashes mean a configuration was evaluated locally but not submitted: the competition caps submissions at 5/day, and we traded submissions for being decisive at the end of each week.

**Per-effect deltas (negative = better).**

| Effect (baseline → +effect) | CV | Holdout | LB |
|---|---|---|---|
| Features v1 (zeros → LGBM v1) | −0.1346 | −0.1355 | – |
| Zero-sum (LGBM v1 → + zs) | −0.0043 | −0.0035 | – |
| Refit (static zs → refit zs) | −0.0036 | −0.0068 | – |
| Refit + CatBoost blend (static zs → final) | −0.0065 | – | −0.0179 |

*Each delta compares two configurations from the same run: rows one and two from week 1 (800 boosting rounds, where static LGBM + zero-sum scored 6.1622), rows three and four from week 2 (600 rounds, where it scored 6.1611). The results table above lists the week-2 value for that row.*

**Blend weight selection.** The CatBoost weight was chosen by leave-one-fold-out: for each fold, the weight is picked on the other three folds and applied to the held-out fold. It came out at 0.3 on every fold (CV grid from 0.0 to 1.0 in 0.1 steps). No holdout tuning.

**Per-fold MAE for the two submitted configurations, for a sense of variance:**

| | f0 | f1 | f2 | f3 | mean |
|---|---|---|---|---|---|
| LGBM v1 + zero-sum (baseline) | 5.7629 | 6.2854 | 6.2712 | 6.3250 | 6.1611 |
| 0.7 refit + 0.3 CatBoost + zs | 5.7582 | 6.2776 | 6.2628 | 6.3197 | 6.1546 |
| per-fold gain | 0.0047 | 0.0078 | 0.0084 | 0.0053 | 0.0065 (paired SE 0.0009, n = 4) |

## 5. What did not work
- **Index-relative features** (per-stock signal minus its index-weighted mean, over six signals): CV change was +0.00003 MAE — statistical noise against a 6.17 baseline. The within-time cross-sectional de-meaning in v1 already captures the same information.
- **Per-stock median baseline:** no better than predicting zeros.

## 6. Engineering lessons
- **Example API ≠ live API.** The example `test` frame has a `time_id` column; the live one does not. Our local replay mirrored the examples, so the first model submission crashed with `KeyError: 'time_id'` after training. The fix was to derive `time_id` from `date_id` and `seconds_in_bucket` everywhere (identical formula during training and during serving), and to stress-test the notebook against missing or extra columns, shuffled rows and columns, and float id dtypes.
- **Pin the 2023 environment.** The API binary is a CPython 3.10 extension; a notebook on the latest image cannot import it. In the Kaggle notebook, choose "Pin to original environment".
- **Zeros probe first.** It confirms late submissions are scored on the hidden test set, and anchors the leaderboard: 5.6036 is the reference every other number is compared against.
- **Independent review found two refit-protocol drifts** before the second submission went out. Both were caught by regression tests that fail on the pre-fix code:
  1. The API first replays the last 3 training days before new days begin. Counting them as "new days" shifted every refit 3 days earlier than CV did. Fix: a `n_new_days` counter that only increments when `date_id > TRAIN_MAX_DATE`.
  2. "Dates ≤ d−2" was computed as `d − 2` on the live `date_id` numbers. If the hidden dates have gaps, the most recent served day could enter a refit before its targets had been revealed, and training-time "previous-day target" features linked to literal `d−1` instead of the previous day actually present. Fix: refits work from the order days were served (exclude the most recent); training features link each date to its actual predecessor in the data, matching how `_parse_revealed` relabels at serving.

**Late-submission context.** Four late submissions in total: (1) zeros probe → 5.6036; (2) static LGBM + zero-sum → failed with `KeyError: 'time_id'`; (3) the same configuration, fixed → 5.4750; (4) refit + CatBoost blend → 5.4571. Each submission came from a configuration frozen before the submission, not from choosing the best of several; the leaderboard was not used for tuning. The zeros baseline differs between holdout (5.9936) and private LB (5.6036), a regime effect of 0.39 bp that is larger than any model gain — which is why the write-up reports effect *signs and sizes* rather than absolute levels across stages.

## 7. Reproducibility
Data hashes (zip, extracted files, parquet), the experiment log (`reports/experiments.csv`), and all 12 tests are in the repository. The notebook trains from scratch inside Kaggle, so no model artifacts need to be uploaded. Local estimates at 4 threads: initial LightGBM ≈ 3.5 min, CatBoost ≈ 19 min, ≈ 7 min per refit.
