# Week 1 retrospective (2026-11-19)

## What was built
- Data: downloaded via the Kaggle CLI, with SHA256 recorded at each stage (zip, extracted files, parquet). 88 missing targets stored as the literal string "nan" were normalised.
- Validation: four rolling-origin folds (validation blocks 240–399) with a 1-day embargo. Holdout 400–480 was evaluated exactly once, with configs declared in advance.
- Leakage guards (all in `tests/`, 6 tests):
  1. Batch features equal features streamed through a local API replay. A mutation test that injects `wap.shift(-1)` is caught.
  2. The pandas port used on Kaggle matches the polars reference column for column, checked under both the local env and a Python 3.10 / pandas 2.0.3 / LightGBM 3.3.2 env that mimics the 2023 Kaggle image.
  3. End-to-end: notebook predictions streamed through the replay equal offline batch predictions (max diff < 1e-6) under both readings of the reveal semantics (per-second vs full-day at s=0).
- Robustness: the notebook handles gaps in date_id numbering (verified), derives the revealed row's own second from `revealed_time_id`, and builds training features in date chunks. The chunked matrix is the same row set as the unchunked one.
- Synthetic-index weights recovered exactly (R²=1.0, sum=1). They drive the zero-sum post-processing.

## Results (MAE, lower is better)
| config | selection CV mean (f0–f3) | holdout 400–480 (once) |
|---|---|---|
| zeros | 6.3011 | 5.9936 |
| LGBM raw | 6.2174 | — |
| LGBM v1 (no zero-sum) | 6.1665 | 5.8581 |
| LGBM v1 + zero-sum | 6.1622 | **5.8546** |
| LGBM v1 + zero-sum + refit every 10d | f2/f3: −0.0052/−0.0033 vs static | **5.8478** |

- Gain over zeros: CV 0.139, holdout 0.139. Zero-sum gain: CV 0.004, holdout 0.0036. Online refit gain: CV 0.003–0.005, holdout 0.0067. The direction and size of every effect carried over from CV to holdout.
- Hardest window: s=240–299, where the zero baseline is about 9.0. The largest relative gain is at s=540 (6.04 → 5.44).

## Failed / neutral ideas
- **v2 index-relative features** (stock return minus weighted index return, for six signals): +0.00003 mean over 4 folds, 2 better and 2 worse, i.e. noise. The cross-sectional de-meaning in v1 already captures this. Dropped.
- **Per-stock median baseline**: no better than zeros.

## Runtime budget (local, 4 threads, Python 3.10 env)
- Feature build 51 s, training 449 s, inference about 0.08 s per time step, peak RSS 4.4 GB. Well inside 9 h / 30 GB.

## Leaderboard (late submissions, private = full forecasting-phase test set)
| submission | private MAE | gain vs zeros |
|---|---|---|
| zeros probe | 5.6036 | — |
| LGBM v1 + zero-sum (first attempt) | failed: `KeyError: 'time_id'` | — |
| LGBM v1 + zero-sum (fixed) | **5.4750** | 0.129 |

- CV → holdout → LB gain vs zeros: 0.139 → 0.139 → 0.129. The local validation predicted the live improvement to within about 0.01.
- Reference: 1st place 5.4030 on the same test set.
- Failure lesson: the live API `test` frame has no `time_id` column, while the example files do. The local replay was built from the example files, so it missed this. The fix derives `time_id` from `date_id` and `seconds_in_bucket`. The tests now cover both formats, plus an API-format stress test (shuffled rows and columns, extra columns, float id columns).

## Open, to confirm with the first late submissions
- What the late private score covers, and whether public is 0 (the zeros probe answers this).
- The LightGBM version in the pinned 2023 image (printed by the owner).

## Week 2 plan
1. Online refit inside the notebook, refitting every 10 days on train plus revealed test targets. Gain confirmed on CV and holdout.
2. Diversity: CatBoost and/or a small GRU over the intraday sequence. Blend weights chosen on folds only.
3. Compare CV with the LB, then start the public write-up draft (validation design first).
