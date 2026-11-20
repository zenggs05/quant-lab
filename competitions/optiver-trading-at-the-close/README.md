# Optiver – Trading at the Close: late-submission study

A leakage-tested pipeline for the Kaggle competition [Optiver – Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close), built after the competition ended and scored through late submissions.

- **Private MAE 5.4571** (late submission; all-zeros probe 5.6036, static single LightGBM 5.4750).
- **Would have placed 31st of 3,225 teams** on the final private leaderboard (30th: 5.4567, 31st: 5.4575). This is a late submission, so it is not officially ranked.
- Full write-up: [reports/writeup.md](reports/writeup.md). It covers the validation design, the tests that enforce it, feature families, results across CV / holdout / leaderboard, failed ideas, and engineering lessons.

## What is here
| Path | Contents |
|---|---|
| `src/` | Research code (polars): data conversion, EDA and index-weight recovery, date-based CV, feature engineering, experiments, one-shot holdout evaluation, blend analysis |
| `kaggle/` | Kaggle notebook source (pandas port for the 2023 Python 3.10 image), notebook builder, local replay of the time-series API |
| `kaggle/build/` | The exact single-cell notebooks submitted (`submission_model.py`, `submission_zeros.py`) |
| `tests/` | 12 tests: batch-vs-streaming leakage check with a mutation test, polars/pandas feature parity, end-to-end streaming vs offline predictions, online-refit protocol |
| `reports/` | Write-up, experiment log, notes, JSON results |

## Reproduce
Competition data is not included (competition rules); download it with the Kaggle CLI after accepting the rules:
```
kaggle competitions download optiver-trading-at-the-close -p data/raw
unzip data/raw/optiver-trading-at-the-close.zip -d data/raw/extracted
python src/convert_raw.py data/raw/extracted data/processed
python src/eda.py                 # recovers index weights -> reports/eda_day2.json
python -m pytest tests
python kaggle/build_notebook.py   # builds kaggle/build/*.py from kaggle/submission.py
```
Pinned package versions: `requirements.lock.txt` (Python 3.13 research environment). The Kaggle notebook must run on the pinned 2023 image (Python 3.10).
