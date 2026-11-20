# quant-lab

Quantitative research and trading-competition work by [zenggs05](https://github.com/zenggs05).
Each project is self-contained in its own directory, with its own README, code, tests, and write-up.
The emphasis throughout is on validation you can trust: time-based splits, leakage tests, and checking how local estimates carry over to out-of-sample results.

## Competitions
| Project | Status | Result | Write-up |
|---|---|---|---|
| [Optiver – Trading at the Close](competitions/optiver-trading-at-the-close) (Kaggle, late submission) | Done | Private MAE 5.4571; would have placed 31st of 3,225 teams (not officially ranked) | [writeup](competitions/optiver-trading-at-the-close/reports/writeup.md) |

## Research
| Project | Description |
|---|---|
| [QuantForge](https://github.com/zenggs05/QuantForge) (separate repository) | Pre-registered study of 1-minute BTCUSDT return prediction: strict time semantics, leakage control, Clark–West / Diebold–Mariano tests, HAC inference, power analysis. Preprint DOI [10.5281/zenodo.24330979](https://doi.org/10.5281/zenodo.24330979) |

New research topics will be added under `research/`.

## Data
No competition or market data is stored in this repository. Each project's README explains how to download its data under the data provider's terms.
