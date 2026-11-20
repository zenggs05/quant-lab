"""Date-based validation splits for Optiver Trading at the Close.

Design:
- Splits are by date_id only. Rows of the same date share the synthetic-index move,
  so random row splits leak cross-sectional information.
- Model selection uses rolling-origin folds inside date_id 0..399 with a 1-day embargo.
- date_id 400..480 is a holdout, evaluated sparingly (frozen protocol, not tuned on).
"""
from __future__ import annotations

from dataclasses import dataclass

EMBARGO_DAYS = 1
HOLDOUT_START = 400
LAST_DATE = 480


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: int
    train_end: int  # inclusive
    valid_start: int
    valid_end: int  # inclusive

    def __post_init__(self) -> None:
        assert self.train_end + EMBARGO_DAYS < self.valid_start, "embargo violated"
        assert self.train_start <= self.train_end < self.valid_start <= self.valid_end


def selection_folds() -> list[Fold]:
    """Four expanding-window folds, each validated on a 40-day block before the holdout."""
    folds = []
    for k, vs in enumerate(range(240, HOLDOUT_START, 40)):
        folds.append(Fold(f"f{k}", 0, vs - 1 - EMBARGO_DAYS, vs, vs + 39))
    return folds


def holdout_fold() -> Fold:
    return Fold("holdout", 0, HOLDOUT_START - 1 - EMBARGO_DAYS, HOLDOUT_START, LAST_DATE)


if __name__ == "__main__":
    for f in selection_folds() + [holdout_fold()]:
        print(f)
