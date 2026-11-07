from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray


def expanding_folds(
    n: int, initial_train: int, test_size: int, step: int, embargo: int = 0
) -> list[Fold]:
    folds = []
    test_start = initial_train
    while test_start + test_size <= n:
        train_end = max(0, test_start - embargo)
        folds.append(Fold(np.arange(train_end), np.arange(test_start, test_start + test_size)))
        test_start += step
    if not folds:
        raise ValueError("not enough observations for the requested validation design")
    return folds
