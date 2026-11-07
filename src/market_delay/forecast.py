from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .validation import expanding_folds


@dataclass(frozen=True)
class ForecastResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def _models(seed: int) -> dict[str, object]:
    return {
        "ridge": Pipeline(
            [("impute", SimpleImputer()), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80,
                        max_depth=6,
                        min_samples_leaf=10,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def rolling_forecast(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    initial_train: int,
    test_size: int,
    step: int,
    embargo: int,
    seed: int,
) -> ForecastResult:
    rows = []
    clean_all = frame.dropna(subset=["target_return"])
    for market_name, clean in clean_all.groupby("market"):
        clean = clean.sort_values("session_date").reset_index(drop=True)
        folds = expanding_folds(len(clean), initial_train, test_size, step, embargo)
        for set_name, columns in feature_sets.items():
            for model_name, model in _models(seed).items():
                for fold_id, fold in enumerate(folds):
                    model.fit(
                        clean.loc[fold.train, columns], clean.loc[fold.train, "target_return"]
                    )
                    pred = model.predict(clean.loc[fold.test, columns])
                    for idx, value in zip(fold.test, pred, strict=True):
                        rows.append(
                            {
                                "row_id": int(idx),
                                "session_date": clean.loc[idx, "session_date"],
                                "market": market_name,
                                "feature_set": set_name,
                                "model": model_name,
                                "fold": fold_id,
                                "actual": clean.loc[idx, "target_return"],
                                "prediction": float(value),
                            }
                        )
    predictions = pd.DataFrame(rows)
    metrics = []
    for keys, group in predictions.groupby(["market", "feature_set", "model"]):
        market, feature_set, model = keys
        metrics.append(
            {
                "market": market,
                "feature_set": feature_set,
                "model": model,
                "mae": mean_absolute_error(group["actual"], group["prediction"]),
                "rmse": mean_squared_error(group["actual"], group["prediction"]) ** 0.5,
                "directional_accuracy": np.mean(
                    np.sign(group["actual"]) == np.sign(group["prediction"])
                ),
                "n": len(group),
            }
        )
    return ForecastResult(predictions, pd.DataFrame(metrics))
