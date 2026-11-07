from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

from .config import Settings
from .inference import benjamini_hochberg, block_bootstrap_delay, granger_test, price_delay
from .report import write_frame
from .validation import expanding_folds


def _feature_frame(raw: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    frame = raw.sort_values(["market", "timestamp"]).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    groups = frame.groupby("market", group_keys=False)
    for lag in settings.features.return_lags:
        frame[f"return_lag_{lag}"] = groups["return"].shift(lag)
    for signal in [
        "equity_uncertainty",
        "us_policy_uncertainty",
        "uk_policy_uncertainty",
        "vix",
        "gbp_usd",
    ]:
        frame[f"{signal}_change"] = groups[signal].transform(lambda x: np.log1p(x).diff())
        for lag in settings.features.sentiment_lags:
            frame[f"{signal}_lag_{lag}"] = groups[f"{signal}_change"].shift(lag)
    for window in settings.features.volatility_windows:
        frame[f"volatility_{window}"] = groups["return"].transform(
            lambda x, w=window: x.shift(1).rolling(w).std()
        )
    frame["target_return"] = groups["return"].shift(-1)
    frame["crisis"] = frame["timestamp"].between("2020-02-20", "2020-06-30").astype(int)
    return frame


def _diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, group in frame.groupby("market"):
        values = group["return"].dropna()
        adf = adfuller(values, autolag="BIC")
        k = kpss(values, regression="c", nlags="auto")
        lb = acorr_ljungbox(values, lags=[10], return_df=True).iloc[0]
        arch = het_arch(values, nlags=10)
        rows.append(
            {
                "market": market,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(),
                "skew": stats.skew(values),
                "excess_kurtosis": stats.kurtosis(values),
                "adf_stat": adf[0],
                "adf_p": adf[1],
                "kpss_stat": k[0],
                "kpss_p": k[1],
                "ljung_box_10": lb["lb_stat"],
                "ljung_box_p": lb["lb_pvalue"],
                "arch_lm": arch[0],
                "arch_p": arch[1],
            }
        )
    return pd.DataFrame(rows)


def _forecast(frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = [c for c in frame if c.startswith("return_lag_") or c.startswith("volatility_")]
    signal = [c for c in frame if "_lag_" in c and c not in price]
    feature_sets = {"price_only": price, "news_and_controls": price + signal}
    models = {
        "ridge": lambda: Pipeline(
            [("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": lambda: Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=7,
                        min_samples_leaf=10,
                        random_state=settings.project.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": lambda: Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        max_depth=4,
                        learning_rate=0.04,
                        l2_regularization=1.0,
                        random_state=settings.project.seed,
                    ),
                ),
            ]
        ),
    }
    predictions = []
    for market, group in frame.dropna(subset=["target_return"]).groupby("market"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        folds = expanding_folds(
            len(group),
            settings.validation.initial_train_size,
            settings.validation.test_size,
            settings.validation.step_size,
            settings.validation.embargo,
        )
        for feature_name, columns in feature_sets.items():
            for model_name, factory in models.items():
                for fold_id, fold in enumerate(folds):
                    model = factory()
                    model.fit(
                        group.loc[fold.train, columns], group.loc[fold.train, "target_return"]
                    )
                    pred = model.predict(group.loc[fold.test, columns])
                    for idx, value in zip(fold.test, pred, strict=True):
                        predictions.append(
                            {
                                "market": market,
                                "timestamp": group.loc[idx, "timestamp"],
                                "feature_set": feature_name,
                                "model": model_name,
                                "fold": fold_id,
                                "actual": group.loc[idx, "target_return"],
                                "prediction": float(value),
                            }
                        )
    pred = pd.DataFrame(predictions)
    metrics = []
    for keys, g in pred.groupby(["market", "model", "feature_set"]):
        market, model, feature_set = keys
        hit = np.sign(g["actual"]) == np.sign(g["prediction"])
        position = np.sign(g["prediction"])
        turnover = position.diff().abs().fillna(position.abs())
        net = position * g["actual"] - turnover * settings.analysis.transaction_cost_bps / 10000
        metrics.append(
            {
                "market": market,
                "model": model,
                "feature_set": feature_set,
                "n": len(g),
                "mae": mean_absolute_error(g["actual"], g["prediction"]),
                "rmse": mean_squared_error(g["actual"], g["prediction"]) ** 0.5,
                "directional_accuracy": hit.mean(),
                "annualised_net_return": net.mean() * 252,
                "annualised_net_volatility": net.std() * np.sqrt(252),
                "net_sharpe": net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan,
                "turnover": turnover.mean(),
            }
        )
    return pred, pd.DataFrame(metrics)


def run_empirical_analysis(settings: Settings) -> Path:
    out = settings.project.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(settings.data.market_file)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    features = _feature_frame(raw, settings)
    write_frame(features, out / "empirical_features.csv")
    write_frame(_diagnostics(features), out / "diagnostics.csv")

    inference_rows = []
    for market, group in features.groupby("market"):
        signal = -group["equity_uncertainty_change"]
        granger = granger_test(group["return"], signal, settings.analysis.max_lag)
        delay = price_delay(group["return"], signal, settings.analysis.max_lag)
        ci = block_bootstrap_delay(
            group["return"],
            signal,
            settings.analysis.max_lag,
            settings.analysis.bootstrap_samples,
            settings.analysis.bootstrap_block,
            settings.project.seed,
        )
        for test in granger:
            inference_rows.append(
                {
                    "market": market,
                    **test.__dict__,
                    **delay,
                    "delay_ci_low": ci[0],
                    "delay_ci_high": ci[1],
                }
            )
    inference = pd.DataFrame(inference_rows)
    adjusted = benjamini_hochberg(inference["p_value"].to_numpy(), settings.analysis.fdr_alpha)
    inference[["p_adjusted", "reject_fdr"]] = adjusted[["p_adjusted", "reject"]].to_numpy()
    write_frame(inference, out / "granger_and_delay.csv")

    pred, metrics = _forecast(features, settings)
    write_frame(pred, out / "predictions.csv")
    write_frame(metrics, out / "forecast_metrics.csv")
    # Paired loss comparisons with HAC standard errors.
    dm_rows = []
    for (market, model), group in pred.groupby(["market", "model"]):
        wide = group.pivot_table(
            index="timestamp",
            columns="feature_set",
            values=["actual", "prediction"],
            aggfunc="first",
        )
        actual = wide[("actual", "price_only")]
        loss_price = (actual - wide[("prediction", "price_only")]) ** 2
        loss_news = (actual - wide[("prediction", "news_and_controls")]) ** 2
        differential = loss_price - loss_news
        result = sm.OLS(differential.to_numpy(), np.ones((len(differential), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )
        dm_rows.append(
            {
                "market": market,
                "model": model,
                "mean_loss_improvement": differential.mean(),
                "hac_t": result.tvalues[0],
                "hac_p": result.pvalues[0],
            }
        )
    write_frame(pd.DataFrame(dm_rows), out / "forecast_loss_tests.csv")

    # Event paths and regime comparisons are descriptive robustness analyses.
    events = pd.DataFrame(
        [
            ("COVID-19 sell-off", "2020-02-24"),
            ("WHO pandemic declaration", "2020-03-11"),
            ("Brexit transition end", "2020-12-31"),
            ("Russia invades Ukraine", "2022-02-24"),
            ("US banking stress", "2023-03-10"),
            ("Israel-Hamas conflict", "2023-10-09"),
        ],
        columns=["event", "event_date"],
    )
    event_rows = []
    for market, group in features.groupby("market"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for event in events.itertuples(index=False):
            target = pd.Timestamp(event.event_date, tz="UTC")
            candidates = np.flatnonzero(group["timestamp"].ge(target).to_numpy())
            if len(candidates) == 0:
                continue
            center = int(candidates[0])
            window = group.iloc[max(0, center - 5) : center + 11].copy()
            window["relative_day"] = np.arange(len(window)) - min(5, center)
            window["cumulative_return"] = window["return"].fillna(0).cumsum()
            for row in window.itertuples(index=False):
                event_rows.append(
                    {
                        "market": market,
                        "event": event.event,
                        "relative_day": row.relative_day,
                        "cumulative_return": row.cumulative_return,
                    }
                )
    write_frame(pd.DataFrame(event_rows), out / "event_paths.csv")

    regime_rows = []
    for (market, crisis), group in features.groupby(["market", "crisis"]):
        paired = group[["return", "equity_uncertainty_change"]].dropna()
        regime_rows.append(
            {
                "market": market,
                "regime": "COVID crisis" if crisis else "other sessions",
                "n": len(paired),
                "return_uncertainty_correlation": paired["return"].corr(
                    -paired["equity_uncertainty_change"]
                ),
                "mean_abs_return": paired["return"].abs().mean(),
            }
        )
    write_frame(pd.DataFrame(regime_rows), out / "regime_results.csv")

    rng = np.random.default_rng(settings.project.seed)
    placebo_rows = []
    for market, group in features.groupby("market"):
        pair = group[["return", "equity_uncertainty_change"]].dropna()
        observed = pair["return"].corr(-pair["equity_uncertainty_change"].shift(1))
        signal = pair["equity_uncertainty_change"].to_numpy()
        for draw in range(500):
            shift = int(rng.integers(30, len(signal) - 30))
            placebo = np.roll(signal, shift)
            placebo_rows.append(
                {
                    "market": market,
                    "draw": draw,
                    "correlation": np.corrcoef(pair["return"].to_numpy(), -np.roll(placebo, 1))[
                        0, 1
                    ],
                    "observed": observed,
                }
            )
    write_frame(pd.DataFrame(placebo_rows), out / "placebo_results.csv")
    summary = {
        "observations": int(len(raw)),
        "markets": sorted(raw["market"].unique()),
        "start": str(raw["timestamp"].min().date()),
        "end": str(raw["timestamp"].max().date()),
        "disclaimer": (
            "News uncertainty is an observable proxy, not identical to direct "
            "FinBERT polarity or StockTwits sentiment."
        ),
    }
    (out / "empirical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out
