from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Settings
from .data import generate_synthetic_data, load_market, load_text
from .features import aggregate_sentiment, assign_information_session, build_features
from .forecast import rolling_forecast
from .inference import block_bootstrap_delay, granger_test, price_delay
from .nlp import LexiconScorer
from .report import write_frame, write_manifest


def _analyse(settings: Settings, market: pd.DataFrame, text: pd.DataFrame, demo: bool) -> Path:
    output = settings.project.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if "sentiment" not in text:
        scores = LexiconScorer().score(text["text"].astype(str).tolist())
        text = pd.concat([text.reset_index(drop=True), scores], axis=1)
    mapped = assign_information_session(text, market)
    daily = aggregate_sentiment(mapped)
    features = build_features(
        market,
        daily,
        settings.features.return_lags,
        settings.features.sentiment_lags,
        settings.features.volatility_windows,
    )
    write_frame(features, output / "feature_table.csv")
    inference_rows = []
    for market_name, group in features.groupby("market"):
        g = granger_test(group["return"], group["sentiment"], settings.analysis.max_lag)
        delay = price_delay(group["return"], group["sentiment"], settings.analysis.max_lag)
        ci = block_bootstrap_delay(
            group["return"],
            group["sentiment"],
            settings.analysis.max_lag,
            settings.analysis.bootstrap_samples,
            settings.analysis.bootstrap_block,
            settings.project.seed,
        )
        for item in g:
            inference_rows.append(
                {
                    "market": market_name,
                    **item.__dict__,
                    **delay,
                    "delay_ci_low": ci[0],
                    "delay_ci_high": ci[1],
                }
            )
    write_frame(pd.DataFrame(inference_rows), output / "inference.csv")
    price_features = [
        c for c in features if c.startswith("return_lag_") or c.startswith("volatility_")
    ]
    enhanced_features = (
        price_features
        + [c for c in features if c.startswith("sentiment_lag_")]
        + ["attention", "sentiment_dispersion", "sentiment_missing"]
    )
    forecasts = rolling_forecast(
        features,
        {"price_only": price_features, "sentiment_enhanced": enhanced_features},
        settings.validation.initial_train_size,
        settings.validation.test_size,
        settings.validation.step_size,
        settings.validation.embargo,
        settings.project.seed,
    )
    write_frame(forecasts.predictions, output / "predictions.csv")
    write_frame(forecasts.metrics, output / "forecast_metrics.csv")
    write_manifest(
        output / "run_manifest.json",
        settings.model_dump(),
        {"mode": "synthetic" if demo else "empirical"},
    )
    return output


def run_demo(settings: Settings) -> Path:
    synthetic = generate_synthetic_data(settings.project.seed)
    return _analyse(settings, synthetic.market, synthetic.text, demo=True)


def run_empirical(settings: Settings) -> Path:
    market = load_market(settings.data.market_file)
    text = load_text(settings.data.text_file)
    return _analyse(settings, market, text, demo=False)
