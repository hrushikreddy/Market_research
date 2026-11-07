from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_COLUMNS = {"timestamp", "market", "close"}
TEXT_COLUMNS = {"published_at", "market", "text", "source"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_market(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_columns(frame, MARKET_COLUMNS, "market data")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["market", "timestamp"]).drop_duplicates(
        ["market", "timestamp"], keep="last"
    )
    if (frame["close"] <= 0).any():
        raise ValueError("close prices must be strictly positive")
    frame["return"] = frame.groupby("market")["close"].transform(lambda x: np.log(x).diff())
    return frame


def load_text(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_columns(frame, TEXT_COLUMNS, "text data")
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame["text_hash"] = (
        frame["text"]
        .astype(str)
        .map(lambda value: hashlib.sha256(" ".join(value.lower().split()).encode()).hexdigest())
    )
    return frame.drop_duplicates(["market", "text_hash", "published_at"])


@dataclass(frozen=True)
class SyntheticData:
    market: pd.DataFrame
    text: pd.DataFrame


def generate_synthetic_data(seed: int = 2026, periods: int = 900) -> SyntheticData:
    """Generate deterministic research fixtures with a known one-day sentiment effect.

    Synthetic outputs are for software validation only and must never be reported as
    empirical dissertation findings.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=periods, tz="UTC")
    market_rows: list[pd.DataFrame] = []
    text_rows: list[dict[str, object]] = []
    for market, beta, base in [("SP500", 0.0015, 3200.0), ("FTSE100", 0.0025, 7500.0)]:
        sentiment = np.zeros(periods)
        for i in range(1, periods):
            sentiment[i] = 0.65 * sentiment[i - 1] + rng.normal(0, 0.35)
        noise = rng.normal(0, 0.009 if market == "SP500" else 0.011, periods)
        returns = noise + beta * np.r_[0.0, sentiment[:-1]]
        close = base * np.exp(np.cumsum(returns))
        market_rows.append(
            pd.DataFrame({"timestamp": dates, "market": market, "close": close, "return": returns})
        )
        for date, score in zip(dates, sentiment, strict=True):
            adjective = "strong profit growth" if score >= 0 else "weak loss risk"
            text_rows.append(
                {
                    "published_at": date + pd.Timedelta(hours=12),
                    "market": market,
                    "source": "synthetic_fixture",
                    "text": f"{market} outlook shows {adjective} score {score:.5f}",
                    "sentiment": float(np.tanh(score)),
                }
            )
    return SyntheticData(pd.concat(market_rows, ignore_index=True), pd.DataFrame(text_rows))
