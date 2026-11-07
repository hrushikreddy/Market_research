from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MARKET_CLOSE = {
    "SP500": (ZoneInfo("America/New_York"), time(16, 0)),
    "FTSE100": (ZoneInfo("Europe/London"), time(16, 30)),
}


def assign_information_session(text: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Assign each document to the first session whose forecast origin follows publication.

    Daily forecasts are formed at the prior close. Therefore even intraday text enters the
    next session's feature set, a conservative rule that prevents same-day look-ahead.
    """
    outputs = []
    for market, group in text.groupby("market", sort=False):
        calendar = sessions.loc[sessions["market"].eq(market), ["timestamp"]].sort_values(
            "timestamp"
        )
        if calendar.empty:
            continue
        left = group.sort_values("published_at").copy()
        left["eligible_after"] = left["published_at"].dt.normalize() + pd.Timedelta(days=1)
        right = calendar.rename(columns={"timestamp": "session_date"}).copy()
        right["session_date"] = right["session_date"].dt.normalize()
        mapped = pd.merge_asof(
            left.sort_values("eligible_after"),
            right.sort_values("session_date"),
            left_on="eligible_after",
            right_on="session_date",
            direction="forward",
        )
        outputs.append(mapped)
    if not outputs:
        return text.assign(session_date=pd.NaT)
    return pd.concat(outputs, ignore_index=True)


def aggregate_sentiment(mapped: pd.DataFrame) -> pd.DataFrame:
    frame = mapped.dropna(subset=["session_date"]).copy()
    if "sentiment" not in frame:
        raise ValueError("text frame must include sentiment scores")
    grouped = frame.groupby(["market", "session_date"], as_index=False).agg(
        sentiment=("sentiment", "mean"),
        sentiment_dispersion=("sentiment", "std"),
        attention=("sentiment", "size"),
    )
    grouped["sentiment_dispersion"] = grouped["sentiment_dispersion"].fillna(0.0)
    grouped["attention"] = np.log1p(grouped["attention"])
    return grouped


def build_features(
    market: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    return_lags: list[int],
    sentiment_lags: list[int],
    volatility_windows: list[int],
) -> pd.DataFrame:
    frame = market.copy()
    frame["session_date"] = frame["timestamp"].dt.normalize()
    frame = frame.merge(daily_sentiment, on=["market", "session_date"], how="left")
    frame["sentiment_missing"] = frame["sentiment"].isna().astype(int)
    for col in ["sentiment", "sentiment_dispersion", "attention"]:
        frame[col] = frame[col].fillna(0.0)
    groups = frame.groupby("market", group_keys=False)
    for lag in return_lags:
        frame[f"return_lag_{lag}"] = groups["return"].shift(lag)
    for lag in sentiment_lags:
        frame[f"sentiment_lag_{lag}"] = groups["sentiment"].shift(lag)
    for window in volatility_windows:
        frame[f"volatility_{window}"] = groups["return"].transform(
            lambda x, w=window: x.shift(1).rolling(w).std()
        )
    frame["target_return"] = groups["return"].shift(-1)
    return frame.sort_values(["session_date", "market"]).reset_index(drop=True)
