from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    rss = float(residual @ residual)
    return beta, rss, residual


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[
        ::-1
    ]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return pd.DataFrame({"p_value": p, "p_adjusted": adjusted, "reject": adjusted <= alpha})


@dataclass(frozen=True)
class GrangerResult:
    lag: int
    f_statistic: float
    p_value: float
    df_num: int
    df_den: int


def granger_test(target: pd.Series, signal: pd.Series, max_lag: int = 5) -> list[GrangerResult]:
    data = pd.concat([target.rename("y"), signal.rename("s")], axis=1).dropna()
    results = []
    for lag in range(1, max_lag + 1):
        work = data.copy()
        for j in range(1, lag + 1):
            work[f"y{j}"] = work["y"].shift(j)
            work[f"s{j}"] = work["s"].shift(j)
        work = work.dropna()
        y = work["y"].to_numpy()
        restricted = work[[f"y{j}" for j in range(1, lag + 1)]].to_numpy()
        unrestricted = work[
            [f"y{j}" for j in range(1, lag + 1)] + [f"s{j}" for j in range(1, lag + 1)]
        ].to_numpy()
        _, rss_r, _ = _ols(y, restricted)
        _, rss_u, _ = _ols(y, unrestricted)
        df_num = lag
        df_den = len(y) - unrestricted.shape[1] - 1
        f_stat = max(((rss_r - rss_u) / df_num) / (rss_u / df_den), 0.0)
        results.append(
            GrangerResult(lag, f_stat, float(stats.f.sf(f_stat, df_num, df_den)), df_num, df_den)
        )
    return results


def price_delay(target: pd.Series, signal: pd.Series, max_lag: int = 5) -> dict[str, float]:
    data = pd.concat([target.rename("y"), signal.rename("s")], axis=1)
    for lag in range(1, max_lag + 1):
        data[f"s{lag}"] = data["s"].shift(lag)
    data = data.dropna()
    y = data["y"].to_numpy()
    _, rss_r, _ = _ols(y, data[["s"]].to_numpy())
    _, rss_u, _ = _ols(y, data[["s"] + [f"s{i}" for i in range(1, max_lag + 1)]].to_numpy())
    tss = float(((y - y.mean()) ** 2).sum())
    r2_r, r2_u = 1 - rss_r / tss, 1 - rss_u / tss
    delay = np.nan if r2_u <= 0 else 1 - r2_r / r2_u
    return {"r2_restricted": r2_r, "r2_unrestricted": r2_u, "delay": float(delay)}


def block_bootstrap_delay(
    target: pd.Series,
    signal: pd.Series,
    max_lag: int,
    samples: int,
    block: int,
    seed: int,
) -> tuple[float, float]:
    data = pd.concat([target, signal], axis=1).dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    values = []
    starts = np.arange(max(len(data) - block + 1, 1))
    for _ in range(samples):
        chunks = []
        while sum(len(c) for c in chunks) < len(data):
            start = int(rng.choice(starts))
            chunks.append(data.iloc[start : start + block])
        sample = pd.concat(chunks, ignore_index=True).iloc[: len(data)]
        values.append(price_delay(sample.iloc[:, 0], sample.iloc[:, 1], max_lag)["delay"])
    valid = np.asarray([v for v in values if np.isfinite(v)])
    return float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


def event_study(
    returns: pd.DataFrame, events: pd.DataFrame, pre: int = 1, post: int = 3
) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        eligible = returns[returns["market"].eq(event.market)].sort_values("session_date")
        event_date = pd.Timestamp(event.event_time_utc).tz_convert("UTC").normalize()
        positions = np.flatnonzero(eligible["session_date"].ge(event_date).to_numpy())
        if len(positions) == 0:
            continue
        loc = positions[0]
        window = eligible.iloc[max(0, loc - pre) : loc + post + 1]
        for relative, (_, obs) in enumerate(window.iterrows(), start=-min(pre, loc)):
            rows.append(
                {
                    "event_id": event.event_id,
                    "market": event.market,
                    "relative_day": relative,
                    "abnormal_return": obs["return"],
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["car"] = result.groupby(["event_id", "market"])["abnormal_return"].cumsum()
    return result
