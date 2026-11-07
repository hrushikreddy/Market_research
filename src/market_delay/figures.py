from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Settings


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures(settings: Settings) -> Path:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    out = settings.project.output_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        settings.project.output_dir / "empirical_features.csv", parse_dates=["timestamp"]
    )
    pred = pd.read_csv(settings.project.output_dir / "predictions.csv", parse_dates=["timestamp"])
    metrics = pd.read_csv(settings.project.output_dir / "forecast_metrics.csv")
    inference = pd.read_csv(settings.project.output_dir / "granger_and_delay.csv")
    loss_tests = pd.read_csv(settings.project.output_dir / "forecast_loss_tests.csv")
    events = pd.read_csv(settings.project.output_dir / "event_paths.csv")
    regimes = pd.read_csv(settings.project.output_dir / "regime_results.csv")
    placebo = pd.read_csv(settings.project.output_dir / "placebo_results.csv")

    for market, group in frame.groupby("market"):
        group = group.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(group.timestamp, group.close, lw=1.2)
        ax.set(title=f"{market} daily close", ylabel="Index level", xlabel="")
        _save(fig, out / f"01_{market.lower()}_close.png")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(
            group.timestamp,
            group["return"].rolling(21).std() * np.sqrt(252),
            lw=1.1,
            color="#B44C43",
        )
        ax.set(
            title=f"{market} annualised 21-session volatility",
            ylabel="Annualised volatility",
            xlabel="",
        )
        _save(fig, out / f"02_{market.lower()}_volatility.png")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(group["return"].dropna(), bins=60, kde=True, ax=ax, color="#2A6F97")
        ax.set(title=f"{market} daily return distribution", xlabel="Log return")
        _save(fig, out / f"03_{market.lower()}_return_distribution.png")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(group.timestamp, group.equity_uncertainty, color="#5C4B8A", lw=1)
        ax.set(
            title=f"News-based equity uncertainty aligned with {market}",
            ylabel="7-day mean index",
            xlabel="",
        )
        _save(fig, out / f"04_{market.lower()}_uncertainty.png")
        cols = [
            "return",
            "equity_uncertainty_change",
            "vix_change",
            "gbp_usd_change",
            "volatility_21",
        ]
        corr = group[cols].corr()
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        ax.set_title(f"{market} correlation matrix")
        _save(fig, out / f"05_{market.lower()}_correlation.png")
        lags = range(-10, 11)
        values = [
            group["return"].corr((-group["equity_uncertainty_change"]).shift(lag)) for lag in lags
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(list(lags), values, color=["#B44C43" if x > 0 else "#2A6F97" for x in lags])
        ax.axhline(0, color="black", lw=0.8)
        ax.set(
            title=f"{market} return and news-sentiment proxy lead-lag correlation",
            xlabel="Signal lag in sessions",
            ylabel="Correlation",
        )
        _save(fig, out / f"06_{market.lower()}_lead_lag.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(data=frame, x="timestamp", y="equity_uncertainty", hue="market", ax=ax)
    ax.set(title="Common news-based equity uncertainty by market sample", xlabel="", ylabel="Index")
    _save(fig, out / "07_uncertainty_comparison.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=metrics, x="model", y="rmse", hue="feature_set", errorbar=None, ax=ax)
    ax.set(title="Out-of-sample RMSE by model and feature set", xlabel="", ylabel="RMSE")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out / "08_forecast_rmse.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(
        data=metrics, x="model", y="directional_accuracy", hue="feature_set", errorbar=None, ax=ax
    )
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.set(title="Out-of-sample directional accuracy", xlabel="", ylabel="Share correct")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out / "09_directional_accuracy.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=metrics, x="model", y="net_sharpe", hue="feature_set", errorbar=None, ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Cost-adjusted strategy Sharpe ratio", xlabel="", ylabel="Annualised Sharpe")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out / "10_net_sharpe.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    best = pred[(pred.model == "ridge") & (pred.feature_set == "news_and_controls")].copy()
    best["cumulative_actual"] = best.groupby("market")["actual"].cumsum()
    sns.lineplot(data=best, x="timestamp", y="cumulative_actual", hue="market", ax=ax)
    ax.set(
        title="Cumulative realised return over forecast evaluation windows",
        xlabel="",
        ylabel="Cumulative log return",
    )
    _save(fig, out / "11_evaluation_cumulative_returns.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=inference, x="lag", y="p_value", hue="market", marker="o", ax=ax)
    ax.axhline(0.05, color="black", ls="--", lw=1)
    ax.set(title="Granger non-causality p-values across lags", ylabel="p-value", xlabel="Lag order")
    _save(fig, out / "12_granger_pvalues.png")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    d = inference.groupby("market", as_index=False).first()
    sns.barplot(data=d, x="market", y="delay", hue="market", legend=False, ax=ax)
    ax.set(title="Estimated news-information delay ratio", xlabel="", ylabel="Delay ratio")
    _save(fig, out / "13_delay_ratio.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    draw = frame.dropna(subset=["return", "equity_uncertainty_change"]).sample(
        min(1600, len(frame)), random_state=settings.project.seed
    )
    sns.regplot(
        data=draw,
        x="equity_uncertainty_change",
        y="return",
        scatter_kws={"alpha": 0.25, "s": 12},
        line_kws={"color": "black"},
        ax=ax,
    )
    ax.set(
        title="Contemporaneous uncertainty innovation and market return",
        xlabel="Change in log news-uncertainty",
        ylabel="Daily return",
    )
    _save(fig, out / "14_uncertainty_return_scatter.png")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=loss_tests, x="model", y="mean_loss_improvement", hue="market", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title="Forecast loss improvement from news and controls",
        xlabel="",
        ylabel="Price-only squared loss minus enhanced loss",
    )
    ax.tick_params(axis="x", rotation=15)
    _save(fig, out / "15_loss_improvement.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=events,
        x="relative_day",
        y="cumulative_return",
        hue="market",
        style="event",
        units="event",
        estimator=None,
        alpha=0.72,
        ax=ax,
    )
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set(
        title="Cumulative market paths around pre-specified events",
        xlabel="Trading sessions from event",
        ylabel="Cumulative log return",
    )
    _save(fig, out / "16_event_paths.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.barplot(data=regimes, x="market", y="return_uncertainty_correlation", hue="regime", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Return and news-sentiment correlation by regime", xlabel="", ylabel="Correlation")
    _save(fig, out / "17_regime_correlation.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (market, g) in zip(axes, placebo.groupby("market"), strict=True):
        sns.histplot(g["correlation"], bins=35, ax=ax, color="#7A8FA6")
        ax.axvline(g["observed"].iloc[0], color="#B44C43", lw=2, label="Observed")
        ax.set(title=f"{market} placebo distribution", xlabel="Lag-one correlation")
        ax.legend()
    _save(fig, out / "18_placebo_correlations.png")
    temp = frame.sort_values(["market", "timestamp"]).copy()
    temp["rolling_corr"] = (
        temp.groupby("market", group_keys=False)
        .apply(
            lambda g: g["return"].rolling(126).corr(-g["equity_uncertainty_change"]),
            include_groups=False,
        )
        .reset_index(level=0, drop=True)
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(data=temp, x="timestamp", y="rolling_corr", hue="market", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Rolling 126-session return-news correlation", xlabel="", ylabel="Correlation")
    _save(fig, out / "19_rolling_correlation.png")
    return out
