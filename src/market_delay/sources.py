from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FRED_SERIES = {
    "SP500": "SP500",
    "vix": "VIXCLS",
    "equity_uncertainty": "WLEMUINDXD",
    "us_policy_uncertainty": "USEPUINDXD",
    "uk_policy_uncertainty": "UKEPUINDXM",
    "gbp_usd": "DEXUSUK",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_fred(series_id: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        f"id={series_id}&cosd=2020-01-01&coed=2025-12-31"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "academic-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        destination.write_bytes(response.read())
    return destination


def load_fred(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame.columns) != 2:
        raise ValueError(f"Unexpected FRED schema in {path}")
    frame.columns = ["date", name]
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def histdata_zip_to_daily(path: Path, market: str) -> pd.DataFrame:
    """Aggregate public HistData minute bars to UTC-labelled daily OHLC observations."""
    with zipfile.ZipFile(path) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        raw = archive.read(csv_name)
    frame = pd.read_csv(
        io.BytesIO(raw),
        sep=";",
        header=None,
        names=["stamp", "open", "high", "low", "close", "volume"],
        usecols=range(6),
    )
    frame["timestamp"] = pd.to_datetime(frame["stamp"], format="%Y%m%d %H%M%S", utc=True)
    frame["date"] = frame["timestamp"].dt.normalize()
    daily = frame.groupby("date", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        minute_count=("close", "size"),
    )
    daily["market"] = market
    return daily


def prepare_empirical_data(root: Path) -> dict[str, Path]:
    raw_fred = root / "data/raw/fred"
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    fred_paths = {}
    for name, series_id in FRED_SERIES.items():
        path = raw_fred / f"{series_id}.csv"
        if not path.exists() or path.stat().st_size < 100:
            download_fred(series_id, path)
        fred_paths[name] = path

    hist_root = root / "data/raw/histdata"
    daily_parts = []
    for market, pair in [("SP500", "SPXUSD"), ("FTSE100", "UKXGBP")]:
        for path in sorted(hist_root.glob(f"DAT_ASCII_{pair}_M1_*.zip")):
            daily_parts.append(histdata_zip_to_daily(path, market))
    market = pd.concat(daily_parts, ignore_index=True).sort_values(["market", "date"])
    # FRED closes fill the final SP500 year when minute data are unavailable.
    fred_sp = load_fred(fred_paths["SP500"], "fred_close").rename(columns={"date": "date"})
    sp_extra = fred_sp[fred_sp["date"].dt.year.eq(2025)].dropna().assign(market="SP500")
    sp_extra["open"] = sp_extra["high"] = sp_extra["low"] = sp_extra["fred_close"]
    sp_extra["close"] = sp_extra["fred_close"]
    sp_extra["minute_count"] = np.nan
    market = pd.concat([market, sp_extra[market.columns]], ignore_index=True)
    market = market.drop_duplicates(["market", "date"], keep="first").sort_values(
        ["market", "date"]
    )
    market["return"] = market.groupby("market")["close"].transform(lambda x: np.log(x).diff())
    market["range_volatility"] = np.log(market["high"] / market["low"])

    controls = load_fred(fred_paths["vix"], "vix")
    for key in ["equity_uncertainty", "us_policy_uncertainty", "gbp_usd"]:
        controls = controls.merge(load_fred(fred_paths[key], key), on="date", how="outer")
    uk_monthly = load_fred(fred_paths["uk_policy_uncertainty"], "uk_policy_uncertainty")
    controls = controls.merge(uk_monthly, on="date", how="outer").sort_values("date")
    controls["uk_policy_uncertainty"] = controls["uk_policy_uncertainty"].ffill(limit=31)
    controls["equity_uncertainty"] = controls["equity_uncertainty"].rolling(7, min_periods=1).mean()

    merged = market.merge(controls, on="date", how="left")
    merged = merged.rename(columns={"date": "timestamp"})
    market_path = processed / "market_empirical.csv"
    merged.to_csv(market_path, index=False)

    # Convert news-derived uncertainty indices into an explicitly named proxy table.
    proxies = []
    for market_name in ["SP500", "FTSE100"]:
        local = controls.copy()
        local["market"] = market_name
        local["published_at"] = local["date"] + pd.Timedelta(hours=6)
        local["source"] = "FRED/Baker-Bloom-Davis"
        local["text"] = "News-based uncertainty index observation"
        market_epu = (
            local["us_policy_uncertainty"]
            if market_name == "SP500"
            else local["uk_policy_uncertainty"]
        )
        combined = 0.7 * np.log1p(local["equity_uncertainty"]) + 0.3 * np.log1p(market_epu)
        local["sentiment"] = -(
            (combined - combined.rolling(252, min_periods=60).mean())
            / combined.rolling(252, min_periods=60).std()
        )
        proxies.append(local[["published_at", "market", "source", "text", "sentiment"]])
    proxy_path = processed / "news_proxy.csv"
    pd.concat(proxies, ignore_index=True).dropna(subset=["sentiment"]).to_csv(
        proxy_path, index=False
    )

    manifest = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "sources": {
            "HistData": "https://www.histdata.com/download-free-forex-data/",
            "FRED": "https://fred.stlouisfed.org/",
        },
        "files": {
            str(path.relative_to(root)): _sha256(path)
            for path in [*fred_paths.values(), market_path, proxy_path]
        },
        "notes": (
            "SPXUSD and UKXGBP minute quotations aggregated to daily; "
            "FRED S&P 500 fills 2025 SP500 closes."
        ),
    }
    manifest_path = processed / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"market": market_path, "proxy": proxy_path, "manifest": manifest_path}
