# Market Information Delay Research

This repository implements the empirical framework for the dissertation **Measuring Information Delays in Developed Equity Markets Through Financial News and Investor Sentiment Analytics**.

## Research questions implemented

The package tests whether news or investor sentiment contains information that reaches daily S&P 500 and FTSE 100 returns with delay. It provides:

- conservative publication-time to market-session alignment;
- FinBERT probability scoring with a transparent lexicon fallback;
- daily polarity, disagreement and attention features;
- leakage-safe lagged feature construction;
- Granger non-causality tests implemented from restricted and unrestricted OLS models;
- a Hou–Moskowitz-inspired price-delay ratio with moving-block bootstrap intervals;
- event-study utilities;
- expanding-window forecasts comparing price-only and sentiment-enhanced models;
- immutable configuration, hashes, output ledgers and unit tests;
- deterministic synthetic fixtures for software validation;
- public FRED and HistData acquisition with hashes and provenance manifests;
- ADF, KPSS, Ljung–Box and ARCH diagnostics;
- corrected Granger tests, block-bootstrap delay intervals and placebo shifts;
- paired Ridge, Random Forest and histogram-gradient-boosting forecasts;
- 22 publication-resolution empirical figures.

## Quick start

```bash
make install
make test
make data
make empirical
make figures
```

Generated empirical tables and figures appear in `reports/empirical/`. Synthetic results in `reports/generated/` are software fixtures only and must not be used as dissertation findings.

## Empirical input contracts

`data/processed/market.csv`:

| field | description |
|---|---|
| timestamp | UTC exchange-session timestamp |
| market | `SP500` or `FTSE100` |
| close | positive adjusted index close |

`data/processed/text.csv`:

| field | description |
|---|---|
| published_at | first public availability timestamp in UTC |
| market | `SP500` or `FTSE100` |
| text | headline, article excerpt or public investor message |
| source | provider or platform name |

To rebuild the empirical study:

```bash
make data
make empirical
make figures
```

The reproducible public-data run uses news-derived uncertainty indices as auditable market-level proxies. The optional text pipeline uses a lightweight lexicon baseline unless the ingestion workflow first writes FinBERT scores. To score a licensed article corpus directly, instantiate `FinBERTScorer` and append its probability columns before feature aggregation. This separation avoids representing a proxy series as article-level FinBERT output.

## Leakage controls

Every document is assigned to the next eligible trading session after publication. Predictors are shifted within market, rolling volatility excludes the current target, preprocessing is fitted inside each training fold, and an embargo separates training from testing. The forecast ledger stores every prediction rather than reporting only aggregate metrics.

## Interpretation

Granger rejection means lagged sentiment improves conditional prediction; it does not prove structural causality. A delay conclusion should require agreement across timing checks, inference, rolling forecasts, robustness analyses and economic materiality. Statistical significance alone is insufficient.

## Project layout

```text
config/                 Versioned analysis settings
data/                   Input contracts and pre-specified event register
src/market_delay/       Research package
tests/                  Leakage, inference and integration tests
reports/generated/      Reproducible outputs
reports/empirical/      Statistical tables and dissertation figures
Makefile                Environment, QA and run targets
```

## Academic and technical influences

- Udacity, *Artificial Intelligence for Trading* repository: project organisation, factor thinking, backtesting discipline and out-of-sample validation.
- Granger (1969): temporal predictive-content testing.
- Hou and Moskowitz (2005): price-delay measurement principle.
- MacKinlay (1997): event-study methodology.
- Araci (2019) and the ProsusAI FinBERT model card: financial-domain sentiment scoring.
- Official scikit-learn documentation: ordered time-series validation principles.

## Research integrity

Do not commit licensed raw text, platform usernames or provider credentials. Preserve only permitted aggregate outputs. Freeze and hash empirical inputs before the final holdout is opened. Report null findings and model limitations alongside significant results.
