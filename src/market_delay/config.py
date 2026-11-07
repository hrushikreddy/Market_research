from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    seed: int = 42
    output_dir: Path = Path("reports/generated")
    data_dir: Path = Path("data/processed")


class AnalysisConfig(BaseModel):
    max_lag: int = Field(5, ge=1, le=20)
    nw_lags: int = Field(5, ge=0, le=30)
    fdr_alpha: float = Field(0.05, gt=0, lt=1)
    bootstrap_samples: int = Field(1000, ge=100)
    bootstrap_block: int = Field(10, ge=2)
    transaction_cost_bps: float = Field(5.0, ge=0)


class ValidationConfig(BaseModel):
    initial_train_size: int = Field(756, ge=100)
    test_size: int = Field(21, ge=1)
    step_size: int = Field(21, ge=1)
    embargo: int = Field(1, ge=0)


class FeatureConfig(BaseModel):
    return_lags: list[int]
    sentiment_lags: list[int]
    volatility_windows: list[int]

    @model_validator(mode="after")
    def positive_unique(self) -> FeatureConfig:
        for name in ("return_lags", "sentiment_lags", "volatility_windows"):
            values = getattr(self, name)
            if any(v <= 0 for v in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique positive integers")
        return self


class NLPConfig(BaseModel):
    model_name: str = "ProsusAI/finbert"
    batch_size: int = Field(16, ge=1)
    max_length: int = Field(256, ge=32, le=512)
    fallback_lexicon: bool = True


class DataConfig(BaseModel):
    market_file: Path
    text_file: Path
    event_file: Path


class Settings(BaseModel):
    project: ProjectConfig
    analysis: AnalysisConfig
    validation: ValidationConfig
    features: FeatureConfig
    nlp: NLPConfig
    data: DataConfig


def load_settings(path: str | Path) -> Settings:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return Settings.model_validate(raw)
