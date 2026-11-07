from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class SentimentScorer(Protocol):
    def score(self, texts: Sequence[str]) -> pd.DataFrame: ...


@dataclass
class LexiconScorer:
    """Transparent fallback and test baseline; not a replacement for validated FinBERT."""

    positive: frozenset[str] = frozenset(
        {"gain", "growth", "strong", "profit", "beat", "upgrade", "bullish", "recovery"}
    )
    negative: frozenset[str] = frozenset(
        {"loss", "weak", "risk", "miss", "downgrade", "bearish", "crisis", "decline"}
    )

    def score(self, texts: Sequence[str]) -> pd.DataFrame:
        rows = []
        for text in texts:
            tokens = {token.strip(".,:;!?()[]").lower() for token in text.split()}
            raw = len(tokens & self.positive) - len(tokens & self.negative)
            polarity = float(np.tanh(raw / 2))
            p_pos = max(polarity, 0.0)
            p_neg = max(-polarity, 0.0)
            p_neu = 1.0 - max(p_pos, p_neg)
            total = p_pos + p_neu + p_neg
            rows.append((p_pos / total, p_neu / total, p_neg / total, polarity))
        return pd.DataFrame(rows, columns=["p_positive", "p_neutral", "p_negative", "sentiment"])


class FinBERTScorer:
    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 256) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the NLP extra with: pip install -e '.[nlp]'") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self.batch_size = batch_size
        self.max_length = max_length
        label_map = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self._indices = {label: idx for idx, label in label_map.items()}
        required = {"positive", "neutral", "negative"}
        if not required.issubset(self._indices):
            raise ValueError(f"Unexpected model labels: {label_map}")

    def score(self, texts: Sequence[str]) -> pd.DataFrame:
        output: list[tuple[float, float, float, float]] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self._tokenizer(
                    list(texts[start : start + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                probs = self._torch.softmax(self._model(**encoded).logits, dim=1).cpu().numpy()
                for row in probs:
                    pos = float(row[self._indices["positive"]])
                    neu = float(row[self._indices["neutral"]])
                    neg = float(row[self._indices["negative"]])
                    output.append((pos, neu, neg, pos - neg))
        return pd.DataFrame(output, columns=["p_positive", "p_neutral", "p_negative", "sentiment"])
