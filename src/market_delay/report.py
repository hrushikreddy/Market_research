from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_manifest(path: Path, settings: dict[str, Any], inputs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "settings": settings,
        "input_sha256": inputs,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
