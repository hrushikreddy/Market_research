import pandas as pd
import pytest

from market_delay.data import generate_synthetic_data, validate_columns


def test_synthetic_data_is_deterministic() -> None:
    first = generate_synthetic_data(seed=7, periods=20)
    second = generate_synthetic_data(seed=7, periods=20)
    pd.testing.assert_frame_equal(first.market, second.market)
    pd.testing.assert_frame_equal(first.text, second.text)


def test_validate_columns_reports_missing() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(pd.DataFrame({"a": [1]}), {"a", "b"}, "fixture")
