import numpy as np
import pandas as pd

from market_delay.inference import benjamini_hochberg, granger_test, price_delay


def test_bh_adjustment_is_monotone_in_rank() -> None:
    result = benjamini_hochberg(np.array([0.001, 0.02, 0.2]))
    assert (result.p_adjusted >= result.p_value).all()
    assert result.reject.tolist() == [True, True, False]


def test_granger_detects_known_lag() -> None:
    rng = np.random.default_rng(4)
    signal = pd.Series(rng.normal(size=700))
    target = 0.5 * signal.shift(1) + pd.Series(rng.normal(scale=0.5, size=700))
    result = granger_test(target, signal, max_lag=2)
    assert result[0].p_value < 0.001


def test_price_delay_increases_with_lagged_signal() -> None:
    rng = np.random.default_rng(5)
    signal = pd.Series(rng.normal(size=800))
    target = 0.6 * signal.shift(2) + pd.Series(rng.normal(scale=0.4, size=800))
    result = price_delay(target, signal, max_lag=3)
    assert result["r2_unrestricted"] > result["r2_restricted"]
    assert result["delay"] > 0.5
