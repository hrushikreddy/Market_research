
from market_delay.data import generate_synthetic_data
from market_delay.features import aggregate_sentiment, assign_information_session, build_features


def test_text_is_assigned_to_future_session() -> None:
    data = generate_synthetic_data(periods=30)
    mapped = assign_information_session(data.text, data.market)
    valid = mapped.dropna(subset=["session_date"])
    assert (valid["session_date"] > valid["published_at"].dt.normalize()).all()


def test_feature_lags_do_not_use_future() -> None:
    data = generate_synthetic_data(periods=40)
    daily = aggregate_sentiment(assign_information_session(data.text, data.market))
    features = build_features(data.market, daily, [1], [1], [5])
    group = features[features.market.eq("SP500")].reset_index(drop=True)
    assert group.loc[10, "return_lag_1"] == group.loc[9, "return"]
    assert group.loc[10, "target_return"] == group.loc[11, "return"]
