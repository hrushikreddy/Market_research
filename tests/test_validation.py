from market_delay.validation import expanding_folds


def test_expanding_folds_are_ordered_and_embargoed() -> None:
    folds = expanding_folds(200, initial_train=100, test_size=20, step=20, embargo=2)
    assert len(folds) == 5
    for fold in folds:
        assert fold.train.max() + 2 < fold.test.min()
