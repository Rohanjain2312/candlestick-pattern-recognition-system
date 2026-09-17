"""Tests for the downstream feature and label construction.

`test_features_are_causal` is the leakage guard for the price features, the
counterpart to the same test on the labeller.
"""

import numpy as np
import pandas as pd
import pytest

from src.downstream_signal.features import BASELINE_FEATURES, build_features, rsi
from src.downstream_signal.labels import base_rate, next_day_direction, next_day_return


def test_schema_and_index(synthetic_ohlc):
    feats = build_features(synthetic_ohlc)
    assert list(feats.columns) == BASELINE_FEATURES
    assert feats.index.equals(synthetic_ohlc.index)


def test_features_are_causal(synthetic_ohlc):
    """Values for the first n dates must not change when later bars are removed.

    Any rolling window accidentally written as centred or forward-looking would
    break this, and would silently hand the classifier tomorrow's information.
    """
    cutoff = 250
    full = build_features(synthetic_ohlc).iloc[:cutoff]
    truncated = build_features(synthetic_ohlc.iloc[:cutoff])
    pd.testing.assert_frame_equal(full, truncated)


def test_no_nans_after_warmup(synthetic_ohlc):
    feats = build_features(synthetic_ohlc)
    # 50-day SMA is the longest lookback.
    assert not feats.iloc[60:].isna().any().any()


def test_rsi_stays_in_bounds(synthetic_ohlc):
    values = rsi(synthetic_ohlc["Close"], 14)
    assert values.min() >= 0.0 and values.max() <= 100.0


def test_rsi_saturates_on_a_monotonic_series():
    up = pd.Series(np.linspace(100, 200, 100))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_next_day_direction_matches_definition(synthetic_ohlc):
    y = next_day_direction(synthetic_ohlc)
    close = synthetic_ohlc["Close"]
    # Last bar has no next day and must be dropped.
    assert len(y) == len(synthetic_ohlc) - 1
    expected = (close.shift(-1) > close).iloc[:-1].astype(int)
    assert (y.to_numpy() == expected.to_numpy()).all()


def test_direction_is_the_sign_of_the_return(synthetic_ohlc):
    y = next_day_direction(synthetic_ohlc)
    r = next_day_return(synthetic_ohlc)
    assert ((r > 0).astype(int).to_numpy() == y.to_numpy()).all()


def test_flat_day_counts_as_down():
    idx = pd.bdate_range("2022-01-03", periods=3)
    df = pd.DataFrame({"Open": [1, 1, 1], "High": [1, 1, 1],
                       "Low": [1, 1, 1], "Close": [10.0, 10.0, 11.0]}, index=idx)
    assert next_day_direction(df).tolist() == [0, 1]


def test_base_rate_is_a_fraction(synthetic_ohlc):
    rate = base_rate(next_day_direction(synthetic_ohlc))
    assert 0.0 <= rate <= 1.0
