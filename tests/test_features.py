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
    """Column order and index must match the source bars exactly; the model selects columns by name."""
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
    """Past the longest lookback every feature must be finite, so rows are dropped for warm-up only."""
    feats = build_features(synthetic_ohlc)
    # 50-day SMA is the longest lookback.
    assert not feats.iloc[60:].isna().any().any()


def test_rsi_stays_in_bounds(synthetic_ohlc):
    """RSI is defined on [0,100]; anything outside means the gain/loss smoothing is wrong."""
    values = rsi(synthetic_ohlc["Close"], 14)
    assert values.min() >= 0.0 and values.max() <= 100.0


def test_rsi_saturates_on_a_monotonic_series():
    """With no down-moves at all, RSI is 100 -- not the neutral 50 an unguarded divide-by-zero would give."""
    up = pd.Series(np.linspace(100, 200, 100))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_next_day_direction_matches_definition(synthetic_ohlc):
    """The target is close[t+1] > close[t], with the final undecidable bar dropped."""
    y = next_day_direction(synthetic_ohlc)
    close = synthetic_ohlc["Close"]
    # Last bar has no next day and must be dropped.
    assert len(y) == len(synthetic_ohlc) - 1
    expected = (close.shift(-1) > close).iloc[:-1].astype(int)
    assert (y.to_numpy() == expected.to_numpy()).all()


def test_direction_is_the_sign_of_the_return(synthetic_ohlc):
    """Label and return must agree, or the P&L check would be measuring a different thing from the classifier."""
    y = next_day_direction(synthetic_ohlc)
    r = next_day_return(synthetic_ohlc)
    assert ((r > 0).astype(int).to_numpy() == y.to_numpy()).all()


def test_flat_day_counts_as_down():
    """An exactly unchanged close is not a rise; 'will it go up' answers no."""
    idx = pd.bdate_range("2022-01-03", periods=3)
    df = pd.DataFrame({"Open": [1, 1, 1], "High": [1, 1, 1],
                       "Low": [1, 1, 1], "Close": [10.0, 10.0, 11.0]}, index=idx)
    assert next_day_direction(df).tolist() == [0, 1]


def test_base_rate_is_a_fraction(synthetic_ohlc):
    """The majority-class rate every accuracy is compared against must be a valid proportion."""
    rate = base_rate(next_day_direction(synthetic_ohlc))
    assert 0.0 <= rate <= 1.0
