"""Tests for the TA-Lib rule-based labeller.

The important one is `test_labels_are_causal`: the entire no-leakage claim of
the project rests on TA-Lib only looking backwards, so it is asserted rather
than assumed.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.labeling.talib_labeler import PATTERN_RULES, label_patterns, pattern_counts


def test_returns_expected_schema(synthetic_ohlc):
    """Downstream code indexes these columns by name, so the schema is part of the contract."""
    labels = label_patterns(synthetic_ohlc)
    assert list(labels.columns) == [
        "position", "date", "pattern", "class_id", "span", "direction"
    ]
    assert set(labels["pattern"]) <= set(config.CLASSES)


def test_every_class_has_a_rule():
    """A configured class with no rule would never be labelled, and the detector would learn it cannot occur."""
    # A class present in config but missing a rule would silently never be
    # labelled, and the detector would learn it can never occur.
    assert set(PATTERN_RULES) == set(config.CLASSES)
    assert set(config.PATTERN_SPAN) == set(config.CLASSES)


def test_class_id_matches_config(synthetic_ohlc):
    """Ids are baked into the label files; a mismatch yields a model that reports the wrong class forever."""
    labels = label_patterns(synthetic_ohlc)
    for row in labels.itertuples():
        assert config.CLASSES[row.class_id] == row.pattern


def test_labels_are_causal(synthetic_ohlc):
    """Truncating the future must not change any past label.

    If this fails, every downstream result is contaminated: a chart for date t
    would carry information from after t.
    """
    cutoff = 220
    full = label_patterns(synthetic_ohlc)
    truncated = label_patterns(synthetic_ohlc.iloc[:cutoff])

    full_past = full[full["position"] < cutoff].reset_index(drop=True)
    trunc = truncated[truncated["position"] < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        full_past[["position", "pattern", "class_id"]],
        trunc[["position", "pattern", "class_id"]],
    )


def test_span_never_runs_off_the_start(synthetic_ohlc):
    """A pattern whose span predates the series would produce a box over candles that do not exist."""
    labels = label_patterns(synthetic_ohlc)
    assert (labels["position"] - labels["span"] + 1 >= 0).all()


def test_engulfing_split_is_disjoint(synthetic_ohlc):
    """Bullish and bearish engulfing come from one TA-Lib function split by
    sign, so the same bar must never be labelled both."""
    labels = label_patterns(synthetic_ohlc)
    bull = set(labels.loc[labels.pattern == "BullishEngulfing", "position"])
    bear = set(labels.loc[labels.pattern == "BearishEngulfing", "position"])
    assert not (bull & bear)


def test_explicit_doji_is_detected():
    """A bar whose open equals its close is a textbook Doji."""
    n = 40
    idx = pd.bdate_range("2021-01-01", periods=n, name="Date")
    base = np.linspace(100, 110, n)
    df = pd.DataFrame({
        "Open": base, "Close": base * 1.01,
        "High": base * 1.02, "Low": base * 0.99,
    }, index=idx)
    # Force bar 30 to be a perfect doji with long wicks.
    df.iloc[30] = [base[30], base[30], base[30] * 1.03, base[30] * 0.97]
    labels = label_patterns(df)
    hits = labels[(labels.position == 30) & (labels.pattern == "Doji")]
    assert len(hits) == 1


def test_pattern_counts_includes_absent_classes(synthetic_ohlc):
    """Absent classes must report 0 rather than vanish, so the balance table stays complete."""
    counts = pattern_counts(label_patterns(synthetic_ohlc))
    assert list(counts.index) == config.CLASSES
    assert (counts >= 0).all()


def test_empty_input_returns_empty_frame():
    """An empty series must yield an empty label frame with the right columns,
    not raise -- a ticker with no history should be skipped, not crash a build."""
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close"], dtype="float64")
    out = label_patterns(empty)
    assert len(out) == 0
    assert list(out.columns) == [
        "position", "date", "pattern", "class_id", "span", "direction"
    ]
