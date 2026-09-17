"""Tests for chronological splitting and the embargo gap.

These guard the claim the README makes: that validation and test charts share no
candle with any training chart. Because consecutive windows overlap in 19 of 20
candles, splitting purely on end-date is *not* sufficient, and a regression here
would inflate every reported detection metric.
"""

import pandas as pd
import pytest

from src import config
from src.labeling.build_dataset import _labels_in_window, split_ranges


@pytest.fixture
def daily_frame() -> pd.DataFrame:
    """A frame spanning all three split periods, indexed by business day."""
    idx = pd.bdate_range("2005-01-03", "2024-12-31", name="Date")
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)


def test_splits_are_ordered_and_non_overlapping(daily_frame):
    """Train must precede val must precede test, with every split non-empty."""
    r = split_ranges(daily_frame)
    assert r["train"].stop <= r["val"].start
    assert r["val"].stop <= r["test"].start
    assert all(len(rg) > 0 for rg in r.values())


def test_val_windows_contain_no_train_bar(daily_frame):
    """The embargo must push the first val window entirely past the train end."""
    r = split_ranges(daily_frame)
    first_val_window_start = r["val"].start - config.WINDOW + 1
    last_train_bar = r["train"].stop - 1
    assert first_val_window_start > last_train_bar


def test_test_windows_contain_no_val_bar(daily_frame):
    """The embargo applies at both boundaries, not just the first."""
    r = split_ranges(daily_frame)
    first_test_window_start = r["test"].start - config.WINDOW + 1
    last_val_bar = r["val"].stop - 1
    assert first_test_window_start > last_val_bar


def test_split_boundaries_match_configured_dates(daily_frame):
    """Splits must land where config says, or the README's stated periods are wrong."""
    r = split_ranges(daily_frame)
    assert daily_frame.index[r["train"].stop - 1] <= pd.Timestamp(config.TRAIN_END)
    assert daily_frame.index[r["val"].start] > pd.Timestamp(config.TRAIN_END)
    assert daily_frame.index[r["test"].start] > pd.Timestamp(config.VAL_END)


def test_no_window_starts_before_the_series(daily_frame):
    """A window needs WINDOW bars of history; the first valid end position is WINDOW-1."""
    r = split_ranges(daily_frame)
    assert r["train"].start >= config.WINDOW - 1


def test_only_fully_contained_spans_are_labelled():
    """A three-candle pattern completing on the window's second candle reaches
    back before the window starts, and must be dropped rather than clipped."""
    window, end = 20, 100
    start = end - window + 1
    by_position = {
        start + 1: [{"span": 3, "pattern": "MorningStar"}],   # reaches to start-1
        start + 2: [{"span": 3, "pattern": "EveningStar"}],   # exactly fits
        end: [{"span": 1, "pattern": "Doji"}],
    }
    kept = {h["pattern"] for h in _labels_in_window(by_position, end, window)}
    assert kept == {"EveningStar", "Doji"}


def test_labels_outside_the_window_are_ignored():
    """Hits before the window starts or after it ends must not be drawn on this chart."""
    window, end = 20, 100
    by_position = {
        end - window: [{"span": 1, "pattern": "Doji"}],       # one bar too early
        end + 1: [{"span": 1, "pattern": "Hammer"}],          # in the future
        end: [{"span": 1, "pattern": "Harami"}],
    }
    kept = {h["pattern"] for h in _labels_in_window(by_position, end, window)}
    assert kept == {"Harami"}
