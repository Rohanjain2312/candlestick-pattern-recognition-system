"""Next-day direction target for the downstream study.

The target on date *t* is whether the close on the **next** trading day is above
the close on *t*.  This is the only place in the project that deliberately looks
forward, and it is the thing being predicted -- never a feature.  The final bar
of the series has no next day and is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def next_day_direction(df: pd.DataFrame) -> pd.Series:
    """Binary up/down label for the move from *t* to *t+1*.

    Args:
        df: OHLC bars indexed by ascending date.

    Returns:
        Int series indexed by date: 1 if ``close[t+1] > close[t]`` else 0, with
        the final date dropped because its outcome is unknown.  Exactly-flat
        days are labelled 0 (not up), which is the conservative choice for a
        "will it rise" question.
    """
    close = df["Close"].astype("float64")
    fwd = close.shift(-1) / close - 1.0
    return (fwd > 0).astype("int8").iloc[:-1]


def next_day_return(df: pd.DataFrame) -> pd.Series:
    """Simple (not log) next-day return, used for the illustrative P&L check."""
    close = df["Close"].astype("float64")
    return (close.shift(-1) / close - 1.0).iloc[:-1]


def base_rate(labels: pd.Series) -> float:
    """Fraction of up-days -- the accuracy a majority-class guesser would get.

    Reported alongside every model score, because on a long-drifting index like
    SPY the majority-class rate is meaningfully above 50% and a model that only
    matches it has learned nothing.
    """
    return float(np.mean(labels))
