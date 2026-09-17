"""Shared fixtures. A deterministic synthetic price series keeps the test suite
offline and reproducible -- no network, no dependence on what SPY did today."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_ohlc() -> pd.DataFrame:
    """300 bars of plausible random-walk OHLC with a fixed seed."""
    rng = np.random.default_rng(7)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.006, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range("2020-01-01", periods=n, name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.uniform(1e6, 5e6, n)},
        index=idx,
    )
