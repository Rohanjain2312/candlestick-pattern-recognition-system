"""Baseline price features for the next-day direction study.

Every column here is **causal**: the value on date *t* is computed only from
bars up to and including *t*.  That is the whole point of the exercise -- the
comparison is "baseline price features known at the close of *t*" versus "those
same features plus what the detector saw on the chart ending at *t*", both
predicting the move from *t* to *t+1*.  A single lookahead column would make the
comparison meaningless, so the causality of each feature is noted inline.

The baseline is intentionally the plain, well-known set named in the project
plan -- returns, rolling volatility, RSI and moving-average ratios.  It is not
tuned to be weak: a straw-man baseline would make the pattern features look good
for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "vol_10", "vol_20",
    "rsi_14",
    "close_over_sma5", "close_over_sma20", "close_over_sma50",
    "sma5_over_sma20",
    "volume_ratio_20",
]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    Uses an exponentially weighted mean of gains and losses, which depends only
    on past bars, so the value at *t* is known at the close of *t*.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # A window with no down-moves at all makes RS infinite, which is RSI 100 --
    # not "undefined". Conflating that with the warm-up period (where the value
    # genuinely is unknown) would inject a fake neutral reading at exactly the
    # moments the indicator is most extreme.
    out = pd.Series(np.nan, index=close.index, dtype="float64")
    warm = avg_gain.notna() & avg_loss.notna()
    no_loss = warm & (avg_loss == 0.0)
    ok = warm & (avg_loss > 0.0)
    rs = avg_gain[ok] / avg_loss[ok]
    out[ok] = 100.0 - 100.0 / (1.0 + rs)
    out[no_loss] = np.where(avg_gain[no_loss] > 0.0, 100.0, 50.0)
    # Warm-up rows stay NaN and are dropped by the caller, never imputed.
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the baseline feature matrix from OHLCV bars.

    Args:
        df: OHLCV bars indexed by ascending date.

    Returns:
        DataFrame indexed by the same dates, one column per entry in
        ``BASELINE_FEATURES``.  Early rows contain NaN while the longest
        lookback (50-day SMA) warms up; callers are expected to drop them
        rather than have them imputed silently.
    """
    close = df["Close"].astype("float64")
    volume = df["Volume"].astype("float64") if "Volume" in df else pd.Series(
        1.0, index=df.index
    )
    log_close = np.log(close)
    out = pd.DataFrame(index=df.index)

    # Trailing log returns over several horizons: all differences of past closes.
    for n in (1, 5, 10, 20):
        out[f"ret_{n}"] = log_close.diff(n)

    # Realised volatility of daily log returns; `rolling` is backward-looking.
    daily = log_close.diff()
    for n in (10, 20):
        out[f"vol_{n}"] = daily.rolling(n).std()

    out["rsi_14"] = rsi(close, 14)

    # Moving-average ratios rather than raw levels, so the feature is
    # scale-free and comparable across a 30-year price range.
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["close_over_sma5"] = close / sma5 - 1.0
    out["close_over_sma20"] = close / sma20 - 1.0
    out["close_over_sma50"] = close / sma50 - 1.0
    out["sma5_over_sma20"] = sma5 / sma20 - 1.0

    out["volume_ratio_20"] = volume / volume.rolling(20).mean()

    return out[BASELINE_FEATURES]
