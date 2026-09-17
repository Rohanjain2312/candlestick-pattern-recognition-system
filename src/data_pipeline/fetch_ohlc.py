"""Download and cache daily OHLC bars from Yahoo Finance.

Bars are cached as parquet under ``data/raw`` so that every later stage of the
pipeline -- rendering, labeling, the downstream signal -- reads byte-identical
prices.  Re-running the pipeline against a freshly downloaded series would
otherwise shift split boundaries and make results irreproducible.

Prices are split/dividend adjusted (``auto_adjust=True``).  Adjustment scales
open, high, low and close for a given day by the same factor, so the *shape* of
each candle -- which is all the detector sees -- is unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

OHLC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker: str) -> Path:
    return config.RAW_DIR / f"{ticker}.parquet"


def fetch_ohlc(
    ticker: str = config.TICKER,
    start: str = config.START_DATE,
    end: str | None = config.END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV bars for ``ticker``, downloading only when needed.

    Args:
        ticker: Yahoo Finance symbol, e.g. ``"SPY"``.
        start: ISO date string for the first bar requested.
        end: ISO date string for the last bar, or None for "up to today".
        force_refresh: Ignore any cached parquet and re-download.

    Returns:
        DataFrame indexed by a tz-naive ``DatetimeIndex`` named ``Date``, with
        float columns Open/High/Low/Close/Volume, sorted ascending and with any
        row containing a null OHLC value dropped.

    Raises:
        RuntimeError: if Yahoo returns no rows for the symbol.
    """
    path = _cache_path(ticker)
    if path.exists() and not force_refresh:
        logger.info("loading cached bars for %s from %s", ticker, path)
        return pd.read_parquet(path)

    import yfinance as yf  # imported lazily: keeps `import config` cheap in tests

    logger.info("downloading %s from Yahoo Finance", ticker)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    df = _normalise(raw, ticker)
    if df.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {ticker!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("cached %d bars for %s (%s -> %s)", len(df), ticker,
                df.index[0].date(), df.index[-1].date())
    return df


def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's output into a plain OHLCV frame.

    yfinance returns a MultiIndex column frame (field, ticker) even for a single
    symbol in recent versions; older versions return flat columns.  Both shapes
    are handled here so the rest of the codebase never has to care.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=OHLC_COLUMNS)

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the ticker level; for a single-symbol download it is constant.
        level = 1 if ticker in df.columns.get_level_values(1) else 0
        df.columns = df.columns.droplevel(level)

    df = df[[c for c in OHLC_COLUMNS if c in df.columns]].astype("float64")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def fetch_many(tickers: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    """Fetch several tickers, skipping any that fail rather than aborting.

    A dead symbol in the detector's auxiliary universe should not take the whole
    dataset build down with it.
    """
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = fetch_ohlc(t, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately tolerant
            logger.warning("skipping %s: %s", t, exc)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = fetch_ohlc()
    print(frame.tail())
    print(f"\n{len(frame)} bars, {frame.index[0].date()} -> {frame.index[-1].date()}")
