"""Rule-based (weak) pattern labels from TA-Lib's CDLxxx candlestick functions.

**These are not human-verified ground truth.**  Every label in this project is
whatever TA-Lib's published heuristic says, and TA-Lib's definition of, say, a
Hammer will not always agree with what a discretionary trader would mark on the
same chart.  The detector can therefore only ever be as right as the rule it was
trained to imitate, and the downstream study must be read as "does TA-Lib's
notion of a pattern, seen through a detector, carry signal" -- not "do
candlestick patterns work".  This limitation is stated in the README too.

Two further properties of TA-Lib matter for how labels are placed:

* The functions are **causal**: the value at index *i* depends only on bars up
  to and including *i*.  Nothing here looks into the future.
* A multi-bar pattern is reported at the index where it **completes**, so a
  three-bar Morning Star flagged at *i* physically occupies bars *i-2 .. i*.
  ``config.PATTERN_SPAN`` encodes that, and the bounding box is drawn over the
  whole span rather than over the final bar alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from src import config

# Each entry maps a project class name to the TA-Lib function that detects it
# and a sign filter.  ``sign`` of +1/-1 keeps only positive/negative TA-Lib
# output (used to split the single CDLENGULFING function into the bullish and
# bearish classes); ``None`` keeps any non-zero output.
PATTERN_RULES: dict[str, tuple[str, int | None]] = {
    "Hammer": ("CDLHAMMER", None),
    "ShootingStar": ("CDLSHOOTINGSTAR", None),
    "BullishEngulfing": ("CDLENGULFING", 1),
    "BearishEngulfing": ("CDLENGULFING", -1),
    "MorningStar": ("CDLMORNINGSTAR", None),
    "EveningStar": ("CDLEVENINGSTAR", None),
    "Doji": ("CDLDOJI", None),
    "Harami": ("CDLHARAMI", None),
}


def label_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Run every configured CDLxxx rule over an OHLC series.

    Args:
        df: OHLC bars indexed by date, ascending, with Open/High/Low/Close
            columns.  Must be the *full* series, not a window: TA-Lib needs the
            preceding bars for its trend and averaging lookbacks, so labelling a
            20-bar slice in isolation gives different answers from labelling the
            whole history and slicing afterwards.

    Returns:
        One row per (bar, pattern) hit, with columns:
            ``position``   integer row offset into ``df``
            ``date``       the bar's date (pattern completion bar)
            ``pattern``    class name from ``config.CLASSES``
            ``class_id``   integer id matching ``config.CLASSES`` order
            ``span``       number of bars the pattern occupies
            ``direction``  +1 bullish / -1 bearish / 0 neutral prior
        Sorted by position then class_id.  A single bar may appear several
        times: a Doji that is also a Harami is genuinely both, and the detector
        is trained to output both boxes.
    """
    o = np.ascontiguousarray(df["Open"].to_numpy(dtype="float64"))
    h = np.ascontiguousarray(df["High"].to_numpy(dtype="float64"))
    l = np.ascontiguousarray(df["Low"].to_numpy(dtype="float64"))
    c = np.ascontiguousarray(df["Close"].to_numpy(dtype="float64"))

    rows: list[dict] = []
    for pattern, (fn_name, sign) in PATTERN_RULES.items():
        raw = getattr(talib, fn_name)(o, h, l, c)
        if sign is None:
            hits = np.flatnonzero(raw != 0)
        elif sign > 0:
            hits = np.flatnonzero(raw > 0)
        else:
            hits = np.flatnonzero(raw < 0)

        span = config.PATTERN_SPAN[pattern]
        # Drop hits whose span would run off the start of the series.
        hits = hits[hits >= span - 1]
        for i in hits:
            rows.append({
                "position": int(i),
                "date": df.index[i],
                "pattern": pattern,
                "class_id": config.CLASS_TO_ID[pattern],
                "span": span,
                "direction": config.PATTERN_DIRECTION[pattern],
            })

    if not rows:
        return pd.DataFrame(
            columns=["position", "date", "pattern", "class_id", "span", "direction"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["position", "class_id"])
        .reset_index(drop=True)
    )


def pattern_counts(labels: pd.DataFrame) -> pd.Series:
    """Instances per class, including classes with zero hits (as 0)."""
    counts = labels["pattern"].value_counts() if len(labels) else pd.Series(dtype=int)
    return counts.reindex(config.CLASSES, fill_value=0).astype(int)
