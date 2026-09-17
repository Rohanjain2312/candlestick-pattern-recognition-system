"""Build the YOLO detection dataset: chart PNGs plus normalised label files.

This is where rendering (``render_charts``), rule-based labels
(``talib_labeler``) and box geometry (``bbox_utils``) are combined into the
on-disk layout Ultralytics expects::

    data/processed/images/{train,val,test}/TICKER_YYYYMMDD.png
    data/processed/labels/{train,val,test}/TICKER_YYYYMMDD.txt

Three design points that are load-bearing for the honesty of the final result:

**Causality.**  A window ending on date *t* is drawn from bars *t-19 .. t* only.
TA-Lib is run once over the whole price series (it needs the long lookback) and
the resulting hits are then *sliced* into windows -- never recomputed on a
20-bar fragment, which would give different answers.

**Every pattern in frame gets a box.**  A pattern is emitted if its entire span
falls inside the window, not just if it completes on the last candle.  The
alternative -- labelling only the right-most candle -- would show the detector
the same candle as a positive in one image and as unlabelled background in the
next, which is contradictory supervision.

**Chronological splits with an embargo.**  Consecutive windows share 19 of 20
candles, so a window ending just after a split boundary is nearly the same image
as one ending just before it.  The first ``config.EMBARGO_BARS`` windows of the
val and test splits are therefore discarded, making the three splits disjoint in
pixels and not merely in end-date.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src import config
from src.data_pipeline.fetch_ohlc import fetch_ohlc
from src.data_pipeline.render_charts import render_window
from src.labeling.bbox_utils import pattern_bbox, to_yolo_line
from src.labeling.talib_labeler import label_patterns

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def split_ranges(df: pd.DataFrame) -> dict[str, range]:
    """Return the window-end positions belonging to each chronological split.

    Args:
        df: the full OHLC series for one ticker.

    Returns:
        Mapping split name -> range of integer end positions.  Ranges exclude
        the embargo zone after each boundary, so no val/test image shares a
        candle with a train image.
    """
    dates = df.index
    train_end = int((dates <= pd.Timestamp(config.TRAIN_END)).sum()) - 1
    val_end = int((dates <= pd.Timestamp(config.VAL_END)).sum()) - 1
    last = len(df) - 1
    first = config.WINDOW - 1
    emb = config.EMBARGO_BARS

    return {
        "train": range(first, train_end + 1),
        "val": range(max(first, train_end + 1 + emb), val_end + 1),
        "test": range(max(first, val_end + 1 + emb), last + 1),
    }


def _labels_in_window(
    by_position: dict[int, list[dict]], end: int, window: int
) -> list[dict]:
    """Collect pattern hits whose whole span lies inside the window.

    ``end`` is the position of the window's right-most candle.  A hit recorded
    at position *p* with span *s* occupies positions ``p-s+1 .. p``; it is kept
    only if that entire range sits within ``end-window+1 .. end``.
    """
    start = end - window + 1
    out: list[dict] = []
    for p in range(start, end + 1):
        for hit in by_position.get(p, ()):
            if p - hit["span"] + 1 >= start:
                out.append(hit)
    return out


def build_ticker(
    ticker: str,
    strides: dict[str, int],
    splits_wanted: tuple[str, ...] = SPLITS,
) -> Counter:
    """Render and label every window for one ticker.

    Args:
        ticker: symbol to process.
        strides: step between window end positions, per split.  Train is
            subsampled to keep epochs short; val and test are not, because
            discarding evaluation windows only widens the error bars.
        splits_wanted: which splits to emit; the auxiliary tickers only
            contribute to ``("train",)``.

    Returns:
        Counter keyed by ``"<split>/<pattern>"`` plus ``"<split>/_images"``,
        used to report per-class balance once all tickers are done.
    """
    df = fetch_ohlc(ticker)
    labels = label_patterns(df)
    by_position: dict[int, list[dict]] = {}
    for rec in labels.to_dict("records"):
        by_position.setdefault(rec["position"], []).append(rec)

    highs = df["High"].to_numpy(dtype="float64")
    lows = df["Low"].to_numpy(dtype="float64")
    ranges = split_ranges(df)
    stats: Counter = Counter()

    for split in splits_wanted:
        img_dir = config.IMAGES_DIR / split
        lbl_dir = config.LABELS_DIR / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for end in ranges[split][::strides[split]]:
            start = end - config.WINDOW + 1
            win = df.iloc[start: end + 1]
            hits = _labels_in_window(by_position, end, config.WINDOW)

            stem = f"{ticker}_{df.index[end].strftime('%Y%m%d')}"
            image, mapper = render_window(win, img_dir / f"{stem}.png")

            lines: list[str] = []
            for hit in hits:
                last_c = hit["position"] - start          # index within window
                first_c = last_c - hit["span"] + 1
                lo = float(lows[start + first_c: start + last_c + 1].min())
                hi = float(highs[start + first_c: start + last_c + 1].max())
                box = pattern_bbox(mapper, hi, lo, first_c, last_c)
                lines.append(to_yolo_line(hit["class_id"], box))
                stats[f"{split}/{hit['pattern']}"] += 1

            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[f"{split}/_images"] += 1

    logger.info("%s done: %s", ticker, dict(stats))
    return stats


def build_signal_set(ticker: str = config.TICKER) -> int:
    """Render a stride-1 chart for every bar of the downstream-signal ticker.

    The detection dataset is subsampled (``config.DETECTION_STRIDE``) because
    overlapping windows add little to training.  The downstream study, by
    contrast, needs one detector prediction per trading day, so these charts are
    rendered separately and kept out of the detection splits entirely.

    Returns:
        Number of images written.
    """
    df = fetch_ohlc(ticker)
    out_dir = config.PROCESSED_DIR / "signal" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for end in range(config.WINDOW - 1, len(df)):
        win = df.iloc[end - config.WINDOW + 1: end + 1]
        render_window(win, out_dir / f"{ticker}_{df.index[end].strftime('%Y%m%d')}.png")
        n += 1
    logger.info("signal set: %d images for %s", n, ticker)
    return n


def main() -> None:
    """Build the whole dataset and print the per-class balance table.

    Tickers are processed in parallel because rendering is CPU-bound and each
    ticker is fully independent. A ticker that fails is logged and skipped
    rather than aborting a build that is minutes in.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=config.DETECTION_STRIDE,
                        help="train-split stride; val/test always use EVAL_STRIDE")
    parser.add_argument("--only", default=None,
                        help="comma-separated tickers, for a partial rebuild")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-signal", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # SPY contributes to all three splits: the held-out detection metrics should
    # be measured on the same instrument the downstream study runs on.  The
    # auxiliary tickers exist only to give the rare three-candle classes enough
    # training examples, so they contribute to train alone -- otherwise the test
    # split would be reporting accuracy on names the study never trades.
    jobs = [(config.TICKER, SPLITS)]
    jobs += [(t, ("train",)) for t in config.DETECTION_EXTRA_TICKERS]
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        jobs = [j for j in jobs if j[0] in wanted]

    strides = {
        "train": args.stride,
        "val": config.EVAL_STRIDE,
        "test": config.EVAL_STRIDE,
    }

    total: Counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(build_ticker, t, strides, sp): t for t, sp in jobs
        }
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                total.update(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.error("ticker %s failed: %s", ticker, exc)

    if not args.skip_signal:
        build_signal_set()

    report = {
        split: {
            "images": total.get(f"{split}/_images", 0),
            "instances": {c: total.get(f"{split}/{c}", 0) for c in config.CLASSES},
        }
        for split in SPLITS
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'class':<20}" + "".join(f"{s:>10}" for s in SPLITS))
    for c in config.CLASSES:
        print(f"{c:<20}" + "".join(f"{report[s]['instances'][c]:>10}" for s in SPLITS))
    print(f"{'IMAGES':<20}" + "".join(f"{report[s]['images']:>10}" for s in SPLITS))


if __name__ == "__main__":
    main()
