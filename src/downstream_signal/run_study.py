"""Orchestrator for the downstream predictive-signal study.

Pipeline::

    SPY bars ─┬─> baseline price features (causal, known at close of t)
              ├─> next-day up/down target (the only forward-looking quantity)
              └─> chart for each date t ──> YOLO detector ──> pattern features

    then: walk-forward logistic regression, baseline vs baseline+patterns

The pattern features come from the **detector's own predictions**, never from
the TA-Lib labels it was trained on. That distinction is the whole reason the
study is worth running: using the labels would only re-test whether TA-Lib
agrees with itself, whereas using the detector's output measures what a model
that had to *look at the picture* can actually extract, errors included.

**One asymmetry worth stating.** The classifier is fitted on dates running back
to 1993, and the detector *was* trained on charts from that early period. So the
pattern features the classifier learns from are sharper than the ones it meets
at test time, where the detector is genuinely out of sample. That shift makes
the ``with_patterns`` variant look slightly *worse* than a perfectly matched
setup would, not better -- so it cannot manufacture a positive result. The
evaluation window itself starts after the detector's validation period ends,
which is what keeps the reported comparison out-of-sample for both models.

Detector inference over ~8,400 charts takes several minutes on CPU, so the
resulting feature frame is cached to parquet and reused.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_pipeline.fetch_ohlc import fetch_ohlc
from src.detection.infer import PATTERN_FEATURES, detect, detections_to_features
from src.downstream_signal.backtest import run_comparison
from src.downstream_signal.features import build_features
from src.downstream_signal.labels import next_day_direction, next_day_return
from src.downstream_signal.model import assemble_matrix

logger = logging.getLogger(__name__)

CACHE_PATH = config.PROCESSED_DIR / "pattern_features.parquet"


def signal_image_paths(ticker: str = config.TICKER) -> pd.Series:
    """Map each chart's end-date to its rendered PNG path.

    The filename's date is the window's last candle, so the returned index is
    exactly the set of dates for which a same-day detector prediction exists.
    """
    root = config.PROCESSED_DIR / "signal" / ticker
    if not root.exists():
        raise FileNotFoundError(
            f"{root} missing -- run `python -m src.labeling.build_dataset` first"
        )
    paths, dates = [], []
    for p in sorted(root.glob(f"{ticker}_*.png")):
        dates.append(pd.Timestamp(p.stem.split("_")[-1]))
        paths.append(str(p))
    return pd.Series(paths, index=pd.DatetimeIndex(dates, name="Date"))


def build_pattern_features(
    ticker: str = config.TICKER,
    weights: str | None = None,
    batch: int = 32,
    force: bool = False,
) -> pd.DataFrame:
    """Run the detector over every chart and reduce each to a feature row.

    Args:
        ticker: which signal set to score.
        weights: optional local checkpoint; defaults to the Hub weights.
        batch: inference batch size.
        force: ignore the parquet cache and re-run inference.

    Returns:
        DataFrame indexed by date with the columns in ``PATTERN_FEATURES``.
    """
    if CACHE_PATH.exists() and not force:
        logger.info("using cached pattern features at %s", CACHE_PATH)
        return pd.read_parquet(CACHE_PATH)

    paths = signal_image_paths(ticker)
    logger.info("running the detector over %d charts (CPU: several minutes)", len(paths))
    rows = np.zeros((len(paths), len(PATTERN_FEATURES)), dtype="float64")
    for start in range(0, len(paths), batch):
        chunk = paths.iloc[start: start + batch].tolist()
        for offset, dets in enumerate(detect(chunk, local_path=weights, batch=batch)):
            rows[start + offset] = detections_to_features(dets)
        if start and start % (batch * 25) == 0:
            logger.info("  %d / %d", start, len(paths))

    frame = pd.DataFrame(rows, index=paths.index, columns=PATTERN_FEATURES)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE_PATH)
    logger.info("cached pattern features -> %s", CACHE_PATH)
    return frame


def run(weights: str | None = None, force: bool = False, cost_bps: float = 1.0) -> dict:
    """Execute the full study and write ``results/downstream_signal_comparison.json``."""
    bars = fetch_ohlc(config.TICKER)
    baseline = build_features(bars)
    target = next_day_direction(bars)
    fwd = next_day_return(bars)
    pattern = build_pattern_features(config.TICKER, weights=weights, force=force)

    X, y = assemble_matrix(baseline, pattern, target)
    logger.info("study matrix: %d rows x %d features, %s -> %s",
                len(X), X.shape[1], X.index.min().date(), X.index.max().date())

    report = run_comparison(
        X, y, fwd, cost_bps=cost_bps,
        out_path=config.RESULTS_DIR / "downstream_signal_comparison.json",
    )
    # Record how often the detector fired at all: a pattern feature that is
    # almost always zero cannot help, and the reader should be able to tell the
    # difference between "patterns do not predict" and "nothing was detected".
    fired = float((pattern.loc[X.index, "det_any"] > 0).mean())
    report["detector_activity"] = {
        "share_of_days_with_a_detection_on_the_last_candle": fired,
        "mean_detections_per_day": float(pattern.loc[X.index, "det_count"].mean()),
    }
    Path(config.RESULTS_DIR / "downstream_signal_comparison.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--force", action="store_true", help="re-run detector inference")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rep = run(args.weights, args.force, args.cost_bps)
    v = rep["variants"]
    print(f"\n{'metric':<24}{'baseline':>12}{'+patterns':>12}")
    for k in ("accuracy", "roc_auc", "brier", "strategy_return_net", "sharpe_net"):
        print(f"{k:<24}{v['baseline'][k]:>12.4f}{v['with_patterns'][k]:>12.4f}")
    b = rep["comparison"]["bootstrap"]
    print(f"\ndelta accuracy {b['delta_accuracy']:+.4f} "
          f"(95% CI {b['ci95_low']:+.4f} to {b['ci95_high']:+.4f})")
    print(f"McNemar p = {rep['comparison']['mcnemar']['p_value']:.4f}")
