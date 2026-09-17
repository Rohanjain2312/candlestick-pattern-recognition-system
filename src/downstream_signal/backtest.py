"""Chronological walk-forward comparison of the two classifier variants.

Design choices, and why each one is there:

* **Expanding window, annual refit.**  On each 1 January the model is refitted
  on everything up to that date and then used, untouched, for the next twelve
  months.  This mirrors how the model would actually have been run, and it never
  fits on a day it later scores.
* **No random splits anywhere.**  Financial series are autocorrelated; shuffling
  would let near-identical neighbouring days sit on both sides of the split and
  inflate every score.
* **A significance test, not just two numbers.**  Two accuracies that differ by
  half a percentage point on ~1,700 days are indistinguishable from noise.
  McNemar's test on the paired predictions and a paired bootstrap interval on
  the accuracy difference are reported so the write-up can say whether any gap
  is real, and the README can state the honest answer either way.
* **Trading costs are charged.**  A long/flat rule that flips position daily
  looks far better gross than net.  Costs are a parameter and the gross number
  is reported alongside, so nothing is hidden.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from src import config
from src.downstream_signal.model import VARIANTS, FitResult, make_model

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
DEFAULT_COST_BPS = 1.0  # per position change, one-way, in basis points


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    columns: list[str],
    name: str,
    eval_start: str,
    refit_freq: str = "YS",
) -> FitResult:
    """Fit repeatedly on the past and predict the following period.

    Args:
        X: full feature matrix indexed by date.
        y: aligned binary target.
        columns: subset of ``X``'s columns this variant may use.
        name: variant label, carried into the result.
        eval_start: first date to produce an out-of-sample prediction for.
        refit_freq: pandas offset alias for refit boundaries ("YS" = each
            January 1st).

    Returns:
        ``FitResult`` holding one out-of-sample probability per evaluated date.

    Raises:
        ValueError: if no evaluation dates remain after ``eval_start``.
    """
    Xc = X[columns]
    boundaries = pd.date_range(
        start=pd.Timestamp(eval_start), end=X.index.max(), freq=refit_freq
    )
    boundaries = (
        pd.DatetimeIndex([pd.Timestamp(eval_start), *boundaries]).unique().sort_values()
    )

    dates: list[pd.Timestamp] = []
    probs: list[float] = []
    for i, start in enumerate(boundaries):
        stop = boundaries[i + 1] if i + 1 < len(boundaries) else None
        train_mask = X.index < start
        test_mask = X.index >= start
        if stop is not None:
            test_mask &= X.index < stop
        if train_mask.sum() < 250 or test_mask.sum() == 0:
            continue
        model = make_model()
        model.fit(Xc[train_mask], y[train_mask])
        p = model.predict_proba(Xc[test_mask])[:, 1]
        dates.extend(X.index[test_mask])
        probs.extend(p)

    if not dates:
        raise ValueError(f"no evaluation rows for {name} after {eval_start}")

    idx = pd.DatetimeIndex(dates)
    return FitResult(
        name=name,
        dates=idx,
        y_true=y.loc[idx].to_numpy(),
        y_prob=np.asarray(probs, dtype="float64"),
    )


@dataclass
class VariantMetrics:
    """Scores for one variant over the out-of-sample period."""

    name: str
    n_days: int
    accuracy: float
    base_rate: float
    accuracy_over_base: float
    roc_auc: float
    f1: float
    brier: float
    strategy_return_gross: float
    strategy_return_net: float
    buy_hold_return: float
    sharpe_net: float
    buy_hold_sharpe: float
    n_trades: int
    # Share of days the model said "up". A value near 1.0 means the strategy is
    # essentially buy-and-hold, and its return says nothing about the model.
    share_predicted_up: float


def _strategy_stats(
    y_prob: np.ndarray, fwd_returns: np.ndarray, cost_bps: float
) -> tuple[float, float, float, int]:
    """Long-when-p>0.5, flat otherwise; returns gross, net, Sharpe, trade count.

    This is an illustration of what the classifier's edge would have been worth,
    not a trading recommendation: it ignores slippage, financing, taxes and the
    fact that the model was selected after seeing this period exists.
    """
    position = (y_prob >= 0.5).astype("float64")
    gross = position * fwd_returns
    turnover = np.abs(np.diff(position, prepend=0.0))
    cost = turnover * (cost_bps / 10_000.0)
    net = gross - cost
    sharpe = (
        float(np.mean(net) / np.std(net) * np.sqrt(TRADING_DAYS))
        if np.std(net) > 0 else 0.0
    )
    return (
        float(np.prod(1.0 + gross) - 1.0),
        float(np.prod(1.0 + net) - 1.0),
        sharpe,
        int(turnover.sum()),
    )


def score_variant(
    res: FitResult, fwd_returns: pd.Series, cost_bps: float = DEFAULT_COST_BPS
) -> VariantMetrics:
    """Compute every reported metric for one variant."""
    fwd = fwd_returns.loc[res.dates].to_numpy()
    base = float(np.mean(res.y_true))
    acc = float(accuracy_score(res.y_true, res.y_pred))
    gross, net, sharpe, trades = _strategy_stats(res.y_prob, fwd, cost_bps)
    # Buy-and-hold on the same days, same units. Quoting a strategy Sharpe
    # without this one invites a comparison against nothing.
    bh_sharpe = (
        float(np.mean(fwd) / np.std(fwd) * np.sqrt(TRADING_DAYS))
        if np.std(fwd) > 0 else 0.0
    )
    return VariantMetrics(
        name=res.name,
        n_days=int(len(res.y_true)),
        accuracy=acc,
        base_rate=base,
        accuracy_over_base=acc - max(base, 1.0 - base),
        roc_auc=float(roc_auc_score(res.y_true, res.y_prob)),
        f1=float(f1_score(res.y_true, res.y_pred, zero_division=0)),
        brier=float(brier_score_loss(res.y_true, res.y_prob)),
        strategy_return_gross=gross,
        strategy_return_net=net,
        buy_hold_return=float(np.prod(1.0 + fwd) - 1.0),
        sharpe_net=sharpe,
        buy_hold_sharpe=bh_sharpe,
        n_trades=trades,
        share_predicted_up=float(np.mean(res.y_pred)),
    )


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test on two classifiers' paired right/wrong outcomes.

    Only the discordant pairs carry information: days where A was right and B
    wrong (``b01``) versus the reverse (``b10``).  Uses the exact binomial test,
    which is valid however few discordant pairs there are.

    Returns:
        Dict with the two discordant counts and a two-sided p-value.
    """
    from scipy.stats import binomtest

    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b01 = int(np.sum(a_ok & ~b_ok))   # baseline right, patterns wrong
    b10 = int(np.sum(~a_ok & b_ok))   # patterns right, baseline wrong
    n = b01 + b10
    p = float(binomtest(b10, n, 0.5).pvalue) if n else 1.0
    return {
        "baseline_right_patterns_wrong": b01,
        "patterns_right_baseline_wrong": b10,
        "discordant_pairs": n,
        "p_value": p,
    }


def bootstrap_accuracy_delta(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_pat: np.ndarray,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap confidence interval for (patterns - baseline) accuracy.

    Resamples days, not predictions, keeping the two models' outcomes paired so
    the interval reflects the difference rather than each model's own spread.
    """
    rng = np.random.default_rng(seed)
    base_ok = (pred_base == y_true).astype("float64")
    pat_ok = (pred_pat == y_true).astype("float64")
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = pat_ok[idx].mean() - base_ok[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_accuracy": float(pat_ok.mean() - base_ok.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "prob_delta_positive": float(np.mean(deltas > 0)),
    }


def run_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    fwd_returns: pd.Series,
    eval_start: str | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    out_path=None,
) -> dict:
    """Run both variants walk-forward and assemble the comparison report.

    ``eval_start`` defaults to the day after the detector's own validation
    period ends, so every scored day is out-of-sample for *both* models: the
    detector never trained on those charts and the classifier never fitted on
    those rows.
    """
    eval_start = eval_start or str(
        (pd.Timestamp(config.VAL_END) + pd.Timedelta(days=1)).date()
    )
    results = {
        name: walk_forward(X, y, cols, name, eval_start)
        for name, cols in VARIANTS.items()
    }
    # Score both on exactly the same days, so the comparison is paired.
    common = results["baseline"].dates.intersection(results["with_patterns"].dates)
    for r in results.values():
        keep = r.dates.isin(common)
        r.dates, r.y_true, r.y_prob = r.dates[keep], r.y_true[keep], r.y_prob[keep]

    metrics = {n: asdict(score_variant(r, fwd_returns, cost_bps)) for n, r in results.items()}
    y_true = results["baseline"].y_true
    report = {
        "evaluation": {
            "ticker": config.TICKER,
            "eval_start": eval_start,
            "eval_end": str(common.max().date()),
            "n_days": int(len(common)),
            "cost_bps_per_turn": cost_bps,
            "protocol": "expanding-window walk-forward, annual refit, no shuffling",
        },
        "variants": metrics,
        "comparison": {
            "mcnemar": mcnemar(
                y_true, results["baseline"].y_pred, results["with_patterns"].y_pred
            ),
            "bootstrap": bootstrap_accuracy_delta(
                y_true, results["baseline"].y_pred, results["with_patterns"].y_pred
            ),
            "delta_roc_auc": metrics["with_patterns"]["roc_auc"] - metrics["baseline"]["roc_auc"],
        },
    }

    if out_path:
        from pathlib import Path
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", p)
    return report
