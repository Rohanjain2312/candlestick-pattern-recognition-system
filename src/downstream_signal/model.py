"""The two classifiers whose difference is the entire point of the project.

* **baseline**  -- plain price features only.
* **+patterns** -- the same features plus the detector's own predictions on the
  chart ending that day.

Both are logistic regression on standardised inputs.  A deliberately simple,
well-understood model is the right choice here: the question is whether the
pattern features *carry information*, and a high-variance model would make the
answer depend mostly on how it was tuned.  The two variants differ only in their
input columns -- same estimator, same regularisation, same fitting procedure --
so any difference in score is attributable to the features and nothing else.

The pattern features are the **detector's predictions**, not the TA-Lib labels
used to train it.  Feeding the labels back in would be circular and would test
nothing except that TA-Lib is self-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.detection.infer import PATTERN_FEATURES
from src.downstream_signal.features import BASELINE_FEATURES

VARIANTS: dict[str, list[str]] = {
    "baseline": list(BASELINE_FEATURES),
    "with_patterns": list(BASELINE_FEATURES) + list(PATTERN_FEATURES),
}


def make_model(seed: int = 0) -> Pipeline:
    """Standardiser + L2 logistic regression.

    Standardisation is inside the pipeline so it is re-fitted on each
    walk-forward training window using only that window's statistics; scaling on
    the full series first would leak test-period mean and variance backwards.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )),
    ])


@dataclass
class FitResult:
    """Out-of-sample predictions for one variant over the evaluation period."""

    name: str
    dates: pd.DatetimeIndex
    y_true: np.ndarray
    y_prob: np.ndarray

    @property
    def y_pred(self) -> np.ndarray:
        """Hard class predictions at the natural 0.5 threshold.

        The threshold is deliberately not tuned. Choosing it to maximise the
        score on the very period being reported would be a hidden form of
        fitting to the test set.
        """
        return (self.y_prob >= 0.5).astype(int)


def assemble_matrix(
    baseline: pd.DataFrame,
    pattern: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Join features and target on date and drop rows that are not fully formed.

    Args:
        baseline: output of ``features.build_features``.
        pattern: one row per date, columns ``PATTERN_FEATURES``.
        target: output of ``labels.next_day_direction``.

    Returns:
        ``(X, y)`` sharing an index, with warm-up NaNs removed.  Dates present
        in only one of the three inputs are dropped, so a day with no rendered
        chart simply does not take part rather than being imputed.
    """
    X = baseline.join(pattern, how="inner")
    joined = X.join(target.rename("_y"), how="inner").dropna()
    return joined.drop(columns=["_y"]), joined["_y"].astype(int)
