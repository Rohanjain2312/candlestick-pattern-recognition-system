"""Held-out detection metrics, reported per class rather than as one number.

Aggregate mAP on this dataset is close to meaningless on its own. The classes
are wildly imbalanced -- Doji outnumbers Evening Star by more than an order of
magnitude -- so a macro-averaged mAP is dominated by whichever rare class the
model happens to do worst on, and can look poor while the model is reliable on
the patterns that actually occur often. Both views are therefore written out,
and the README shows the per-class table.

The test split is {ticker}-only and chronologically after every training chart,
with an embargo gap, so these numbers describe genuinely unseen data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src import config
from src.detection.dataset_yaml import write_yaml
from src.detection.infer import resolve_weights

logger = logging.getLogger(__name__)


def label_instance_counts(split: str = "test") -> dict[str, int]:
    """Count ground-truth instances per class in a split's label files.

    Reported alongside the metrics so a reader can immediately see which
    per-class scores rest on a handful of examples.
    """
    counts = {c: 0 for c in config.CLASSES}
    for txt in (config.LABELS_DIR / split).glob("*.txt"):
        for line in txt.read_text().splitlines():
            if line.strip():
                cid = int(line.split()[0])
                if 0 <= cid < config.NUM_CLASSES:
                    counts[config.CLASSES[cid]] += 1
    return counts


def evaluate(
    weights: str | None = None,
    split: str = "test",
    out_path: str | Path | None = None,
) -> dict:
    """Run Ultralytics validation and write the metrics report.

    Args:
        weights: checkpoint path; defaults to the published Hub weights.
        split: which split to score.
        out_path: destination JSON, default ``results/detection_metrics.json``.

    Returns:
        The report dict, with ``aggregate``, ``per_class`` and ``support``.
    """
    from ultralytics import YOLO

    yaml_path = write_yaml()
    model = YOLO(resolve_weights(weights))
    logger.info("validating on the %s split", split)
    m = model.val(data=str(yaml_path), split=split, imgsz=config.IMG_SIZE, verbose=False)

    support = label_instance_counts(split)
    per_class: dict[str, dict | None] = {}
    for i, name in enumerate(config.CLASSES):
        try:
            p, r, ap50, ap = m.class_result(i)
            per_class[name] = {
                "precision": float(p), "recall": float(r),
                "mAP50": float(ap50), "mAP50_95": float(ap),
                "support": support[name],
            }
        except Exception:  # noqa: BLE001 - class simply absent from the split
            per_class[name] = {"precision": None, "recall": None, "mAP50": None,
                               "mAP50_95": None, "support": support[name]}

    report = {
        "split": split,
        "n_images": len(list((config.IMAGES_DIR / split).glob("*.png"))),
        "aggregate": {
            "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map),
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
        },
        "per_class": per_class,
        "caveats": [
            "Labels are TA-Lib CDLxxx rules, not human annotation; these metrics "
            "measure agreement with that rule, not with a trader's judgement.",
            "Splits are chronological with an embargo gap; never random.",
            f"The {split} split contains {config.TICKER} only.",
        ],
    }

    out_path = Path(out_path) if out_path else config.RESULTS_DIR / "detection_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rep = evaluate(args.weights, args.split)
    print(json.dumps(rep["aggregate"], indent=2))
    print(f"\n{'class':<20}{'support':>9}{'P':>8}{'R':>8}{'mAP50':>8}")
    for c, v in rep["per_class"].items():
        fmt = lambda x: f"{x:>8.3f}" if isinstance(x, float) else f"{'-':>8}"
        print(f"{c:<20}{v['support']:>9}{fmt(v['precision'])}{fmt(v['recall'])}{fmt(v['mAP50'])}")
