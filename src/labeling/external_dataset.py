"""Optional: fold a third-party YOLO-format candlestick dataset into train.

**Off by default, and deliberately so.**  The original plan allowed bootstrapping
extra images from a public Roboflow Universe candlestick set.  That was a hedge
against having too few examples of the rare three-candle patterns.  Rendering an
auxiliary basket of tickers (``config.DETECTION_EXTRA_TICKERS``) solved that
problem outright -- every class clears the minimum-instance floor -- and mixing
in an external set now costs more than it buys:

* **Style shift.**  Those charts are drawn with different colours, candle
  widths, gridlines and axis chrome.  The detector would spend capacity learning
  that two visual dialects mean the same thing, on a task where our own charts
  are the only ones it will ever be scored on.
* **Taxonomy drift.**  Their "Hammer" is annotated by whoever built that set,
  under their own definition.  Ours is exactly TA-Lib's ``CDLHAMMER``.  Merging
  the two silently makes the label definition a blend of both, which would make
  the honest claim "the detector imitates TA-Lib" no longer true.

The importer is kept because it is genuinely useful if you later *want* that
trade-off -- for example to test robustness to chart style.  Supply an already
downloaded dataset directory; nothing here reaches out to a third-party API, so
no extra credentials are required.

Usage::

    python -m src.labeling.external_dataset --src /path/to/roboflow-export \\
        --mapping '{"hammer": "Hammer", "doji": "Doji"}'
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

# Class names seen in public candlestick sets, lower-cased, mapped onto this
# project's taxonomy.  Anything not listed here is dropped rather than guessed.
DEFAULT_MAPPING: dict[str, str] = {
    "hammer": "Hammer",
    "inverted hammer": "Hammer",
    "shooting star": "ShootingStar",
    "shootingstar": "ShootingStar",
    "bullish engulfing": "BullishEngulfing",
    "bearish engulfing": "BearishEngulfing",
    "morning star": "MorningStar",
    "evening star": "EveningStar",
    "doji": "Doji",
    "harami": "Harami",
    "bullish harami": "Harami",
    "bearish harami": "Harami",
}


def _read_source_names(src: Path) -> list[str]:
    """Read the source dataset's class-name list from its data.yaml."""
    yaml_path = next(src.glob("*.yaml"), None) or next(src.glob("*.yml"), None)
    if yaml_path is None:
        raise FileNotFoundError(f"no data.yaml found under {src}")
    import yaml  # ultralytics dependency; only needed on this path

    doc = yaml.safe_load(yaml_path.read_text())
    names = doc["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def import_external(
    src: str | Path,
    mapping: dict[str, str] | None = None,
    prefix: str = "ext",
) -> dict[str, int]:
    """Copy an external YOLO dataset into this project's *train* split.

    Args:
        src: root of the external dataset; expected to contain ``*/images`` and
            ``*/labels`` subtrees and a ``data.yaml`` naming its classes.
        mapping: external class name (lower-case) -> one of ``config.CLASSES``.
            Unmapped classes are dropped, and an image left with no boxes is
            still copied, serving as a background example.
        prefix: filename prefix so external images can never collide with, or be
            mistaken for, rendered ones.

    Returns:
        Counts per destination class, plus ``"_images"`` and ``"_dropped"``.

    Note:
        Writes only into the train split.  Validation and test must stay purely
        our own charts, or the reported metrics stop describing the system the
        demo actually runs.
    """
    src = Path(src)
    mapping = {k.lower(): v for k, v in (mapping or DEFAULT_MAPPING).items()}
    source_names = _read_source_names(src)

    img_out = config.IMAGES_DIR / "train"
    lbl_out = config.LABELS_DIR / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {c: 0 for c in config.CLASSES}
    stats["_images"] = stats["_dropped"] = 0

    for img_path in sorted(src.rglob("images/*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        lines_out: list[str] = []
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            src_id = int(parts[0])
            src_name = source_names[src_id].lower() if src_id < len(source_names) else ""
            dest = mapping.get(src_name)
            if dest is None:
                stats["_dropped"] += 1
                continue
            lines_out.append(" ".join([str(config.CLASS_TO_ID[dest]), *parts[1:]]))
            stats[dest] += 1

        stem = f"{prefix}_{img_path.stem}"
        shutil.copy2(img_path, img_out / f"{stem}{img_path.suffix}")
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
        stats["_images"] += 1

    logger.info("imported %d external images: %s", stats["_images"], stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="root of the external dataset")
    parser.add_argument("--mapping", default=None, help="JSON object of name overrides")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(import_external(args.src, json.loads(args.mapping) if args.mapping else None))
