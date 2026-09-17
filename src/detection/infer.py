"""Run the trained YOLO detector over chart images.

Weights are pulled from the Hugging Face model repo rather than a local file so
that the evaluation scripts, the notebook and the Gradio Space all score with
the identical checkpoint.  A local override exists for development, but the
published numbers should always come from the Hub copy.

The second job of this module is turning raw boxes into the single row of
features the downstream classifier consumes.  That conversion is where the
causality of the study is enforced: only detections covering the chart's
**right-most** candle describe a pattern that has completed as of date *t*.  A
Morning Star sitting in the middle of the window completed days ago and its
outcome is already in the price history; counting it as a signal for *t+1*
would be a subtle form of stale-signal double counting.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import numpy as np

from src import config
from src.data_pipeline.render_charts import LAST_CANDLE_X_NORM

logger = logging.getLogger(__name__)


def resolve_weights(local_path: str | Path | None = None) -> str:
    """Return a filesystem path to the detector weights.

    Args:
        local_path: explicit checkpoint to use.  When None, the environment
            variable ``YOLO_WEIGHTS`` is consulted, then the Hugging Face model
            repo named in ``config.HF_MODEL_REPO``.

    Returns:
        Path to a ``.pt`` checkpoint.

    Raises:
        FileNotFoundError: if an explicit path was given but does not exist.
    """
    if local_path is not None:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"weights not found: {p}")
        return str(p)

    env = os.environ.get("YOLO_WEIGHTS")
    if env and Path(env).exists():
        return env

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=config.HF_MODEL_REPO,
        filename=config.WEIGHTS_FILENAME,
        token=os.environ.get("HF_TOKEN"),
    )


@functools.lru_cache(maxsize=2)
def load_model(local_path: str | None = None):
    """Load and cache the Ultralytics model (loading it is slow; reuse it)."""
    from ultralytics import YOLO

    weights = resolve_weights(local_path)
    logger.info("loading detector from %s", weights)
    return YOLO(weights)


def detect(
    images,
    conf: float = config.CONF_THRESHOLD,
    local_path: str | None = None,
    batch: int = 32,
) -> list[list[dict]]:
    """Detect patterns in one or more chart images.

    Args:
        images: a single image (path or HxWx3 array) or a list of them.
        conf: confidence floor; detections below it are discarded.
        local_path: optional explicit weights path.
        batch: inference batch size.

    Returns:
        One list of detections per input image, each detection a dict with
        ``class_id``, ``pattern``, ``confidence`` and ``xyxyn`` (the box in
        normalised x1,y1,x2,y2 form).  Order of the outer list matches the
        order of ``images``.
    """
    model = load_model(local_path)
    single = not isinstance(images, (list, tuple))
    items = [images] if single else list(images)

    out: list[list[dict]] = []
    for i in range(0, len(items), batch):
        chunk = items[i: i + batch]
        results = model.predict(chunk, conf=conf, verbose=False)
        for res in results:
            dets = []
            boxes = res.boxes
            if boxes is not None and len(boxes):
                xyxyn = boxes.xyxyn.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for b, c, p in zip(xyxyn, cls, confs):
                    dets.append({
                        "class_id": int(c),
                        "pattern": config.CLASSES[int(c)]
                        if 0 <= int(c) < config.NUM_CLASSES else str(c),
                        "confidence": float(p),
                        "xyxyn": [float(v) for v in b],
                    })
            out.append(dets)
    return out


def covers_last_candle(det: dict, x_norm: float = LAST_CANDLE_X_NORM) -> bool:
    """Whether a detection's box spans the chart's most recent candle."""
    x1, _, x2, _ = det["xyxyn"]
    return x1 <= x_norm <= x2


PATTERN_FEATURES = (
    [f"det_{c}" for c in config.CLASSES]
    + ["det_any", "det_bullish", "det_bearish", "det_count"]
)


def detections_to_features(dets: list[dict]) -> np.ndarray:
    """Collapse one chart's detections into the downstream feature row.

    Only detections covering the right-most candle are used -- see the module
    docstring.  Per class the value is the highest confidence among such
    detections (0.0 if the class was not detected), which keeps the feature
    continuous rather than a hard 0/1 and lets the classifier discount
    low-confidence hits on its own.

    Returns:
        1-D float array aligned with ``PATTERN_FEATURES``.
    """
    per_class = np.zeros(config.NUM_CLASSES, dtype="float64")
    current = [d for d in dets if covers_last_candle(d)]
    for d in current:
        cid = d["class_id"]
        if 0 <= cid < config.NUM_CLASSES:
            per_class[cid] = max(per_class[cid], d["confidence"])

    bullish = sum(
        per_class[i] for i, c in enumerate(config.CLASSES)
        if config.PATTERN_DIRECTION[c] > 0
    )
    bearish = sum(
        per_class[i] for i, c in enumerate(config.CLASSES)
        if config.PATTERN_DIRECTION[c] < 0
    )
    extras = np.array(
        [float(per_class.max()), bullish, bearish, float((per_class > 0).sum())],
        dtype="float64",
    )
    return np.concatenate([per_class, extras])
