"""Turn a labelled candle span into a normalised YOLO bounding box.

The box is derived from the *rendered* chart's own coordinate transform (see
``render_charts.PixelMapper``) rather than from assumptions about where
matplotlib put things, so it stays correct if the chart style ever changes.

YOLO's label format is one line per object:

    <class_id> <x_centre> <y_centre> <width> <height>

with all four geometry values normalised to [0, 1] by image width/height.
"""

from __future__ import annotations

from src import config
from src.data_pipeline.render_charts import PixelMapper

# Horizontal half-width of a candle slot, in candle-index units.  Slightly wider
# than the drawn candle body so wicks and edges are enclosed rather than grazed.
CANDLE_HALF_WIDTH = 0.48
# Breathing room added to the top and bottom of a box, in pixels.  Without it a
# Doji -- whose body is a single horizontal line -- yields an almost zero-height
# box that YOLO's loss handles badly.
VERTICAL_PAD_PX = 4.0
# Floor on box height, in pixels, for the same reason.
MIN_BOX_PX = 10.0


def pattern_bbox(
    mapper: PixelMapper,
    high: float,
    low: float,
    first_candle: int,
    last_candle: int,
) -> tuple[float, float, float, float]:
    """Compute a normalised YOLO box for one pattern occurrence.

    Args:
        mapper: transform returned alongside the rendered image.
        high: highest high across the pattern's candles.
        low: lowest low across the pattern's candles.
        first_candle: index of the pattern's first candle *within the window*
            (0 = leftmost candle of the image).
        last_candle: index of the pattern's last candle within the window.

    Returns:
        ``(x_centre, y_centre, width, height)``, each normalised to [0, 1] and
        clipped to the image.  Width/height are always strictly positive.

    Raises:
        ValueError: if the candle indices are reversed.
    """
    if last_candle < first_candle:
        raise ValueError(
            f"last_candle ({last_candle}) precedes first_candle ({first_candle})"
        )

    x_left, _ = mapper.to_pixels(first_candle - CANDLE_HALF_WIDTH, 0.0)
    x_right, _ = mapper.to_pixels(last_candle + CANDLE_HALF_WIDTH, 0.0)
    _, y_top = mapper.to_pixels(0.0, high)
    _, y_bottom = mapper.to_pixels(0.0, low)

    y_top -= VERTICAL_PAD_PX
    y_bottom += VERTICAL_PAD_PX
    if y_bottom - y_top < MIN_BOX_PX:
        centre = 0.5 * (y_top + y_bottom)
        y_top, y_bottom = centre - MIN_BOX_PX / 2, centre + MIN_BOX_PX / 2

    # Clip to the frame: a pattern on the first or last candle can otherwise
    # extend a few pixels past the border.
    x_left = max(0.0, min(x_left, mapper.width))
    x_right = max(0.0, min(x_right, mapper.width))
    y_top = max(0.0, min(y_top, mapper.height))
    y_bottom = max(0.0, min(y_bottom, mapper.height))

    w = max(x_right - x_left, 1.0) / mapper.width
    h = max(y_bottom - y_top, 1.0) / mapper.height
    cx = (x_left + x_right) / 2.0 / mapper.width
    cy = (y_top + y_bottom) / 2.0 / mapper.height

    # Guard against a box whose centre plus half-extent leaves the image, which
    # Ultralytics rejects at load time.
    cx = min(max(cx, w / 2), 1.0 - w / 2)
    cy = min(max(cy, h / 2), 1.0 - h / 2)
    return cx, cy, w, h


def to_yolo_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    """Format one YOLO label line with six-decimal geometry."""
    cx, cy, w, h = box
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
