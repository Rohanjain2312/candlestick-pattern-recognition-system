"""Tests for candle-span -> normalised YOLO box conversion.

Boxes that stray outside [0, 1] are rejected by Ultralytics at load time, and
zero-height boxes destabilise its loss, so both are asserted explicitly.
"""

import pytest

from src import config
from src.data_pipeline.render_charts import PixelMapper, candle_center_norm
from src.labeling.bbox_utils import MIN_BOX_PX, pattern_bbox, to_yolo_line


@pytest.fixture
def mapper() -> PixelMapper:
    """A mapper with the same geometry render_window produces: 20 candles across
    640px with 0.8-slot margins, and a price range of 100 to 110."""
    width = height = config.IMG_SIZE
    x_span = config.WINDOW - 1 + 2 * 0.8
    x_scale = width / x_span
    x0 = 0.8 * x_scale
    # Price 100 maps to the bottom row and 110 to the top, matching how
    # render_window fits the window's low/high to the frame.
    y_scale = -height / 10.0
    y0 = height - y_scale * 100.0
    return PixelMapper(width, height, x0, x_scale, y0, y_scale)


def test_box_is_normalised_and_inside_frame(mapper):
    """All four values in [0,1] with positive extent; Ultralytics rejects anything else at load time."""
    cx, cy, w, h = pattern_bbox(mapper, high=108.0, low=102.0,
                                first_candle=5, last_candle=7)
    for v in (cx, cy, w, h):
        assert 0.0 <= v <= 1.0
    assert cx - w / 2 >= -1e-9 and cx + w / 2 <= 1.0 + 1e-9
    assert cy - h / 2 >= -1e-9 and cy + h / 2 <= 1.0 + 1e-9
    assert w > 0 and h > 0


def test_box_centre_tracks_the_candle_span(mapper):
    """A single-candle box should sit on that candle's x position."""
    for i in (0, 9, config.WINDOW - 1):
        cx, _, _, _ = pattern_bbox(mapper, 108.0, 102.0, i, i)
        assert cx == pytest.approx(candle_center_norm(i), abs=0.01)


def test_wider_span_gives_wider_box(mapper):
    """A three-candle pattern must produce a wider box than a one-candle pattern."""
    _, _, w1, _ = pattern_bbox(mapper, 108.0, 102.0, 10, 10)
    _, _, w2, _ = pattern_bbox(mapper, 108.0, 102.0, 8, 10)
    _, _, w3, _ = pattern_bbox(mapper, 108.0, 102.0, 7, 10)
    assert w1 < w2 < w3


def test_taller_price_range_gives_taller_box(mapper):
    """Box height must track the price range the pattern actually spans."""
    _, _, _, h_small = pattern_bbox(mapper, 104.0, 103.5, 10, 10)
    _, _, _, h_big = pattern_bbox(mapper, 109.0, 101.0, 10, 10)
    assert h_big > h_small


def test_flat_candle_gets_a_minimum_height(mapper):
    """A Doji's high and low can be nearly identical; the box must not collapse."""
    _, _, _, h = pattern_bbox(mapper, 105.0, 105.0, 10, 10)
    assert h * mapper.height >= MIN_BOX_PX - 1e-6


def test_edge_candles_stay_in_frame(mapper):
    """The first and last candles sit against the border; their boxes must still be clipped inside it."""
    for i in (0, config.WINDOW - 1):
        cx, cy, w, h = pattern_bbox(mapper, 110.0, 100.0, i, i)
        assert 0.0 <= cx - w / 2 and cx + w / 2 <= 1.0
        assert 0.0 <= cy - h / 2 and cy + h / 2 <= 1.0


def test_reversed_span_rejected(mapper):
    """A reversed span means the caller mixed up first/last; fail loudly rather than emit a negative-width box."""
    with pytest.raises(ValueError):
        pattern_bbox(mapper, 108.0, 102.0, first_candle=9, last_candle=5)


def test_yolo_line_format(mapper):
    """One line, five fields, class id first, geometry normalised."""
    box = pattern_bbox(mapper, 108.0, 102.0, 5, 6)
    line = to_yolo_line(3, box)
    parts = line.split()
    assert len(parts) == 5
    assert parts[0] == "3"
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_rgb_to_bgr_conversion_swaps_channels():
    """Regression guard for a bug that would only ever show up in the demo.

    Ultralytics takes BGR arrays and trains on cv2-loaded (BGR) PNGs, while the
    renderer emits RGB. If this conversion is dropped, file-path inference stays
    correct and array inference quietly degrades -- the hardest kind of bug to
    notice, because nothing errors.
    """
    import numpy as np

    from src.detection.infer import _as_bgr

    rgb = np.zeros((4, 4, 3), dtype="uint8")
    rgb[..., 0] = 10   # R
    rgb[..., 1] = 20   # G
    rgb[..., 2] = 30   # B
    bgr = _as_bgr(rgb)
    assert bgr[0, 0].tolist() == [30, 20, 10]
    assert bgr.shape == rgb.shape and bgr.flags["C_CONTIGUOUS"]
    # Non-array inputs (paths) must pass through untouched.
    assert _as_bgr("some/chart.png") == "some/chart.png"
