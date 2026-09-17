"""Render fixed-length OHLC windows to candlestick PNGs, plus the pixel mapping.

Two things matter here and both are easy to get subtly wrong.

1. **One rendering style, everywhere.**  The detector only ever sees pixels, so
   the chart drawn during training, the chart scored during evaluation and the
   chart the Gradio demo shows a user must come out of this same function with
   the same style, size, candle width and absence of axes.  Anything else is an
   unannounced distribution shift.

2. **Boxes come from matplotlib, not from arithmetic.**  Rather than guessing
   where candle *i* landed in the image, we render first and then ask the axes
   for its own data->display transform.  That is exact by construction and
   survives any future change to figure padding or candle width.

The chart is deliberately bare: no gridlines, no axis ticks, no volume panel,
no title.  Those are chrome that the detector would have to learn to ignore.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display; must be set before pyplot is imported

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from src import config

# A flat, high-contrast style.  Up candles are drawn hollow-free green and down
# candles red, with the same edge colour as the body so the detector sees a
# solid block rather than an outline that thins out at small sizes.
# Blank space left around the candles, in candle-slot units (x) and as a
# fraction of the window's price range (y).
X_MARGIN = 0.8
Y_MARGIN_FRAC = 0.04

CHART_STYLE = mpf.make_mpf_style(
    base_mpf_style="charles",
    marketcolors=mpf.make_marketcolors(
        up="#26A69A",
        down="#EF5350",
        edge={"up": "#26A69A", "down": "#EF5350"},
        wick={"up": "#26A69A", "down": "#EF5350"},
        volume="in",
    ),
    gridstyle="",
    facecolor="#FFFFFF",
    figcolor="#FFFFFF",
    edgecolor="#FFFFFF",
)


@dataclass(frozen=True)
class PixelMapper:
    """Maps (candle index, price) pairs to pixel coordinates of a rendered chart.

    Attributes:
        width: image width in pixels.
        height: image height in pixels.
        x0, x_scale: pixel_x = x0 + x_scale * candle_index
        y0, y_scale: pixel_y = y0 + y_scale * price  (y_scale is negative,
            because matplotlib's display origin is bottom-left while image rows
            count downwards from the top).

    The mapping is affine because both chart axes are linear, so two probe
    points per axis are enough to recover it exactly.
    """

    width: int
    height: int
    x0: float
    x_scale: float
    y0: float
    y_scale: float

    def to_pixels(self, candle_index: float, price: float) -> tuple[float, float]:
        """Return the (x, y) pixel of a point, in image coordinates."""
        return (self.x0 + self.x_scale * candle_index,
                self.y0 + self.y_scale * price)


def _build_mapper(ax, width: int, height: int) -> PixelMapper:
    """Recover the affine data->image-pixel transform from a drawn axes."""
    # Probe two points per axis and solve; transData already accounts for the
    # axes position inside the figure and for the data limits mplfinance chose.
    (px0, py0), (px1, py1) = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    x_scale = px1 - px0
    y_scale_display = py1 - py0
    # matplotlib display y grows upward; image rows grow downward.
    return PixelMapper(
        width=width,
        height=height,
        x0=px0,
        x_scale=x_scale,
        y0=height - py0,
        y_scale=-y_scale_display,
    )


def render_window(
    window: pd.DataFrame,
    out_path: str | Path | None = None,
) -> tuple[np.ndarray, PixelMapper]:
    """Render one OHLC window as a bare candlestick chart.

    Args:
        window: exactly ``config.WINDOW`` rows of OHLC data, indexed by date and
            sorted ascending.  Candle *i* of the image is row *i* of this frame.
        out_path: if given, the PNG is written here (parent dirs are created).

    Returns:
        ``(image, mapper)`` where ``image`` is an ``(H, W, 3)`` uint8 RGB array
        and ``mapper`` converts candle/price coordinates into pixels of that
        exact image.

    Raises:
        ValueError: if ``window`` does not have ``config.WINDOW`` rows.
    """
    if len(window) != config.WINDOW:
        raise ValueError(
            f"expected {config.WINDOW} rows, got {len(window)}"
        )

    fig, axes = mpf.plot(
        window,
        type="candle",
        style=CHART_STYLE,
        figsize=config.CHART_FIGSIZE,
        volume=False,
        axisoff=True,
        returnfig=True,
        tight_layout=True,
        scale_padding=0.0,
        update_width_config={"candle_linewidth": 0.9, "candle_width": 0.62},
    )
    ax = axes[0]
    # Let the plot area fill the whole canvas: no white margin for the detector
    # to waste capacity on, and a simpler pixel mapping.
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    # Explicit limits, set after mplfinance has drawn, so that the first and
    # last candles are fully inside the frame instead of being sliced in half by
    # the image border -- a half-candle is a different shape to the detector.
    ax.set_xlim(-X_MARGIN, config.WINDOW - 1 + X_MARGIN)
    low, high = float(window["Low"].min()), float(window["High"].max())
    span = max(high - low, 1e-9)
    ax.set_ylim(low - Y_MARGIN_FRAC * span, high + Y_MARGIN_FRAC * span)
    fig.set_dpi(config.CHART_DPI)
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    mapper = _build_mapper(ax, width, height)

    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = buf.reshape(height, width, 4)[:, :, :3].copy()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(out_path, image)

    plt.close(fig)
    return image, mapper


def iter_windows(
    df: pd.DataFrame,
    stride: int = 1,
    window: int = config.WINDOW,
):
    """Yield ``(end_position, end_date, window_frame)`` for each chart window.

    A window ending at position *p* covers rows ``p - window + 1 .. p`` and
    therefore contains no information after ``df.index[p]``.  That is the whole
    basis of the no-leakage claim downstream: the chart for date *t* is built
    only from bars up to and including *t*, and is used to predict *t+1*.
    """
    for end in range(window - 1, len(df), stride):
        yield end, df.index[end], df.iloc[end - window + 1: end + 1]


def candle_center_norm(candle_index: int, window: int = config.WINDOW) -> float:
    """Normalised x position (0-1) of a candle's centre in a rendered chart.

    Because ``render_window`` pins the x limits to ``[-X_MARGIN, window-1+X_MARGIN]``
    and the axes fill the whole canvas, this is a pure function of the geometry
    constants -- no image required.  The downstream feature extractor uses it to
    ask "does this detection cover the most recent candle?" without having to
    re-render the chart or carry a mapper around.
    """
    return (candle_index + X_MARGIN) / (window - 1 + 2 * X_MARGIN)


LAST_CANDLE_X_NORM = candle_center_norm(config.WINDOW - 1)
