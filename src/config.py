"""Single source of truth for every tunable in the project.

Every other module imports from here rather than hard-coding values, so that the
training pipeline, the evaluation pipeline and the Gradio demo cannot silently
drift apart.  In particular the chart *rendering* constants must be identical
between training and inference: a detector trained on 640x640 charts with a
20-candle window will not generalise to a chart drawn any other way.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
LABELS_DIR = PROCESSED_DIR / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------
# SPY is the downstream-signal ticker: the up/down classification study is run
# on SPY alone, so that the result is a statement about one liquid instrument
# rather than an average over a heterogeneous basket.
TICKER = "SPY"

# Additional tickers rendered *only* to enlarge the detector's training set.
# They never enter the downstream signal study.  Rare three-candle patterns
# (Morning/Evening Star) fire only a handful of times in SPY's history, which is
# not enough to train a detector head on; borrowing other liquid large caps is
# the cheapest fix that does not require manual annotation.
DETECTION_EXTRA_TICKERS = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "KO", "GE", "WMT", "PG", "T",
]

START_DATE = "1993-01-01"   # SPY's inception; yfinance clips to what exists.
END_DATE = None             # None -> today.

# --------------------------------------------------------------------------
# Chart rendering (must match between training, evaluation and the demo)
# --------------------------------------------------------------------------
WINDOW = 20        # candles per chart image
IMG_SIZE = 640     # square, in pixels
CHART_DPI = 100
CHART_FIGSIZE = (IMG_SIZE / CHART_DPI, IMG_SIZE / CHART_DPI)

# --------------------------------------------------------------------------
# Pattern classes
# --------------------------------------------------------------------------
# Order defines the integer class id used in the YOLO label files and in
# `data.yaml`.  Never reorder this list once weights have been trained.
CLASSES = [
    "Hammer",             # 0
    "ShootingStar",       # 1
    "BullishEngulfing",   # 2
    "BearishEngulfing",   # 3
    "MorningStar",        # 4
    "EveningStar",        # 5
    "Doji",               # 6
    "Harami",             # 7
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# How many consecutive candles each pattern occupies.  TA-Lib reports a pattern
# on the index where it *completes*, so a span of n means the pattern covers
# candles [i - n + 1 .. i].  This drives the bounding-box width.
PATTERN_SPAN = {
    "Hammer": 1,
    "ShootingStar": 1,
    "BullishEngulfing": 2,
    "BearishEngulfing": 2,
    "MorningStar": 3,
    "EveningStar": 3,
    "Doji": 1,
    "Harami": 2,
}

# Directional prior each pattern is traditionally read as.  Used only to build
# the downstream feature vector, never to train the detector.
PATTERN_DIRECTION = {
    "Hammer": 1,
    "ShootingStar": -1,
    "BullishEngulfing": 1,
    "BearishEngulfing": -1,
    "MorningStar": 1,
    "EveningStar": -1,
    "Doji": 0,
    "Harami": 0,
}

# Classes with fewer than this many training instances are dropped from the
# detector rather than being trained on a handful of examples.
MIN_INSTANCES_PER_CLASS = 30

# --------------------------------------------------------------------------
# Chronological splits
# --------------------------------------------------------------------------
# Splits are by date and never random.  A chart window ending on date t contains
# the previous WINDOW-1 bars, so two windows straddling a split boundary share
# pixels.  EMBARGO_BARS trading days are therefore discarded on each side of a
# boundary, which makes the val/test images strictly disjoint from train images.
TRAIN_END = "2014-12-31"
VAL_END = "2018-12-31"
# test = everything after VAL_END
EMBARGO_BARS = WINDOW

# Consecutive stride-1 windows overlap in 19 of 20 candles.  Training does not
# benefit from that redundancy -- it just makes epochs longer -- so the train
# split is subsampled.
DETECTION_STRIDE = 5
# Evaluation is different: subsampling there only throws away measurement points
# and widens the error bars on every per-class number, for no saving that
# matters (val/test are one ticker).  So val and test use every window.
EVAL_STRIDE = 1
# The downstream signal needs a prediction for every trading day.
SIGNAL_STRIDE = 1

# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
YOLO_MODEL = "yolo11n.pt"
YOLO_EPOCHS = 60
YOLO_BATCH = 32
CONF_THRESHOLD = 0.25   # detection confidence floor used downstream and in demo
IOU_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Hugging Face / GitHub identifiers
# --------------------------------------------------------------------------
HF_USER = "rohanjain2312"
HF_MODEL_REPO = f"{HF_USER}/candlestick-pattern-recognition-system-yolo"
HF_DATASET_REPO = f"{HF_USER}/candlestick-pattern-recognition-system-data"
HF_SPACE_REPO = f"{HF_USER}/candlestick-pattern-recognition-system-demo"
GITHUB_REPO = "candlestick-pattern-recognition-system"
WEIGHTS_FILENAME = "best.pt"
