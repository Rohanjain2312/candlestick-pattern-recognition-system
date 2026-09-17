"""Gradio demo: draw a chart, run the detector on it, and show what it found.

Runs on a free CPU Space. Three things are on show:

1. **Detection** -- live chart for any ticker/date, with the detector's boxes on
   it, next to what TA-Lib's rule says about the same bars. Seeing where the two
   disagree is the fastest way to understand what "weak labels" means.
2. **Results** -- the out-of-sample numbers, including the finding on whether
   pattern features improve next-day prediction at all.
3. **Patterns** -- plain-language descriptions.

The chart is rendered by the *same* ``render_window`` the training data used.
A demo that drew charts even slightly differently would be showing the detector
an input distribution it was never trained on, and the boxes would quietly get
worse for reasons no one could see.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from pattern_descriptions import as_markdown, describe
from src import config
from src.data_pipeline.render_charts import render_window
from src.detection.infer import covers_last_candle, detect

RESULTS_DIR = Path(__file__).parent / "results"
BOX_COLOR = {"bullish": (34, 139, 84), "bearish": (200, 40, 40), "neutral": (90, 90, 90)}


# --------------------------------------------------------------------------
# Data + inference
# --------------------------------------------------------------------------
def load_bars(ticker: str, end_date: str) -> pd.DataFrame:
    """Fetch enough daily bars to draw a window ending on or before ``end_date``.

    Pulls a generous buffer because weekends, holidays and listing gaps mean a
    calendar range of N days yields fewer than N bars.
    """
    import yfinance as yf

    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=int(config.WINDOW * 4) + 60)
    raw = yf.download(ticker, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                      interval="1d", auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise gr.Error(f"No data for {ticker!r} up to {end_date}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()
    if len(df) < config.WINDOW:
        raise gr.Error(
            f"Only {len(df)} bars available for {ticker}; need {config.WINDOW}."
        )
    return df.iloc[-config.WINDOW:]


def _draw_boxes(image: np.ndarray, dets: list[dict]) -> np.ndarray:
    """Outline each detection, thicker for the one on the most recent candle."""
    out = image.copy()
    h, w = out.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d["xyxyn"]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        colour = BOX_COLOR[describe(d["pattern"])["bias"]]
        t = 4 if covers_last_candle(d) else 2
        for a, b in ((y1, y1 + t), (y2 - t, y2)):
            out[max(0, a):min(h, b), max(0, x1):min(w, x2)] = colour
        for a, b in ((x1, x1 + t), (x2 - t, x2)):
            out[max(0, y1):min(h, y2), max(0, a):min(w, b)] = colour
    return out


def talib_opinion(ticker: str, end_date: str) -> pd.DataFrame:
    """What the rule that generated the training labels says about these bars.

    Shown beside the detector's output so the gap between "the rule" and "the
    model that learned the rule from pictures" is visible rather than asserted.
    Returns an empty frame if TA-Lib is unavailable in the Space image.
    """
    try:
        import yfinance as yf
        from src.labeling.talib_labeler import label_patterns
    except Exception:
        return pd.DataFrame(columns=["date", "pattern"])

    end = pd.Timestamp(end_date)
    raw = yf.download(ticker, start=(end - pd.Timedelta(days=400)).date(),
                      end=(end + pd.Timedelta(days=1)).date(),
                      interval="1d", auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["date", "pattern"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    df = raw[["Open", "High", "Low", "Close"]].dropna().sort_index()
    labels = label_patterns(df)
    window_dates = set(df.index[-config.WINDOW:])
    hit = labels[labels["date"].isin(window_dates)]
    return (hit[["date", "pattern"]]
            .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
            .sort_values("date")
            .reset_index(drop=True))


def analyse(ticker: str, end_date: str, conf: float):
    """Main callback: render, detect, and format everything for the UI."""
    ticker = (ticker or "SPY").strip().upper()
    window = load_bars(ticker, end_date)
    image, _ = render_window(window)

    try:
        dets = detect(image, conf=conf)[0]
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(
            "Could not load the detector weights from the Hugging Face model "
            f"repo ({config.HF_MODEL_REPO}). Details: {exc}"
        ) from exc

    boxed = _draw_boxes(image, dets)
    rows = [{
        "pattern": d["pattern"],
        "confidence": round(d["confidence"], 3),
        "on latest candle": "yes" if covers_last_candle(d) else "no",
        "bias": describe(d["pattern"])["bias"],
    } for d in sorted(dets, key=lambda d: -d["confidence"])]
    table = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["pattern", "confidence", "on latest candle", "bias"]
    )

    latest = [d for d in dets if covers_last_candle(d)]
    first, last = window.index[0].date(), window.index[-1].date()
    if latest:
        names = ", ".join(sorted({d["pattern"] for d in latest}))
        headline = (f"### {ticker}: {names} on the most recent candle "
                    f"({last})\n\nWindow: {first} to {last}.")
    else:
        headline = (f"### {ticker}: nothing detected on the most recent candle "
                    f"({last})\n\nWindow: {first} to {last}. "
                    "Boxes further left are patterns that completed earlier in "
                    "this window; they are not a signal for the next day.")
    headline += ("\n\n*The detector reproduces TA-Lib's rules, which are not "
                 "human-verified ground truth. See the Results tab for whether "
                 "any of this predicts anything.*")
    return boxed, table, headline, talib_opinion(ticker, end_date)


# --------------------------------------------------------------------------
# Results tabs
# --------------------------------------------------------------------------
def _load(name: str) -> dict | None:
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def detection_table() -> pd.DataFrame:
    rep = _load("detection_metrics.json")
    if not rep:
        return pd.DataFrame({"note": ["Detection metrics not published yet."]})
    rows = []
    for cls, v in rep["per_class"].items():
        fmt = lambda x: round(x, 3) if isinstance(x, (int, float)) else "n/a"
        rows.append({"class": cls, "test instances": v.get("support", 0),
                     "precision": fmt(v.get("precision")), "recall": fmt(v.get("recall")),
                     "mAP@50": fmt(v.get("mAP50"))})
    agg = rep["aggregate"]
    rows.append({"class": "— ALL (aggregate) —", "test instances": "",
                 "precision": round(agg["precision"], 3), "recall": round(agg["recall"], 3),
                 "mAP@50": round(agg["mAP50"], 3)})
    return pd.DataFrame(rows)


def signal_table() -> pd.DataFrame:
    rep = _load("downstream_signal_comparison.json")
    if not rep:
        return pd.DataFrame({"note": ["Signal study not published yet."]})
    b, p = rep["variants"]["baseline"], rep["variants"]["with_patterns"]
    keys = [("accuracy", "Accuracy"), ("roc_auc", "ROC AUC"), ("brier", "Brier (lower better)"),
            ("strategy_return_net", "Strategy return, net"), ("sharpe_net", "Sharpe, net")]
    return pd.DataFrame([
        {"metric": label, "baseline": round(b[k], 4), "+ patterns": round(p[k], 4),
         "difference": round(p[k] - b[k], 4)}
        for k, label in keys
    ])


def signal_verdict() -> str:
    rep = _load("downstream_signal_comparison.json")
    if not rep:
        return "*Signal study not published yet.*"
    c, ev = rep["comparison"], rep["evaluation"]
    bs, mc = c["bootstrap"], c["mcnemar"]
    significant = mc["p_value"] < 0.05 and not (bs["ci95_low"] <= 0 <= bs["ci95_high"])
    verdict = (
        "**The pattern features did NOT measurably improve next-day prediction.** "
        "The accuracy difference is within noise, so on this evidence a detected "
        "candlestick pattern adds nothing to a plain price-feature baseline."
        if not significant else
        "**The pattern features changed accuracy by a statistically detectable "
        "margin.** The size of the effect still matters more than its "
        "significance — read the interval below before concluding anything."
    )
    return f"""### Does detecting a pattern help predict tomorrow?

{verdict}

| | |
|---|---|
| Out-of-sample period | {ev['eval_start']} to {ev['eval_end']} ({ev['n_days']} trading days) |
| Accuracy difference (+patterns − baseline) | **{bs['delta_accuracy']:+.4f}** |
| 95% bootstrap interval | {bs['ci95_low']:+.4f} to {bs['ci95_high']:+.4f} |
| McNemar p-value | {mc['p_value']:.3f} |
| Protocol | {ev['protocol']} |

A negative or near-zero result is reported here exactly as found. The point of
the project was to test the claim, not to confirm it.
"""


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
INTRO = f"""# Candlestick pattern detector

A YOLO11n object detector that finds {config.NUM_CLASSES} candlestick patterns in a
{config.WINDOW}-candle daily chart — plus an honest test of whether its detections
help predict the next day's direction.

**Labels are rule-based.** Training boxes came from TA-Lib's `CDLxxx` functions,
not from human annotation, so this model imitates a published heuristic.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Candlestick pattern detector",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)

        with gr.Tab("Detect"):
            with gr.Row():
                ticker = gr.Textbox(label="Ticker", value="SPY", scale=2)
                end_date = gr.Textbox(
                    label="Window ends on (YYYY-MM-DD)",
                    value=str(pd.Timestamp.today().date()), scale=2)
                conf = gr.Slider(0.05, 0.9, value=config.CONF_THRESHOLD, step=0.05,
                                 label="Confidence threshold", scale=2)
            run = gr.Button("Detect patterns", variant="primary")
            summary = gr.Markdown()
            with gr.Row():
                chart = gr.Image(label="Detections (thick box = on the latest candle)",
                                 type="numpy", height=520)
                with gr.Column():
                    found = gr.Dataframe(label="What the detector found", wrap=True)
                    rule = gr.Dataframe(
                        label="What TA-Lib's rule says about the same bars", wrap=True)
            gr.Markdown(
                "*Thick boxes sit on the most recent candle and are the only ones the "
                "downstream study treats as a signal for the next day. Thinner boxes "
                "completed earlier in the window — that information is already in the "
                "price history.*"
            )
            run.click(analyse, [ticker, end_date, conf], [chart, found, summary, rule])

        with gr.Tab("Results"):
            gr.Markdown(signal_verdict())
            gr.Markdown("### Next-day direction: baseline vs baseline + detected patterns")
            gr.Dataframe(value=signal_table(), wrap=True)
            gr.Markdown(
                "### Detection accuracy on the held-out test split\n\n"
                "Per class, because the classes are very unevenly represented — a "
                "single aggregate number would hide which patterns actually work."
            )
            gr.Dataframe(value=detection_table(), wrap=True)

        with gr.Tab("Patterns"):
            gr.Markdown(as_markdown())

        gr.Markdown(
            f"[Code](https://github.com/{config.HF_USER}/{config.GITHUB_REPO}) · "
            f"[Model](https://huggingface.co/{config.HF_MODEL_REPO}) · "
            f"[Dataset](https://huggingface.co/datasets/{config.HF_DATASET_REPO})\n\n"
            "Not investment advice. Nothing here is a trading recommendation."
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1")
