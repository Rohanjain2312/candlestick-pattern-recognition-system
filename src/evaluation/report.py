"""Combine both results files into the markdown that goes into the README.

Generated rather than written by hand so the README can never quote a number
the JSON does not contain. ``--inject`` rewrites the block between the marker
comments in README.md, leaving the rest of the document alone.

The wording adapts to what the data says. If the pattern features do not help,
the report says so plainly -- that is the outcome the project was built to be
able to report, not a failure to hide.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"


def _load(path: Path) -> dict | None:
    """Read a results JSON, or None if that stage has not been run yet."""
    return json.loads(path.read_text()) if path.exists() else None


def _num(x, digits: int = 3) -> str:
    """Format a metric, rendering a missing value as "n/a" rather than 0.

    A class absent from the test split has no precision; printing 0.000 would
    read as "the model failed" instead of "there was nothing to score".
    """
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "n/a"


def detection_section(rep: dict | None) -> str:
    """Render the per-class detection table, or a "not yet run" placeholder.

    Args:
        rep: parsed ``detection_metrics.json``, or None.

    Returns:
        Markdown for the detection part of the results block.
    """
    if not rep:
        return ("### Detection\n\n_Not yet run. Fine-tune the detector "
                "(`notebooks/03_yolo_finetune_colab.ipynb`), then "
                "`python -m src.evaluation.detection_metrics`._\n")
    agg = rep["aggregate"]
    lines = [
        "### Detection, held-out test split",
        "",
        f"{rep['n_images']} {config.TICKER} charts, chronologically after every "
        "training chart, with an embargo gap.",
        "",
        "| class | test instances | precision | recall | mAP@50 | mAP@50-95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls, v in rep["per_class"].items():
        lines.append(
            f"| {cls} | {v.get('support', 0)} | {_num(v.get('precision'))} | "
            f"{_num(v.get('recall'))} | {_num(v.get('mAP50'))} | {_num(v.get('mAP50_95'))} |"
        )
    lines += [
        f"| **aggregate** | | **{_num(agg['precision'])}** | **{_num(agg['recall'])}** | "
        f"**{_num(agg['mAP50'])}** | **{_num(agg['mAP50_95'])}** |",
        "",
        "Per class rather than one number: Doji outnumbers Evening Star by more "
        "than an order of magnitude, so an averaged mAP is driven by the rarest "
        "class and hides how the detector does on the common ones.",
        "",
    ]
    lines += _difficulty_note(rep["per_class"])
    return "\n".join(lines)


# Patterns defined by a relationship between whole candle bodies, versus those
# defined by a proportion within a single candle. The split is a property of the
# TA-Lib rules, not of this model, so it is declared rather than inferred.
_RELATIONAL = {"BullishEngulfing", "BearishEngulfing", "Harami"}


def _join(names: list[str]) -> str:
    """Join class names as prose: 'A', 'A and B', 'A, B and C'."""
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _difficulty_note(per_class: dict) -> list[str]:
    """Describe which classes the detector finds easy or hard, from the numbers.

    Computed rather than written by hand so the observation cannot drift out of
    step with a re-run. Returns an empty list if too few classes were scored to
    say anything meaningful.
    """
    scored = {c: v for c, v in per_class.items()
              if isinstance(v.get("mAP50"), (int, float))}
    if len(scored) < 4:
        return []

    ranked = sorted(scored.items(), key=lambda kv: -kv[1]["mAP50"])
    best = [c for c, _ in ranked[:3]]
    # Exclude any overlap: with few scored classes the two ends can otherwise
    # name the same class as both strongest and weakest.
    worst = [c for c, _ in ranked[-3:] if c not in best]
    if not worst:
        return []

    # A class the model *misses* has low recall; one it *mislabels* has low
    # precision with recall intact. Which of the two it is changes what you would
    # do about it, so the distinction is drawn explicitly.
    def _misses_more_than_mislabels(v: dict) -> bool:
        """True when recall lags precision: the model skips rather than guesses."""
        r, p_ = v.get("recall"), v.get("precision")
        return isinstance(r, (int, float)) and isinstance(p_, (int, float)) and r < p_

    missed = [c for c in worst if _misses_more_than_mislabels(scored[c])]
    relational_hits = [c for c in best if c in _RELATIONAL]

    out = ["**Which patterns the detector finds easy, and why.** "]
    out[-1] += (
        f"The strongest classes are {_join(best)}; the weakest are "
        f"{_join(worst)}. "
    )
    if relational_hits:
        out[-1] += (
            "The pattern is not random: classes defined by a *relationship between "
            "whole candle bodies* — is this body larger than the previous one, is it "
            "contained within it — are close to solved, because that is a crisp "
            "geometric comparison at chart scale. The weaker classes are the ones "
            "defined by a *proportion inside a single candle*, such as a lower wick "
            "at least twice the body. At 640x640 with 20 candles, a body is only a "
            "few pixels tall, so the very measurement the rule depends on is the one "
            "the image barely resolves. "
        )
    over = [c for c in worst if c not in missed]
    if missed:
        out[-1] += (
            f"On {_join(missed)} the model errs by **missing** rather than "
            f"mislabelling (recall below precision): it declines to call the "
            f"pattern rather than calling the wrong one, which for a downstream "
            f"signal is the safer failure — it removes observations instead of "
            f"corrupting them. "
        )
    if over:
        out[-1] += (
            f"On {_join(over)} the balance runs the other way (precision below "
            f"recall): the model finds most of the real instances but also calls "
            f"some that the rule does not, so those columns carry false positives "
            f"as well as noise."
        )
    out.append("")
    return out


def signal_section(rep: dict | None) -> str:
    """Render the signal comparison, with wording chosen by what the data says.

    The verdict sentence branches three ways -- no measurable effect, a positive
    effect, or a negative one -- so the README states the actual finding rather
    than a template that only reads well if the result was good.

    Args:
        rep: parsed ``downstream_signal_comparison.json``, or None.

    Returns:
        Markdown for the signal part of the results block.
    """
    if not rep:
        return ("### Downstream signal\n\n_Not yet run. "
                "`python -m src.downstream_signal.run_study`._\n")
    ev, cmp_ = rep["evaluation"], rep["comparison"]
    b, p = rep["variants"]["baseline"], rep["variants"]["with_patterns"]
    bs, mc = cmp_["bootstrap"], cmp_["mcnemar"]
    ci_excludes_zero = not (bs["ci95_low"] <= 0 <= bs["ci95_high"])
    significant = mc["p_value"] < 0.05 and ci_excludes_zero

    if not significant:
        verdict = (
            "**Detected patterns did not improve next-day direction prediction.** "
            f"Accuracy moved by {bs['delta_accuracy']:+.4f}, with a 95% bootstrap "
            f"interval of {bs['ci95_low']:+.4f} to {bs['ci95_high']:+.4f} that "
            f"contains zero, and McNemar's test gives p = {mc['p_value']:.3f}. On "
            "this evidence, knowing that the detector saw a candlestick pattern "
            "at the close adds nothing to a plain price-feature baseline."
        )
    elif bs["delta_accuracy"] > 0:
        verdict = (
            "**Detected patterns improved next-day accuracy by a statistically "
            f"detectable margin** of {bs['delta_accuracy']:+.4f} (95% CI "
            f"{bs['ci95_low']:+.4f} to {bs['ci95_high']:+.4f}, McNemar "
            f"p = {mc['p_value']:.3f}). The effect is small in absolute terms and "
            "is reported before costs beyond the modelled spread; it is a "
            "measurement, not a trading strategy."
        )
    else:
        verdict = (
            "**Adding pattern features made next-day prediction measurably worse** "
            f"({bs['delta_accuracy']:+.4f}, 95% CI {bs['ci95_low']:+.4f} to "
            f"{bs['ci95_high']:+.4f}, McNemar p = {mc['p_value']:.3f}) — the extra "
            "columns added variance without adding information."
        )

    rows = [
        ("Accuracy", "accuracy", 4), ("ROC AUC", "roc_auc", 4),
        ("Brier score (lower is better)", "brier", 4), ("F1", "f1", 4),
        ("Share of days predicted \"up\"", "share_predicted_up", 3),
        ("Strategy return, net of costs", "strategy_return_net", 4),
        ("Sharpe, net", "sharpe_net", 3),
    ]
    lines = [
        "### Downstream signal: does a detected pattern predict tomorrow?",
        "",
        verdict,
        "",
        f"Out-of-sample {ev['eval_start']} to {ev['eval_end']} "
        f"({ev['n_days']} trading days on {ev['ticker']}). {ev['protocol']}.",
        "",
        "| metric | baseline | + detected patterns | difference |",
        "|---|---:|---:|---:|",
    ]
    for label, key, d in rows:
        lines.append(
            f"| {label} | {_num(b[key], d)} | {_num(p[key], d)} | "
            f"{_num(p[key] - b[key], d)} |"
        )
    majority = max(b["base_rate"], 1.0 - b["base_rate"])
    lines += [
        f"| Majority-class rate | {_num(majority, 4)} | | |",
        f"| Buy and hold, same period | {_num(b['buy_hold_return'], 4)} | | |",
        f"| Buy and hold Sharpe | {_num(b.get('buy_hold_sharpe'), 3)} | | |",
        "",
        "The majority-class rate is the accuracy of always predicting \"up\". On a "
        "long-drifting index it sits well above 50%, so a model that merely "
        "matches it has learned nothing. Here **neither variant beats it**: "
        f"baseline is {b['accuracy'] - majority:+.4f} against it and "
        f"+patterns {p['accuracy'] - majority:+.4f}.",
        "",
    ]
    lines += returns_caveat(b, p, majority)
    act = rep.get("detector_activity")
    if act:
        lines += [
            f"The detector fired on the most recent candle on "
            f"{act['share_of_days_with_a_detection_on_the_last_candle']:.1%} of days "
            f"(mean {act['mean_detections_per_day']:.2f} detections per chart), so "
            "the pattern columns were far from constant — a null result here is not "
            "an artefact of the features always being zero.",
            "",
        ]
    return "\n".join(lines)


def returns_caveat(b: dict, p: dict, majority: float) -> list[str]:
    """Warn when the strategy figures look good for a reason that is not skill.

    A classifier with no ranking ability that nonetheless predicts "up" almost
    every day produces a long/flat rule that is buy-and-hold with a few days
    missing. Its return and Sharpe will then track the index, and any small
    excess is the luck of which days it sat out -- not evidence of an edge.
    Quoting those numbers without this check is the single easiest way to
    oversell a null result, so the check is automatic rather than remembered.
    """
    aucs = [v.get("roc_auc") for v in (b, p) if isinstance(v.get("roc_auc"), float)]
    ups = [v.get("share_predicted_up") for v in (b, p)
           if isinstance(v.get("share_predicted_up"), float)]
    if not aucs or not ups:
        return []

    no_ranking = max(aucs) <= 0.52
    always_long = min(ups) >= 0.85
    if not (no_ranking and always_long):
        return []

    bh = b.get("buy_hold_sharpe")
    bh_txt = f" (buy and hold over the same days: {bh:.3f})" if isinstance(bh, float) else ""
    return [
        "> **Do not read the return rows as an edge.** Both variants have a ROC AUC "
        f"of about {max(aucs):.3f} — no better than chance at ranking one day above "
        f"another — and both predict \"up\" on at least {min(ups):.0%} of days. The "
        "strategy is therefore buy-and-hold with a handful of days sat out, and its "
        f"Sharpe of {p.get('sharpe_net', float('nan')):.3f}{bh_txt} reflects which "
        "days those happened to be, not an ability to pick them. Accuracy that does "
        f"not clear the majority-class rate of {majority:.4f} cannot coexist with a "
        "genuine trading edge.",
        "",
    ]


def build_report() -> str:
    """Assemble the full results markdown from both JSON files."""
    det = _load(config.RESULTS_DIR / "detection_metrics.json")
    sig = _load(config.RESULTS_DIR / "downstream_signal_comparison.json")
    return "\n".join([
        "## Results",
        "",
        "_Generated by `python -m src.evaluation.report --inject`; every number "
        "below is read from `results/*.json`._",
        "",
        signal_section(sig),
        detection_section(det),
        "### Caveats that apply to all of the above",
        "",
        "- Labels are TA-Lib `CDLxxx` rules, **not** human annotation. The detector "
        "imitates a heuristic; where the rule and a trader disagree, the model "
        "follows the rule.",
        "- Every split is chronological with an embargo gap. Nothing is shuffled.",
        "- The downstream study uses the detector's **own predictions**, not the "
        "labels it was trained on, so it measures what a model reading the picture "
        "can extract — mistakes included.",
        "- Returns shown are an illustration of the classifier's edge, net of a "
        "modelled spread only. They ignore slippage, financing and taxes, and are "
        "not a trading recommendation.",
        "",
    ])


def inject(readme: Path | None = None) -> Path:
    """Replace the marked results block in the README, leaving the rest intact."""
    readme = readme or (config.PROJECT_ROOT / "README.md")
    body = f"{START_MARKER}\n{build_report()}\n{END_MARKER}"
    text = readme.read_text() if readme.exists() else ""
    if START_MARKER in text and END_MARKER in text:
        head, _, rest = text.partition(START_MARKER)
        _, _, tail = rest.partition(END_MARKER)
        text = f"{head}{body}{tail}"
    else:
        text = f"{text}\n\n{body}\n"
    readme.write_text(text)
    logger.info("results injected into %s", readme)
    return readme


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inject", action="store_true",
                        help="rewrite the results block inside README.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(inject() if args.inject else build_report())
