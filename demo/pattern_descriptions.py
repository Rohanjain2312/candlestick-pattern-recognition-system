"""Plain-language descriptions of each pattern class, for the demo UI.

Written for someone who has never read a candlestick chart. Each entry says what
the shape looks like, what traders traditionally read into it, and -- because
this project is a study rather than a sales pitch -- how much that reading is
actually worth according to the evidence.

The wording deliberately avoids implying that any pattern predicts anything. The
repo's own results are the authority on that, not folklore.
"""

from __future__ import annotations

DESCRIPTIONS: dict[str, dict[str, str]] = {
    "Hammer": {
        "shape": "A small body sitting at the top of the candle, with a long lower "
                 "wick at least twice the body's height and little or no upper wick.",
        "story": "Sellers pushed the price well below the open during the session, "
                 "but buyers took it all back by the close. Traditionally read as a "
                 "bullish reversal when it appears after a decline.",
        "bias": "bullish",
    },
    "ShootingStar": {
        "shape": "The mirror image of a Hammer: a small body near the low, with a "
                 "long upper wick and almost no lower wick.",
        "story": "Buyers drove the price up during the session and then lost all of "
                 "it by the close. Traditionally read as a bearish reversal after "
                 "an advance.",
        "bias": "bearish",
    },
    "BullishEngulfing": {
        "shape": "Two candles. A down candle, then an up candle whose body is large "
                 "enough to completely cover the previous one.",
        "story": "One session of selling is entirely undone by the next session of "
                 "buying. Traditionally read as a shift of control to the buyers.",
        "bias": "bullish",
    },
    "BearishEngulfing": {
        "shape": "Two candles. An up candle, then a down candle whose body completely "
                 "covers the previous one.",
        "story": "A session of buying is entirely erased the next day. Traditionally "
                 "read as a shift of control to the sellers.",
        "bias": "bearish",
    },
    "MorningStar": {
        "shape": "Three candles: a long down candle, a small indecisive candle that "
                 "gaps lower, then a long up candle that recovers much of the first.",
        "story": "A decline stalls, hesitates for a session, then reverses. One of "
                 "the best known bullish reversal formations.",
        "bias": "bullish",
    },
    "EveningStar": {
        "shape": "Three candles: a long up candle, a small indecisive candle that "
                 "gaps higher, then a long down candle.",
        "story": "An advance runs out of momentum and turns over. The bearish "
                 "counterpart to the Morning Star.",
        "bias": "bearish",
    },
    "Doji": {
        "shape": "Open and close at essentially the same price, leaving a body that "
                 "is little more than a horizontal line, usually with wicks either side.",
        "story": "Buyers and sellers finished the session level. Read as indecision "
                 "rather than direction, which is why it carries no directional bias "
                 "here. It is also by far the most common pattern in the data.",
        "bias": "neutral",
    },
    "Harami": {
        "shape": "Two candles. A large candle followed by a small one whose body sits "
                 "entirely inside the previous body.",
        "story": "A strong session is followed by a quiet, contained one. Read as a "
                 "loss of momentum; the direction it then breaks is not implied by "
                 "the pattern itself.",
        "bias": "neutral",
    },
}

HONESTY_NOTE = """
**These descriptions are traditional readings, not findings.**

Every box this demo draws comes from a model trained on TA-Lib's `CDLxxx` rules,
not on human annotation, so the detector reproduces a published heuristic rather
than a trader's judgement. Whether any of these shapes actually helps predict
the next day's move is the question the **Results** tab answers with out-of-sample
numbers, and the answer there outranks the folklore above.
"""


def describe(pattern: str) -> dict[str, str]:
    """Return the description block for a class, or a safe placeholder."""
    return DESCRIPTIONS.get(pattern, {
        "shape": "No description available.", "story": "", "bias": "neutral",
    })


def as_markdown() -> str:
    """Render every description as a markdown document for the demo's tab."""
    icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
    parts = ["## What each pattern means", ""]
    for name, d in DESCRIPTIONS.items():
        parts += [
            f"### {icon[d['bias']]} {name}",
            f"**What it looks like.** {d['shape']}",
            "",
            f"**What it is said to mean.** {d['story']}",
            "",
        ]
    parts.append(HONESTY_NOTE)
    return "\n".join(parts)
