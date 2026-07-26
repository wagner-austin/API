"""Measuring what an extractor is actually worth per second.

The catalogue prices an extractor at 700 and describes it as "-Generates
credits". It does not say how fast, and no shipped file does -- so every
decision that trades credits now against income later has been made without the
one number that decides it. The reserve the economy holds back for the army, and
the choice between a fourth extractor and two more tanks, both rest on a payback
period nobody had measured ([[policy-economy]]).

This is the arithmetic half of measuring it. A driver reads the world while
ordering nothing, tags each stretch of observations with a window, and hands the
readings here; what comes back is a credit rate per window and the marginal
value of each extractor above the baseline.

Windows matter because credits move for two reasons. Income raises them
continuously and spending drops them at once, so a slope taken across a purchase
measures the purchase. A window is therefore a stretch during which nothing was
bought, and one where credits fall is discarded rather than explained -- an
extractor destroyed mid-window, or a unit reclaimed, is a measurement that
cannot be trusted rather than one to correct.

Pure: readings in, rates out. Nothing here opens a socket.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

#: Engine milliseconds in a second, named so the conversion below reads as a
#: unit change rather than a magic number.
MS_PER_SECOND = 1000.0


class Reading(TypedDict):
    """One observation taken while the bot was deliberately doing nothing.

    Attributes:
        window: Which idle stretch this belongs to. Readings from different
            windows are never compared directly, because something was bought
            in between.
        extractors: Finished extractors owned at this moment.
        clock_ms: Engine clock.
        credits: Credits held.
    """

    window: int
    extractors: int
    clock_ms: int
    credits: int


class Rate(TypedDict):
    """What one window measured.

    Attributes:
        window: The window this describes.
        extractors: Finished extractors owned throughout it.
        readings: How many observations it holds.
        span_ms: Engine milliseconds from its first reading to its last.
        gained: Credits gained across it.
        per_second: Credits per second, the slope.
    """

    window: int
    extractors: int
    readings: int
    span_ms: int
    gained: int
    per_second: float


def measure(readings: Sequence[Reading]) -> tuple[Rate, ...]:
    """Reduce readings to one credit rate per window.

    A window needs at least two readings and a clock that advanced; anything
    less has no slope to report and is dropped rather than reported as zero,
    which would be indistinguishable from a real answer of nothing.

    A window whose extractor count changed is also dropped. Construction
    finishing mid-window means the rate belongs to neither count, and averaging
    the two would invent a value the game never produced.

    Args:
        readings: Observations, in the order they were taken.

    Returns:
        One rate per usable window, in window order.
    """
    grouped: dict[int, list[Reading]] = {}
    for reading in readings:
        grouped.setdefault(reading["window"], []).append(reading)

    rates: list[Rate] = []
    for window in sorted(grouped):
        rows = grouped[window]
        if len(rows) < 2:
            continue
        counts = {row["extractors"] for row in rows}
        if len(counts) != 1:
            continue
        span = rows[-1]["clock_ms"] - rows[0]["clock_ms"]
        gained = rows[-1]["credits"] - rows[0]["credits"]
        if span <= 0 or gained < 0:
            continue
        rates.append(
            Rate(
                window=window,
                extractors=rows[0]["extractors"],
                readings=len(rows),
                span_ms=span,
                gained=gained,
                per_second=gained * MS_PER_SECOND / span,
            )
        )
    return tuple(rates)


def marginal(rates: Sequence[Rate]) -> float | None:
    """Return the credits per second one extractor adds.

    Taken as the slope between the fewest and the most extractors measured,
    rather than between neighbours, because a single pair of windows carries
    whatever noise one measurement had and the endpoints spread it over the
    widest base available.

    Args:
        rates: Measured windows.

    Returns:
        Credits per second per extractor, or None when fewer than two distinct
        extractor counts were measured and no slope exists.
    """
    if not rates:
        return None
    # Explicit loops rather than min/max with a key: a lambda over a TypedDict
    # erases to Any under the strict settings this package is checked with.
    fewest = rates[0]
    most = rates[0]
    for rate in rates:
        if rate["extractors"] < fewest["extractors"]:
            fewest = rate
        if rate["extractors"] > most["extractors"]:
            most = rate
    span = most["extractors"] - fewest["extractors"]
    if span <= 0:
        return None
    return (most["per_second"] - fewest["per_second"]) / span


def payback_seconds(price: int, per_extractor: float) -> float | None:
    """Return how long an extractor takes to earn its own price back.

    The number the reserve policy actually needs. An extractor that pays for
    itself in half a minute should be bought ahead of almost anything; one that
    takes five is a bet on the match lasting.

    Args:
        price: What the extractor costs.
        per_extractor: Credits per second it generates.

    Returns:
        Seconds to break even, or None when it generates nothing measurable and
        never breaks even.
    """
    if per_extractor <= 0.0:
        return None
    return price / per_extractor


def format_rates(rates: Sequence[Rate]) -> tuple[str, ...]:
    """Render measured windows as lines.

    Args:
        rates: Measured windows.

    Returns:
        A header and one line per window.
    """
    lines = ["window  extractors  readings   span_ms   gained   credits/s"]
    for rate in rates:
        lines.append(
            f"{rate['window']:>6}  {rate['extractors']:>10}  {rate['readings']:>8}  "
            f"{rate['span_ms']:>8}  {rate['gained']:>6}  {rate['per_second']:>10.2f}"
        )
    return tuple(lines)


__all__ = [
    "MS_PER_SECOND",
    "Rate",
    "Reading",
    "format_rates",
    "marginal",
    "measure",
    "payback_seconds",
]
