"""Monthly CRSP holdings: drifted positions, turnover costs and dated terminal values.

Portfolio value starts at one unit of reinvestable cash. At every rebalance,
the first trading close of a month, the investable value (held positions plus
cash whose terminal valuation date lies strictly before that close) is split
equally across the selected securities. Traded notional is measured against
the drifted pre-trade positions, and its cost is paid out of the portfolio,
so the evaluation is self-financing. Positions then compound their own
ordinary daily returns to the next rebalance. A security delisted inside the
window stops compounding at its delisting date and becomes a receivable worth
its terminal multiplier, applied once and dated by ``crsp_terminal_value``; it
is not reinvested until a rebalance strictly after that date. The last window
also pays the cost of liquidating its remaining positions at its exit close.

Every missing or non-ordinary held return, missing fill, unresolved terminal
event or terminal value dated after the scored exit refuses the whole
evaluation. A partial result is never returned.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import pairwise
from math import isfinite
from typing import NamedTuple

from platform_core.json_utils import JSONObject, require_float, require_int, require_str

from .crsp_calendar import (
    CrspWindow,
    crsp_ordinary_return,
    crsp_terminal_value,
    require_crsp_execution_price,
)
from .crsp_extract import CrspSources
from .crsp_protocol import CrspProtocol


class CrspOutcomes(NamedTuple):
    """Verified daily and terminal evidence, indexed once for outcome accounting."""

    protocol: CrspProtocol
    sessions: tuple[str, ...]
    returns: dict[tuple[int, str], JSONObject]
    delists: dict[int, tuple[JSONObject, ...]]


class CrspPath(NamedTuple):
    """One security's value multipliers across one window.

    Args:
        growth: Compounded ordinary daily returns after entry, through the exit
            or, for a delisted security, through its delisting date.
        terminal: Terminal valuation date and multiplier when the security was
            delisted inside the window, else None.
    """

    growth: float
    terminal: tuple[str, float] | None


class CrspReceivable(NamedTuple):
    """A terminal value awaiting reinvestment, dated by its valuation date."""

    permno: int
    valued: str
    amount: float


class CrspPeriod(NamedTuple):
    """One rebalance-to-rebalance month of a self-financing portfolio.

    Args:
        window: Decision, entry and exit closes.
        selected: Securities bought to equal weight at the entry close.
        turnover: Traded notional over the start value, liquidation included.
        cost: Transaction cost over the start value.
        net_return: End value over start value, minus one.
        held: Drifted weights at the exit close, ascending PERMNO, as shares
            of the end value before any final liquidation.
        receivables: Terminal values outstanding at the exit close, with
            amounts as shares of that same end value.
    """

    window: CrspWindow
    selected: tuple[int, ...]
    turnover: float
    cost: float
    net_return: float
    held: tuple[tuple[int, float], ...]
    receivables: tuple[CrspReceivable, ...]


def index_crsp_outcomes(sources: CrspSources) -> CrspOutcomes:
    """Index verified rows by security and session without imputing anything.

    Args:
        sources: Output of ``load_crsp_sources``; its decoder already refused
            duplicate natural keys.

    Returns:
        Ordered declared sessions, daily rows keyed by (PERMNO, date) and
        terminal events grouped by PERMNO.

    Raises:
        ValueError: CRSP-009 when the declared sessions are not strictly ordered.
    """
    sessions = tuple(
        require_str(row, "caldt") for row in sources.calendar if row["tradingflg"] == "Y"
    )
    if any(later <= earlier for earlier, later in pairwise(sessions)):
        raise ValueError("[CRSP-009] declared sessions must be strictly ordered")
    returns = {
        (require_int(row, "permno"), require_str(row, "dlycaldt")): row for row in sources.returns
    }
    grouped: dict[int, list[JSONObject]] = {}
    for row in sources.delists:
        grouped.setdefault(require_int(row, "permno"), []).append(row)
    delists = {permno: tuple(rows) for permno, rows in grouped.items()}
    return CrspOutcomes(sources.protocol, sessions, returns, delists)


def _sessions_after(sessions: tuple[str, ...], entry: str, end: str) -> tuple[str, ...]:
    """Return declared sessions in ``(entry, end]``, requiring entry to be a session.

    Args:
        sessions: Ordered declared sessions.
        entry: Entry close, which must itself be a declared session.
        end: Inclusive last date.

    Returns:
        The sessions whose returns accrue to a position bought at entry.

    Raises:
        ValueError: CRSP-009 when entry is not a declared session.
    """
    start = bisect_left(sessions, entry)
    if start == len(sessions) or sessions[start] != entry:
        raise ValueError("[CRSP-009] window closes must be declared sessions")
    return sessions[start + 1 : bisect_right(sessions, end)]


def _terminal_row(outcomes: CrspOutcomes, permno: int, window: CrspWindow) -> JSONObject | None:
    """Find the one terminal event that stops a position inside the window.

    Args:
        outcomes: Indexed evidence.
        permno: Held security.
        window: The holding window; events on its exit close belong to the next.

    Returns:
        The event dated in ``[entry, exit)``, or None.

    Raises:
        ValueError: CRSP-009 for two events inside one window.
    """
    rows = [
        row
        for row in outcomes.delists.get(permno, ())
        if window.entry <= require_str(row, "delistingdt") < window.exit
    ]
    if len(rows) > 1:
        raise ValueError("[CRSP-009] one window cannot hold two terminal events")
    return rows[0] if rows else None


def crsp_window_path(outcomes: CrspOutcomes, permno: int, window: CrspWindow) -> CrspPath:
    """Compound one security's ordinary returns and resolve any terminal event.

    The daily series' own delisting-flagged rows after the delisting date are
    never compounded: the terminal multiplier comes from ``stkdelists`` alone,
    so a terminal outcome is counted exactly once.

    Args:
        outcomes: Indexed evidence.
        permno: Security held from the entry close.
        window: Holding window.

    Returns:
        The ordinary growth and, for an in-window delisting, its dated value.

    Raises:
        ValueError: CRSP-009 for a missing or non-ordinary held return or a
            terminal value dated after the exit; CRSP-005 propagates for an
            unresolved terminal event.
    """
    terminal = _terminal_row(outcomes, permno, window)
    end = window.exit if terminal is None else require_str(terminal, "delistingdt")
    growth = 1.0
    previous = window.entry
    for day in _sessions_after(outcomes.sessions, window.entry, end):
        row = outcomes.returns.get((permno, day))
        if row is None or not crsp_ordinary_return(row, previous):
            raise ValueError(f"[CRSP-009] held return for {permno} on {day} is not ordinary")
        growth *= 1 + require_float(row, "dlyret")
        previous = day
    if not isfinite(growth):
        raise ValueError("[CRSP-009] compounded held return is not finite")
    if terminal is None:
        return CrspPath(growth, None)
    valued, multiplier = crsp_terminal_value(terminal)
    if valued > window.exit:
        raise ValueError(f"[CRSP-009] terminal value for {permno} is dated after the exit")
    return CrspPath(growth, (valued, multiplier))


def _require_fill(outcomes: CrspOutcomes, permno: int, day: str) -> None:
    """Require a modeled closing trade for every security traded at a close.

    Args:
        outcomes: Indexed evidence.
        permno: Security bought, sold or rebalanced.
        day: Rebalance or liquidation close.

    Raises:
        ValueError: CRSP-009 when no daily record exists; CRSP-004 propagates
            when the record is not a finite positive closing trade.
    """
    row = outcomes.returns.get((permno, day))
    if row is None:
        raise ValueError(f"[CRSP-009] no execution record for {permno} on {day}")
    require_crsp_execution_price(row, outcomes.protocol)


def _require_design(
    protocol: CrspProtocol,
    windows: tuple[CrspWindow, ...],
    selections: tuple[tuple[int, ...], ...],
    cost_bps: int,
) -> None:
    """Refuse an accounting request outside the frozen monthly design.

    Args:
        protocol: Frozen design naming the permitted cost assumptions.
        windows: Consecutive monthly windows.
        selections: One selection per window.
        cost_bps: Cost per unit of traded notional, in basis points.

    Raises:
        ValueError: CRSP-009 for empty, misaligned or unchained windows, empty
            or repeated selections, or a cost the design does not declare.
    """
    if not windows or len(selections) != len(windows):
        raise ValueError("[CRSP-009] require one selection for every window")
    if any(earlier.exit != later.entry for earlier, later in pairwise(windows)):
        raise ValueError("[CRSP-009] each window must enter at the previous exit")
    if any(not selected or len(set(selected)) != len(selected) for selected in selections):
        raise ValueError("[CRSP-009] selections must be nonempty and distinct")
    if cost_bps not in protocol["cost_bps"]:
        raise ValueError("[CRSP-009] cost must be one the frozen design declares")


def evaluate_crsp_holdings(
    outcomes: CrspOutcomes,
    windows: tuple[CrspWindow, ...],
    selections: tuple[tuple[int, ...], ...],
    cost_bps: int,
) -> tuple[CrspPeriod, ...]:
    """Account for an equal-weight long-only portfolio across consecutive months.

    Args:
        outcomes: Indexed verified evidence.
        windows: Consecutive windows; each enters at the previous exit.
        selections: Securities to hold equally in each window.
        cost_bps: All-in cost per unit of traded notional, in basis points;
            one of the frozen design's declared costs.

    Returns:
        One period per window, only when every window's evidence is complete.

    Raises:
        ValueError: CRSP-009 for any incomplete or inconsistent evidence or an
            exhausted portfolio;
            CRSP-004 and CRSP-005 propagate from the frozen validators.
    """
    _require_design(outcomes.protocol, windows, selections, cost_bps)
    rate = cost_bps / 10000
    positions: dict[int, float] = {}
    receivables: list[CrspReceivable] = []
    cash = 1.0
    periods: list[CrspPeriod] = []
    for index, (window, selected) in enumerate(zip(windows, selections, strict=True)):
        start = cash + sum(positions.values()) + sum(item.amount for item in receivables)
        investable = cash + sum(positions.values())
        investable += sum(item.amount for item in receivables if item.valued < window.entry)
        receivables = [item for item in receivables if item.valued >= window.entry]
        target = investable / len(selected)
        traded = 0.0
        for permno in sorted(set(positions) | set(selected)):
            _require_fill(outcomes, permno, window.entry)
            traded += abs((target if permno in selected else 0.0) - positions.get(permno, 0.0))
        cost = rate * traded
        each = (investable - cost) / len(selected)
        cash = 0.0
        positions = {}
        for permno in selected:
            path = crsp_window_path(outcomes, permno, window)
            if path.terminal is None:
                positions[permno] = each * path.growth
            else:
                valued, multiplier = path.terminal
                amount = each * path.growth * multiplier
                receivables.append(CrspReceivable(permno, valued, amount))
        end = sum(positions.values()) + sum(item.amount for item in receivables)
        if not end > 0:
            raise ValueError("[CRSP-009] portfolio value was exhausted; weights are undefined")
        held = tuple((permno, positions[permno] / end) for permno in sorted(positions))
        owed = tuple(
            CrspReceivable(item.permno, item.valued, item.amount / end) for item in receivables
        )
        if index == len(windows) - 1:
            for permno in positions:
                _require_fill(outcomes, permno, window.exit)
            liquidated = sum(positions.values())
            traded += liquidated
            cost += rate * liquidated
            end -= rate * liquidated
        periods.append(
            CrspPeriod(window, selected, traded / start, cost / start, end / start - 1, held, owed)
        )
    return tuple(periods)


__all__ = [
    "CrspOutcomes",
    "CrspPath",
    "CrspPeriod",
    "CrspReceivable",
    "crsp_window_path",
    "evaluate_crsp_holdings",
    "index_crsp_outcomes",
]
