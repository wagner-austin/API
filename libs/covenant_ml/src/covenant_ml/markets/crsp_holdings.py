"""Monthly CRSP holdings: drifted positions, turnover costs and dated terminal values.

Portfolio value starts at one unit of reinvestable cash. At every rebalance,
the first trading close of a month, the investable value (held positions plus
cash whose terminal valuation date lies strictly before that close) is split
equally across the selected securities. Traded notional is measured against
the drifted pre-trade positions and actual post-cost targets. The budget
equation pays that cost from the portfolio. A new purchase whose entry record
is present but is not a recorded closing trade (a confirmed pre-entry
delisting, a quote, a suspension or an untracked day) is unfilled: its equal
allocation stays in zero-interest cash, without replacement or redistribution.
A position already held still needs a closing trade at every rebalance.
Positions then compound their own evidenced return intervals to the next
rebalance. A security delisted inside the window, or on an exit close at which
it has no closing trade to sell into, stops compounding at its delisting date
and becomes a receivable worth its terminal multiplier, applied once and dated
by ``crsp_resolve_terminal``; it is not reinvested until a rebalance strictly
after that date, which may follow the exit. The last window also pays the cost
of liquidating its remaining positions at its exit close.

Only unpriced rows resolved by an exactly dated multi-session return are
accepted. An absent entry record, a held position without a closing trade,
unresolved held intervals or terminal values refuse the whole evaluation. A
partial result is never returned.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import pairwise
from math import isfinite, prod
from typing import Final, NamedTuple

from platform_core.json_utils import JSONObject, require_int, require_str

from .crsp_calendar import CrspWindow, crsp_executable, crsp_resolve_terminal
from .crsp_extract import CrspSources
from .crsp_protocol import CrspProtocol
from .crsp_return_path import crsp_resolve_held


class CrspOutcomes(NamedTuple):
    """Verified daily and terminal evidence, indexed once for outcome accounting."""

    protocol: CrspProtocol
    sessions: tuple[str, ...]
    returns: dict[tuple[int, str], JSONObject]
    delists: dict[int, tuple[JSONObject, ...]]


class CrspPath(NamedTuple):
    """One security's value multipliers across one window.

    Args:
        filled: Whether the selected purchase executed; a rejected purchase
            retains cash with growth one and no terminal claim.
        growth: Compounded evidenced returns after entry, through the exit
            or, for a delisted security, through its delisting date.
        terminal: Terminal valuation date, possibly after the exit, and
            multiplier when the security was delisted in this window, else None.
    """

    filled: bool
    growth: float
    terminal: tuple[str, float] | None


#: The path of an order with no closing trade at entry: cash, no claim.
UNFILLED: Final = CrspPath(False, 1.0, None)


class CrspReceivable(NamedTuple):
    """A terminal value awaiting reinvestment, dated by its valuation date."""

    permno: int
    valued: str
    amount: float


class CrspPeriod(NamedTuple):
    """One rebalance-to-rebalance month of a self-financing portfolio.

    Args:
        window: Decision, entry and exit closes.
        selected: Decision-time selections, including rejected purchases.
        unfilled: Selections with no closing trade at entry, kept as cash.
        cash: Reinvestable cash as a share of the pre-liquidation end value.
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
    unfilled: tuple[int, ...]
    cash: float
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


def _terminal_rows(
    outcomes: CrspOutcomes, permno: int, window: CrspWindow
) -> tuple[JSONObject, ...]:
    """Find the terminal events that stop a position inside the window.

    DelistingDt is the security's last price on its exchange (CIZ guide p.11).
    An event on the exit close belongs to the next window when the exit record
    is a closing trade, because the position is sold into it. When the exit
    record is present but is no closing trade, nothing can be sold there and
    the position ends with this window's terminal value instead.

    Args:
        outcomes: Indexed evidence.
        permno: Held security.
        window: The holding window.

    Returns:
        Every event dated in ``[entry, exit)``, and one on the exit close when
        the exit record is present without a closing trade; more than one is
        ambiguous.
    """
    unsold = _unsold(outcomes, permno, window)
    return tuple(
        row
        for row in outcomes.delists.get(permno, ())
        if window.entry <= (day := require_str(row, "delistingdt")) < window.exit
        or (unsold and day == window.exit)
    )


def _unsold(outcomes: CrspOutcomes, permno: int, window: CrspWindow) -> bool:
    """Whether the exit record is present but is no closing trade to sell into."""
    exit_row = outcomes.returns.get((permno, window.exit))
    return exit_row is not None and exit_row["dlyprcflg"] != outcomes.protocol["execution_flag"]


def _eventless_delisting(
    outcomes: CrspOutcomes, permno: int, window: CrspWindow, days: tuple[str, ...]
) -> CrspPathRefusal | None:
    """Name a delisting return the daily series holds for a window with no event.

    CIZ stores the delisting return on DelDlyDt, the session after DelistingDt
    (guide p.12), with DlyDelFlg ``Y`` (p.17). A delisting inside the window
    therefore flags a held session, and one on an unsold exit close flags the
    session after the exit. Extraction keeps an event only when its valuation
    date is within the partition, so a flagged row without its event is the
    trace of an event valued after the observation end, or of vendor evidence
    missing it. It is refused by that name, never valued or read as a return.

    Args:
        outcomes: Indexed evidence.
        permno: Held security.
        window: The holding window, which has no terminal event.
        days: Declared sessions in ``(entry, exit]``.

    Returns:
        The refusal for the first flagged session, or None.
    """
    following = bisect_right(outcomes.sessions, window.exit)
    unsold = _unsold(outcomes, permno, window)
    after = outcomes.sessions[following : following + 1] if unsold else ()
    for day in (*days, *after):
        row = outcomes.returns.get((permno, day))
        if row is not None and row["dlydelflg"] == "Y":
            return CrspPathRefusal(
                "delisting return without event",
                f"[CRSP-009] the delisting return for {permno} on {day}"
                " has no terminal event in this partition",
            )
    return None


class CrspPathRefusal(NamedTuple):
    """Why one security's window cannot be valued from the vendor's records.

    Attributes:
        reason: A short key for counting: ``entry absent``, ``entry terminal
            inconsistent``, ``entry not a closing trade`` (a closing-trade flag
            without a positive price), ``two terminal events``, ``delisting
            return without event`` (a flagged daily row whose event the
            partition does not hold), a held-return
            reason of :class:`~covenant_ml.markets.crsp_return_path.HeldRefusal`,
            ``held growth not finite``, ``exit absent``, ``exit not a closing
            trade`` or a terminal reason of
            :func:`~covenant_ml.markets.crsp_calendar.crsp_resolve_terminal`.
        message: The full error, naming its CRSP code.
    """

    reason: str
    message: str


def _fill_refusal(
    outcomes: CrspOutcomes, permno: int, day: str, side: str
) -> CrspPathRefusal | None:
    """Explain why a close cannot execute a modeled trade, or None when it can.

    Args:
        outcomes: Indexed evidence.
        permno: Security bought, sold or rebalanced.
        day: The execution close.
        side: ``entry`` or ``exit``, naming the counting key.

    Returns:
        None for a finite positive closing trade, else the refusal.
    """
    row = outcomes.returns.get((permno, day))
    if row is None:
        return CrspPathRefusal(
            f"{side} absent", f"[CRSP-009] no execution record for {permno} on {day}"
        )
    if not crsp_executable(row, outcomes.protocol):
        return CrspPathRefusal(
            f"{side} not a closing trade",
            "[CRSP-004] execution requires a finite positive closing trade",
        )
    return None


def crsp_resolve_window_path(
    outcomes: CrspOutcomes, permno: int, window: CrspWindow
) -> CrspPath | CrspPathRefusal:
    """Value one security's window, reporting rather than raising the first refusal.

    Execution, held spans and the terminal event are judged in that order,
    exactly as holdings and labels need them. Qualification counts the refusal
    this returns; :func:`crsp_window_path` raises it.

    Args:
        outcomes: Indexed evidence.
        permno: Security ordered at the entry close.
        window: Holding window.

    Returns:
        The path, or the first reason the vendor's records cannot value it.

    Raises:
        ValueError: CRSP-009 when the window's closes are not declared sessions,
            a calendar fault rather than a security's evidence.
    """
    days = _sessions_after(outcomes.sessions, window.entry, window.exit)
    entry = _entry(outcomes, permno, window)
    if entry is not None:
        return entry
    terminals = _terminal_rows(outcomes, permno, window)
    if len(terminals) > 1:
        return CrspPathRefusal(
            "two terminal events", "[CRSP-009] one window cannot hold two terminal events"
        )
    eventless = None if terminals else _eventless_delisting(outcomes, permno, window, days)
    if eventless is not None:
        return eventless
    end = window.exit if not terminals else require_str(terminals[0], "delistingdt")
    held = crsp_resolve_held(
        outcomes.returns, permno, window.entry, days[: bisect_right(days, end)]
    )
    if held.refusal is not None:
        return CrspPathRefusal(held.refusal.reason, held.refusal.message)
    growth = prod(span for _, span in held.spans)
    if not isfinite(growth):
        return CrspPathRefusal(
            "held growth not finite", "[CRSP-009] compounded held return is not finite"
        )
    if not terminals:
        refusal = _fill_refusal(outcomes, permno, window.exit, "exit")
        return CrspPath(True, growth, None) if refusal is None else refusal
    value = crsp_resolve_terminal(terminals[0])
    if value.refusal is not None:
        return CrspPathRefusal(
            value.refusal,
            f"[CRSP-005] terminal return or valuation chronology is unresolved: {value.refusal}",
        )
    return CrspPath(True, growth, (value.valued, value.multiplier))


def crsp_window_path(outcomes: CrspOutcomes, permno: int, window: CrspWindow) -> CrspPath:
    """Validate execution, compound complete return spans and resolve a terminal event.

    The daily series' own delisting-flagged rows after the delisting date are
    never compounded: the terminal multiplier comes from ``stkdelists`` alone,
    so a terminal outcome is counted exactly once. Label construction uses this
    same path as holdings: an unfilled order retains cash, while an executed
    order requires a tradable exit or a terminal event of this window. The
    terminal value may be dated after the exit; the caller dates the label
    and the receivable by it. The partition excludes any event whose
    valuation lies past its observation end; a path whose delisting return
    is then present without its event refuses by that name.

    Args:
        outcomes: Indexed evidence.
        permno: Security held from the entry close.
        window: Holding window.

    Returns:
        The ordinary growth and, for a delisting in this window, its dated value.

    Raises:
        ValueError: :class:`CrspPathRefusal`'s message: CRSP-004 for a
            malformed entry trade or an unexecutable ordinary exit; CRSP-009 for
            a missing entry record, two terminal events or a missing or
            inconsistent held return; CRSP-005 for an unresolved terminal event;
            CRSP-010 for contradictory terminal-entry evidence; CRSP-011 for an
            unresolved price gap at the boundary.
    """
    path = crsp_resolve_window_path(outcomes, permno, window)
    if isinstance(path, CrspPathRefusal):
        raise ValueError(path.message)
    return path


def _entry(
    outcomes: CrspOutcomes, permno: int, window: CrspWindow
) -> CrspPath | CrspPathRefusal | None:
    """Distinguish an executed new purchase from an unfilled one and from no evidence.

    An order fills only at a recorded closing trade (CIZ price flag TR, guide
    p.88). A present entry record with any other price flag (a quote BA, SU,
    HA, NT, MP) is an unfilled order: no security was bought and no return is
    imputed. A terminal daily row is unfilled only when exactly one event at
    the decision close precedes entry, the duration is DD and the price is
    not marked as a closing trade. An absent record, a contradictory terminal
    row or a malformed trade still refuses evaluation.

    Args:
        outcomes: Indexed verified source records.
        permno: Selected security.
        window: Decision and execution dates.

    Returns:
        None for a verified closing trade, the unfilled cash path, or the
        refusal: an absent record (CRSP-009), inconsistent terminal-entry
        evidence (CRSP-010) or a closing-trade flag without a finite positive
        price (CRSP-004).
    """
    row = outcomes.returns.get((permno, window.entry))
    if row is not None and row["dlydelflg"] == "Y":
        events = [
            item
            for item in outcomes.delists.get(permno, ())
            if require_str(item, "delistingdt") == window.decision
        ]
        if (
            len(events) != 1
            or window.decision >= window.entry
            or row["dlyretdurflg"] != "DD"
            or row["dlyprcflg"] == "TR"
        ):
            return CrspPathRefusal(
                "entry terminal inconsistent",
                f"[CRSP-010] inconsistent terminal entry for {permno}",
            )
        return UNFILLED
    if row is not None and row["dlyprcflg"] != outcomes.protocol["execution_flag"]:
        return UNFILLED
    return _fill_refusal(outcomes, permno, window.entry, "entry")


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
        # A held position needs no second fill check here or at liquidation:
        # its own path already required a closing trade on its exit close,
        # which is this entry and, in the last window, the liquidation close.
        paths = {permno: crsp_window_path(outcomes, permno, window) for permno in selected}
        filled = tuple(permno for permno in selected if paths[permno].filled)
        unfilled = tuple(permno for permno in selected if not paths[permno].filled)
        each, traded = _rebalance_notional(investable, positions, filled, len(selected), rate)
        cost = rate * traded
        cash = each * len(unfilled)
        positions = {}
        for permno in filled:
            path = paths[permno]
            if path.terminal is None:
                positions[permno] = each * path.growth
            else:
                valued, multiplier = path.terminal
                amount = each * path.growth * multiplier
                receivables.append(CrspReceivable(permno, valued, amount))
        end = cash + sum(positions.values()) + sum(item.amount for item in receivables)
        if not end > 0:
            raise ValueError("[CRSP-009] portfolio value was exhausted; weights are undefined")
        held = tuple((permno, positions[permno] / end) for permno in sorted(positions))
        owed = tuple(
            CrspReceivable(item.permno, item.valued, item.amount / end) for item in receivables
        )
        cash_weight = cash / end
        if index == len(windows) - 1:
            liquidated = sum(positions.values())
            traded += liquidated
            cost += rate * liquidated
            end -= rate * liquidated
        periods.append(
            CrspPeriod(
                window,
                selected,
                unfilled,
                cash_weight,
                traded / start,
                cost / start,
                end / start - 1,
                held,
                owed,
            )
        )
    return tuple(periods)


def _rebalance_notional(
    investable: float,
    positions: dict[int, float],
    filled: tuple[int, ...],
    slots: int,
    rate: float,
) -> tuple[float, float]:
    """Solve the self-financing equal-allocation budget on its linear segments.

    Every slot, including a rejected order's cash, receives x. Actual turnover
    T(x) is the sum of sells outside filled plus abs(x - prior) inside filled.
    Solve slots*x + rate*T(x) = investable. Its slope is strictly positive
    under the frozen cost rates; breakpoints are prior position values.

    Args:
        investable: Nonnegative cash and held value available to allocate.
        positions: Drifted pre-trade position values.
        filled: Purchases supported by execution evidence.
        slots: Positive count of decision-time selections.
        rate: Frozen cost per unit of actual traded notional.

    Returns:
        Equal post-cost slot value and actual traded notional.
    """
    prior = sorted(positions.get(permno, 0.0) for permno in filled)
    sold = sum(amount for permno, amount in positions.items() if permno not in filled)
    intercept = sold + sum(prior)
    slope = -len(prior)
    for boundary in prior:
        if slots * boundary + rate * (intercept + slope * boundary) >= investable:
            break
        intercept -= 2 * boundary
        slope += 2
    each = (investable - rate * intercept) / (slots + rate * slope)
    traded = sold + sum(abs(each - value) for value in prior)
    return each, traded


__all__ = [
    "CrspOutcomes",
    "CrspPath",
    "CrspPathRefusal",
    "CrspPeriod",
    "CrspReceivable",
    "crsp_resolve_window_path",
    "crsp_window_path",
    "evaluate_crsp_holdings",
    "index_crsp_outcomes",
]
