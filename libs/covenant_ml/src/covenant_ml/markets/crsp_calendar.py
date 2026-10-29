"""CRSP session, execution and terminal valuation rules for the frozen study."""

from __future__ import annotations

from datetime import date, timedelta
from math import isfinite
from typing import NamedTuple

from platform_core.json_utils import JSONObject, require_float, require_int, require_str

from .crsp_protocol import CrspProtocol
from .validation import require_day


class CrspWindow(NamedTuple):
    """Month-end decision and the following two months' first trading closes."""

    decision: str
    entry: str
    exit: str


def _calendar_months(calendar: tuple[tuple[str, str], ...]) -> tuple[list[str], ...]:
    """Validate a nonempty calendar and group its declared sessions by month.

    Args:
        calendar: Ordered (caldt, tradingflg) rows for ExchangeGroup CIZ.
            Include complete calendar months through the final exit month.

    Returns:
        Ordered session lists, one for every complete calendar month.

    Raises:
        ValueError: CRSP-002 for missing dates/months or bad flags.
    """
    months: dict[str, list[str]] = {}
    previous: date | None = None
    for day, flag in calendar:
        current = date.fromisoformat(require_day({"day": day}, "day"))
        if previous is None:
            if current.day != 1:
                raise ValueError("[CRSP-002] calendar must start at a month boundary")
        elif current != previous + timedelta(days=1):
            raise ValueError("[CRSP-002] calendar dates must be unique and contiguous")
        if flag not in ("Y", "N"):
            raise ValueError("[CRSP-002] calendar trading flag must be Y or N")
        sessions = months.setdefault(day[:7], [])
        if flag == "Y":
            sessions.append(day)
        previous = current
    if (date.fromisoformat(calendar[-1][0]) + timedelta(days=1)).day != 1:
        raise ValueError("[CRSP-002] calendar must end at a month boundary")
    if any(not sessions for sessions in months.values()):
        raise ValueError("[CRSP-002] every declared month needs at least one session")
    return tuple(months.values())


def crsp_windows(
    calendar: tuple[tuple[str, str], ...], first_decision: str, last_decision: str
) -> tuple[CrspWindow, ...]:
    """Derive monthly windows from a complete CIZ calendar, including nontrading days.

    Args:
        calendar: Ordered (caldt, tradingflg) rows for ExchangeGroup CIZ.
            Include complete calendar months through the final exit month.
        first_decision: Inclusive lower decision-date bound.
        last_decision: Inclusive upper decision-date bound.

    Returns:
        Windows whose month-end trading session lies within the requested bounds.

    Raises:
        ValueError: CRSP-002 for missing dates/months, bad flags or incomplete exits.
    """
    start = require_day({"day": first_decision}, "day")
    end = require_day({"day": last_decision}, "day")
    if not calendar or start > end:
        raise ValueError("[CRSP-002] require ordered bounds and a complete calendar")
    ordered = _calendar_months(calendar)
    if calendar[0][0] > start or calendar[-1][0] < end:
        raise ValueError("[CRSP-002] calendar must cover both decision bounds")
    windows: list[CrspWindow] = []
    for index, sessions in enumerate(ordered):
        decision = sessions[-1]
        if start <= decision <= end:
            if index + 2 >= len(ordered):
                raise ValueError("[CRSP-002] complete entry and exit months are required")
            windows.append(CrspWindow(decision, ordered[index + 1][0], ordered[index + 2][0]))
    if not windows:
        raise ValueError("[CRSP-002] no decisions lie in the requested bounds")
    return tuple(windows)


def crsp_member_on(
    rows: tuple[JSONObject, ...], permno: int, decision: str, protocol: CrspProtocol
) -> bool:
    """Join legacy numeric share/exchange codes on inclusive effective date bounds.

    Args:
        rows: Historical dsenames rows; never current header classifications.
        permno: Security being considered for entry.
        decision: Canonical decision date.
        protocol: Frozen allowed share and exchange codes.

    Returns:
        Whether exactly one effective classification exists and is eligible.

    Raises:
        ValueError: CRSP-003 for overlapping or reversed membership intervals.
    """
    day = require_day({"decision": decision}, "decision")
    active: list[JSONObject] = []
    for row in rows:
        first, last = require_day(row, "namedt"), require_day(row, "nameendt")
        if first > last:
            raise ValueError("[CRSP-003] membership interval is reversed")
        if require_int(row, "permno") == permno and first <= day <= last:
            active.append(row)
    if len(active) > 1:
        raise ValueError("[CRSP-003] overlapping effective membership is ambiguous")
    if not active:
        return False
    row = active[0]
    return (
        require_int(row, "shrcd") in protocol["share_codes"]
        and require_int(row, "exchcd") in protocol["exchange_codes"]
    )


def require_crsp_execution_price(row: JSONObject, protocol: CrspProtocol) -> float:
    """Accept only a finite positive closing trade for a modeled ordinary fill.

    Args:
        row: The exact execution date's daily record, already identity-joined.
        protocol: Frozen accepted price flag.

    Returns:
        Positive trade price. Quotes, halts and delisting valuations are not fills.

    Raises:
        ValueError: CRSP-004 for an unexecutable or missing price.
    """
    price = require_float(row, "dlyprc")
    if (
        require_str(row, "dlyprcflg") != protocol["execution_flag"]
        or not isfinite(price)
        or price <= 0
    ):
        raise ValueError("[CRSP-004] execution requires a finite positive closing trade")
    return price


def crsp_terminal_value(row: JSONObject) -> tuple[str, float]:
    """Return the terminal valuation date and multiplier, never an actual cash date.

    ``dated-value-next-rebalance`` marks the terminal receivable on the latest of
    DelDlyDt, DelAmtDt and a present DelNextDt. It becomes modeled reinvestable
    value at the next scheduled rebalance strictly after that date. The evaluator
    must refuse an outcome whose required value is dated after its scoring exit.
    This is a retrospective settlement assumption, not observed cash receipt.

    Args:
        row: Complete stkdelists record. DelAmtDt is required; DelNextDt is nullable.

    Returns:
        Conservative recorded valuation date and one plus the delisting return.

    Raises:
        ValueError: CRSP-005 for unresolved value, inconsistent chronology or flags.
        TypeError: Missing or malformed fields, including missing DelAmtDt.
    """
    event = require_day(row, "delistingdt")
    stored = require_day(row, "deldlydt")
    amount = require_day(row, "delamtdt")
    value = require_float(row, "delret")
    dates = [stored, amount]
    if "delnextdt" not in row:
        raise ValueError("[CRSP-005] require an explicit nullable next-price date")
    if row["delnextdt"] is not None:
        dates.append(require_day(row, "delnextdt"))
    if (
        require_str(row, "delretmisstype") != "NA"
        or not isfinite(value)
        or value < -1
        or stored <= event
        or any(day < event for day in dates)
    ):
        raise ValueError("[CRSP-005] terminal return or valuation chronology is unresolved")
    return max(dates), 1 + value
