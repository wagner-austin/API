"""Realized Asian-put contracts from verified CRSP CIZ sources.

One hypothetical contract starts at each monthly decision close on each
security the frozen panel admits there (:func:`~covenant_ml.markets.crsp_panel.build_crsp_panel`:
effective share and exchange codes, minimum price, history and liquidity
rank, all known at the decision). Its path is that security's returns over
the following sessions, normalized to the contract's spot, so no price level
or fill is needed at the start.

TWO FIXING ASSUMPTIONS, both counted in :class:`PathCounts` rather than hidden:

* A session inside a CIZ missing-price span (MP/MR rows) has no price, but an
  Asian average fixes on every session. The last observed close is carried
  through the span, and the NA/P1-P9 return that closes it is applied on the
  closing session (:func:`~covenant_ml.markets.crsp_return_path.crsp_held_spans`),
  so no later information reaches an earlier fixing.
* A security delisted inside the contract's life stops at its delisting date;
  the ``stkdelists`` terminal multiplier
  (:func:`~covenant_ml.markets.crsp_calendar.crsp_terminal_value`) is applied
  once on the next session and the value is held flat to maturity. The
  daily rows after the delisting date are never read, so the terminal outcome
  is counted once, as in the holdings evaluator.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict, require_int
from platform_core.json_utils import require_str as json_str
from typing_extensions import TypedDict

from ..markets.crsp_calendar import crsp_month, crsp_terminal_value
from ..markets.crsp_extract import CrspSources
from ..markets.crsp_holdings import CrspOutcomes, index_crsp_outcomes
from ..markets.crsp_panel import build_crsp_panel
from ..markets.crsp_return_path import crsp_held_spans
from .paths import total_sessions
from .rates import RateObservation, carry_rate
from .study import RealizedContracts
from .types import ERR_PATH_SHAPE, AsianPutContract


class ContractPath(NamedTuple):
    """One contract's session returns and the fixing assumptions it needed.

    Attributes:
        returns: One simple return per session after the start.
        gap_sessions: Sessions carried flat inside a missing-price span.
        terminal: Whether the security was delisted inside the contract's life.
    """

    returns: NDArray[np.float64]
    gap_sessions: int
    terminal: bool


class PathCounts(TypedDict, total=True):
    """What building the realized contracts examined and assumed.

    Args:
        decisions: Monthly decision closes used.
        contracts: Contracts built, one per admitted security per decision.
        gap_contracts: Contracts with at least one session inside a
            missing-price span.
        gap_sessions: Sessions carried flat across all contracts.
        terminal_contracts: Contracts whose security was delisted inside
            their life.
    """

    decisions: int
    contracts: int
    gap_contracts: int
    gap_sessions: int
    terminal_contracts: int


def encode_path_counts(counts: PathCounts) -> JSONObject:
    """Lower path counts to JSON.

    Args:
        counts: The counts.

    Returns:
        A JSON object mirroring :class:`PathCounts`.
    """
    return {
        "decisions": counts["decisions"],
        "contracts": counts["contracts"],
        "gap_contracts": counts["gap_contracts"],
        "gap_sessions": counts["gap_sessions"],
        "terminal_contracts": counts["terminal_contracts"],
    }


def decode_path_counts(value: JSONValue) -> PathCounts:
    """Raise path counts out of untrusted JSON.

    Args:
        value: The parsed object.

    Returns:
        The validated counts.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    obj = narrow_json_to_dict(value)
    return PathCounts(
        decisions=require_int(obj, "decisions"),
        contracts=require_int(obj, "contracts"),
        gap_contracts=require_int(obj, "gap_contracts"),
        gap_sessions=require_int(obj, "gap_sessions"),
        terminal_contracts=require_int(obj, "terminal_contracts"),
    )


def _terminal_event(
    outcomes: CrspOutcomes, permno: int, start: str, last: str
) -> JSONObject | None:
    """Find the one delisting dated in ``[start, last)``.

    Args:
        outcomes: Indexed evidence.
        permno: The contract's security.
        start: The contract's start close.
        last: Its maturity session; an event dated there ends nothing early.

    Returns:
        The event, or None.

    Raises:
        ValueError: CRSP-009 for two events inside one contract's life.
    """
    rows = [
        row
        for row in outcomes.delists.get(permno, ())
        if start <= json_str(row, "delistingdt") < last
    ]
    if len(rows) > 1:
        raise ValueError(f"[CRSP-009] one contract cannot hold two terminal events for {permno}")
    return rows[0] if rows else None


def contract_path(
    outcomes: CrspOutcomes, permno: int, start: str, days: tuple[str, ...]
) -> ContractPath:
    """Build one contract's session returns from verified spans.

    Args:
        outcomes: Indexed evidence.
        permno: The contract's security.
        start: The start close, the first span's required predecessor.
        days: The contract's sessions after the start, maturity last.

    Returns:
        The returns and the fixing assumptions they needed.

    Raises:
        ValueError: The span refusals of
            :func:`~covenant_ml.markets.crsp_return_path.crsp_held_spans` and
            the terminal refusals of
            :func:`~covenant_ml.markets.crsp_calendar.crsp_terminal_value`
            propagate; CRSP-009 for two delistings inside the life.
    """
    event = _terminal_event(outcomes, permno, start, days[-1])
    end = days[-1] if event is None else json_str(event, "delistingdt")
    held = days[: bisect_right(days, end)]
    spans = crsp_held_spans(outcomes.returns, permno, start, held)
    growth: NDArray[np.float64] = np.ones(len(days), dtype=np.float64)
    for index, value in spans:
        growth[index] = value
    if event is not None:
        _, multiplier = crsp_terminal_value(event)
        growth[len(held)] = multiplier
    returns: NDArray[np.float64] = growth - 1.0
    return ContractPath(returns, len(held) - len(spans), event is not None)


def realized_contracts(
    sources: CrspSources,
    first: str,
    last: str,
    contract: AsianPutContract,
    rates: tuple[RateObservation, ...],
) -> tuple[RealizedContracts, PathCounts]:
    """Build every realized contract the panel admits between two decision closes.

    Args:
        sources: Verified CIZ sources, train-validation partition only.
        first: Inclusive first decision bound.
        last: Inclusive last decision bound.
        contract: Sets how many sessions each path spans.
        rates: The carry-rate series; each contract uses the latest
            observation on or before its decision close.

    Returns:
        The contracts in decision order and what building them assumed.

    Raises:
        ValueError: CVML-OPT-002 when a contract would outlive the sources'
            sessions; CVML-OPT-003 when no rate is observed by a decision; the
            panel's, span and terminal refusals propagate.
    """
    outcomes = index_crsp_outcomes(sources)
    sessions = outcomes.sessions
    steps = total_sessions(contract)
    permnos: list[int] = []
    months: list[int] = []
    rows: list[NDArray[np.float64]] = []
    carry: list[float] = []
    gaps = gap_contracts = terminals = decisions = 0
    for decision in build_crsp_panel(sources, first, last):
        start = decision.window.decision
        month = crsp_month(start)
        rate = carry_rate(rates, start)
        offset = bisect_left(sessions, start) + 1
        days = sessions[offset : offset + steps]
        if len(days) < steps:
            raise ValueError(f"[{ERR_PATH_SHAPE}] a contract from {start} outlives the sources")
        decisions += 1
        for feature in decision.features:
            path = contract_path(outcomes, feature.permno, start, days)
            permnos.append(feature.permno)
            months.append(month)
            rows.append(path.returns)
            carry.append(rate)
            gaps += path.gap_sessions
            gap_contracts += int(path.gap_sessions > 0)
            terminals += int(path.terminal)
    contracts = RealizedContracts(
        securities=np.asarray(permnos, dtype=np.int64),
        months=np.asarray(months, dtype=np.int64),
        returns=np.vstack(rows) if rows else np.empty((0, steps), dtype=np.float64),
        carry_rates=np.asarray(carry, dtype=np.float64),
    )
    counts = PathCounts(
        decisions=decisions,
        contracts=len(permnos),
        gap_contracts=gap_contracts,
        gap_sessions=gaps,
        terminal_contracts=terminals,
    )
    return contracts, counts


__all__ = [
    "ContractPath",
    "PathCounts",
    "contract_path",
    "decode_path_counts",
    "encode_path_counts",
    "realized_contracts",
]
