"""Realized Asian-put contracts from verified CRSP CIZ sources.

One hypothetical contract starts at each monthly decision close on each
security the frozen panel admits there (:func:`~covenant_ml.markets.crsp_panel.build_crsp_panel`:
effective share and exchange codes, minimum price, history and liquidity
rank, all known at the decision). Its path is that security's returns over
the following sessions, normalized to the contract's spot, so no price level
or fill is needed at the start.

TWO FIXING ASSUMPTIONS, both counted in :class:`PathCounts` rather than hidden:

* A session inside a CIZ unpriced span (MP/MR return rows whose price flag is
  MP, or SU for a suspension under protocol version 4 and later) has no price,
  but an Asian average fixes on every session. The last observed close is
  carried through the span, and the NA/P1-P9 return that closes it is applied
  on the closing session
  (:func:`~covenant_ml.markets.crsp_return_path.crsp_resolve_held`), so no
  later information reaches an earlier fixing. Carrying a close through a
  vendor-declared trading suspension is a fixing-disruption assumption, counted
  with the other carried sessions.
* A security delisted inside the contract's life stops at its delisting date;
  the ``stkdelists`` terminal multiplier
  (:func:`~covenant_ml.markets.crsp_calendar.crsp_resolve_terminal`) is applied
  once on the next session and the value is held flat to maturity. The
  daily rows after the delisting date are never read, so the terminal outcome
  is counted once, as in the holdings evaluator.

ONE EXCLUSION, also counted: a contract whose life holds a return or a
delisting value the vendor's records cannot supply is not built, and
:class:`PathCounts` counts it under the reason
(:func:`~covenant_ml.markets.crsp_return_path.crsp_resolve_held`,
:func:`~covenant_ml.markets.crsp_calendar.crsp_resolve_terminal`). One such
reason is a return the vendor itself flags missing, such as MV (missing
corporate action value) or GP (gap between prices too large). No value is
imputed for it. The exclusion reads the contract's whole life, so the
population studied is the contracts whose life the vendor fully measured;
the counts say how many that left out and why.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_int,
)
from platform_core.json_utils import require_str as json_str
from typing_extensions import TypedDict

from ..markets.crsp_calendar import crsp_month, crsp_resolve_terminal
from ..markets.crsp_extract import CrspSources
from ..markets.crsp_holdings import CrspOutcomes, index_crsp_outcomes
from ..markets.crsp_panel import build_crsp_panel
from ..markets.crsp_return_path import crsp_resolve_held
from .paths import total_sessions
from .rates import RateObservation, carry_rate
from .study import RealizedContracts
from .types import ERR_PATH_SHAPE, AsianPutContract


class ContractPath(NamedTuple):
    """One contract's session returns and the fixing assumptions it needed.

    Attributes:
        returns: One simple return per session after the start; empty for an
            excluded contract.
        gap_sessions: Sessions carried flat inside a missing-price span.
        terminal: Whether the security was delisted inside the contract's life.
        exclusion: None for a built contract, else why the vendor's records
            cannot value its life.
    """

    returns: NDArray[np.float64]
    gap_sessions: int
    terminal: bool
    exclusion: str | None


def _excluded(reason: str) -> ContractPath:
    """Record a contract that is not built.

    Args:
        reason: Why the vendor's records cannot value its life.

    Returns:
        A path with no returns and no fixing assumptions.
    """
    return ContractPath(np.empty(0, dtype=np.float64), 0, False, reason)


class PathCounts(TypedDict, total=True):
    """What building the realized contracts examined and assumed.

    Args:
        decisions: Monthly decision closes used.
        contracts: Contracts built, one per admitted security per decision
            whose life the vendor's records value.
        gap_contracts: Built contracts with at least one session inside a
            missing-price span.
        gap_sessions: Sessions carried flat across the built contracts.
        terminal_contracts: Built contracts whose security was delisted
            inside their life.
        excluded: Admitted contracts not built, by reason, in reason order.
    """

    decisions: int
    contracts: int
    gap_contracts: int
    gap_sessions: int
    terminal_contracts: int
    excluded: dict[str, int]


def encode_path_counts(counts: PathCounts) -> JSONObject:
    """Lower path counts to JSON.

    Args:
        counts: The counts.

    Returns:
        A JSON object mirroring :class:`PathCounts`.
    """
    excluded: JSONObject = dict(counts["excluded"])
    return {
        "decisions": counts["decisions"],
        "contracts": counts["contracts"],
        "gap_contracts": counts["gap_contracts"],
        "gap_sessions": counts["gap_sessions"],
        "terminal_contracts": counts["terminal_contracts"],
        "excluded": excluded,
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
    excluded = require_dict(obj, "excluded")
    return PathCounts(
        decisions=require_int(obj, "decisions"),
        contracts=require_int(obj, "contracts"),
        gap_contracts=require_int(obj, "gap_contracts"),
        gap_sessions=require_int(obj, "gap_sessions"),
        terminal_contracts=require_int(obj, "terminal_contracts"),
        excluded={reason: require_int(excluded, reason) for reason in excluded},
    )


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
        The returns and the fixing assumptions they needed, or an excluded
        path naming why the vendor's records cannot value the life: a
        held-return refusal of
        :func:`~covenant_ml.markets.crsp_return_path.crsp_resolve_held`, a
        terminal refusal of
        :func:`~covenant_ml.markets.crsp_calendar.crsp_resolve_terminal`, or
        ``two delistings`` for two events dated before maturity.
    """
    events = [
        row
        for row in outcomes.delists.get(permno, ())
        if start <= json_str(row, "delistingdt") < days[-1]
    ]
    if len(events) > 1:
        return _excluded("two delistings")
    event = events[0] if events else None
    end = days[-1] if event is None else json_str(event, "delistingdt")
    held = days[: bisect_right(days, end)]
    resolution = crsp_resolve_held(outcomes.returns, permno, start, held)
    if resolution.refusal is not None:
        return _excluded(resolution.refusal.reason)
    growth: NDArray[np.float64] = np.ones(len(days), dtype=np.float64)
    for index, value in resolution.spans:
        growth[index] = value
    if event is not None:
        terminal = crsp_resolve_terminal(event)
        if terminal.refusal is not None:
            return _excluded(terminal.refusal)
        growth[len(held)] = terminal.multiplier
    returns: NDArray[np.float64] = growth - 1.0
    return ContractPath(returns, len(held) - len(resolution.spans), event is not None, None)


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
        The contracts in decision order and what building them assumed and
        excluded.

    Raises:
        ValueError: CVML-OPT-002 when a contract would outlive the sources'
            sessions; CVML-OPT-003 when no rate is observed by a decision; the
            panel's refusals propagate.
    """
    outcomes = index_crsp_outcomes(sources)
    sessions = outcomes.sessions
    steps = total_sessions(contract)
    permnos: list[int] = []
    months: list[int] = []
    rows: list[NDArray[np.float64]] = []
    carry: list[float] = []
    excluded: dict[str, int] = {}
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
            if path.exclusion is not None:
                excluded[path.exclusion] = excluded.get(path.exclusion, 0) + 1
                continue
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
        excluded=dict(sorted(excluded.items())),
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
