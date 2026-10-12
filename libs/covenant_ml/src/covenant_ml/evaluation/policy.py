"""What acting on a model's probabilities is worth.

A probability is not the product; the decision it drives is. Two models with
the same log loss can make different decisions, because only errors near the
point where the best action changes move a decision at all. So the value of a
model is measured here the way an option's exercise policy is: run the
decisions the probabilities imply on held-out rows and average what they
realise.

THE PAYOFF TABLE is the whole interface. For each held-out row and each
action it holds two numbers, the payoff if the row turns out negative and if
it turns out positive. Everything domain-specific is in how the table is
built; the decision rule and its scoring are shared:

* :func:`alert_payoffs` -- covenant monitoring: ``hold`` or ``alert``, with a
  cost for a false alert, a cost for a missed breach and a value for a caught
  one, the same on every row.
* :func:`binary_contract_payoffs` -- a prediction-market contract that pays 1
  if the event happens (Kalshi, Polymarket): ``pass``, ``buy_yes`` at the ask
  of the yes side or ``buy_no`` at the ask of the no side, each net of the
  fee the caller supplies per row. Prices and fees differ per row, which is
  why the table is per row. No exchange's fee schedule is encoded here;
  the caller computes the fee each venue would charge.

THE DECISION RULE is the Bayes action: on each row take the action whose
expected payoff under the model's probability is highest, ties going to the
first action (``hold``, ``pass``). That is the rule a calibrated model is
optimal under, so a model is scored on the decisions it would actually make.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .paired import paired_mean_difference
from .resampling import Resampling
from .types import (
    ERR_INVALID_COST,
    ERR_INVALID_PRICE,
    ERR_LENGTH_MISMATCH,
    ERR_PAYOFF_SHAPE,
    AlertCosts,
    PolicyOutcome,
)

#: The two actions of :func:`alert_payoffs`, in payoff order.
ALERT_ACTIONS = ("hold", "alert")

#: The three actions of :func:`binary_contract_payoffs`, in payoff order.
CONTRACT_ACTIONS = ("pass", "buy_yes", "buy_no")


class PayoffTable(NamedTuple):
    """Payoff of every action on every row, under each outcome.

    Args:
        action_names: The actions, in column order.
        payoffs: Shape (n_rows, n_actions, 2); ``[..., 0]`` is the payoff if
            the row turns out negative, ``[..., 1]`` if positive.
    """

    action_names: tuple[str, ...]
    payoffs: NDArray[np.float64]


def payoff_table(action_names: Sequence[str], payoffs: NDArray[np.float64]) -> PayoffTable:
    """Validate and bind a payoff table.

    Args:
        action_names: Distinct action names, at least two.
        payoffs: Shape (n_rows, len(action_names), 2).

    Returns:
        The table.

    Raises:
        ValueError: If there are fewer than two actions, the names repeat, or
            the array's shape does not match them.
    """
    names = tuple(action_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(
            f"[{ERR_PAYOFF_SHAPE}] a decision needs at least 2 distinct actions, got {names}"
        )
    n_dims: int = payoffs.ndim
    if n_dims != 3:
        raise ValueError(
            f"[{ERR_PAYOFF_SHAPE}] payoffs must have 3 dimensions (row, action, outcome), "
            f"got {n_dims}"
        )
    n_actions: int = int(payoffs.shape[1])
    n_outcomes: int = int(payoffs.shape[2])
    if n_actions != len(names) or n_outcomes != 2:
        raise ValueError(
            f"[{ERR_PAYOFF_SHAPE}] payoffs must have shape (n_rows, {len(names)}, 2), "
            f"got (n_rows, {n_actions}, {n_outcomes})"
        )
    return PayoffTable(action_names=names, payoffs=payoffs)


def _row_count(table: PayoffTable) -> int:
    """Rows a payoff table covers.

    Args:
        table: The payoffs.

    Returns:
        Its first dimension.
    """
    n_rows: int = int(table.payoffs.shape[0])
    return n_rows


def alert_payoffs(n_rows: int, costs: AlertCosts) -> PayoffTable:
    """The covenant-monitoring decision: hold or alert, same costs on every row.

    Args:
        n_rows: Held-out rows.
        costs: Non-negative magnitudes; costs enter the table negated.

    Returns:
        A ``("hold", "alert")`` table.

    Raises:
        ValueError: If any magnitude is negative, which would turn a cost
            into a reward without saying so.
    """
    magnitudes = (costs["true_alert_value"], costs["false_alert_cost"], costs["missed_cost"])
    if any(value < 0.0 for value in magnitudes):
        raise ValueError(f"[{ERR_INVALID_COST}] alert costs must be non-negative, got {costs}")
    payoffs: NDArray[np.float64] = np.zeros((n_rows, 2, 2), dtype=np.float64)
    payoffs[:, 0, 1] = -costs["missed_cost"]
    payoffs[:, 1, 0] = -costs["false_alert_cost"]
    payoffs[:, 1, 1] = costs["true_alert_value"]
    return payoff_table(ALERT_ACTIONS, payoffs)


def binary_contract_payoffs(
    yes_ask: NDArray[np.float64],
    no_ask: NDArray[np.float64],
    yes_fee: NDArray[np.float64],
    no_fee: NDArray[np.float64],
) -> PayoffTable:
    """A contract paying 1 if the event happens: pass, buy yes, or buy no.

    Positive outcome means the event happened. Buying yes at ``p`` with fee
    ``f`` returns ``1 - p - f`` if it did and ``-(p + f)`` if not; buying no
    is the mirror image.

    Args:
        yes_ask: Price to buy one yes contract on each row, in (0, 1).
        no_ask: Price to buy one no contract on each row, in (0, 1).
        yes_fee: Fee per yes contract on each row, non-negative.
        no_fee: Fee per no contract on each row, non-negative.

    Returns:
        A ``("pass", "buy_yes", "buy_no")`` table.

    Raises:
        ValueError: If the arrays differ in length, a price is outside
            (0, 1), or a fee is negative.
    """
    n_rows = len(yes_ask)
    lengths = (len(no_ask), len(yes_fee), len(no_fee))
    if any(length != n_rows for length in lengths):
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] prices and fees must cover {n_rows} rows, got {lengths}"
        )
    for name, prices in (("yes_ask", yes_ask), ("no_ask", no_ask)):
        if bool(np.any(prices <= 0.0)) or bool(np.any(prices >= 1.0)):
            raise ValueError(f"[{ERR_INVALID_PRICE}] {name} must lie in (0, 1)")
    for name, fees in (("yes_fee", yes_fee), ("no_fee", no_fee)):
        if bool(np.any(fees < 0.0)):
            raise ValueError(f"[{ERR_INVALID_PRICE}] {name} must be non-negative")
    yes_cost: NDArray[np.float64] = yes_ask + yes_fee
    no_cost: NDArray[np.float64] = no_ask + no_fee
    payoffs: NDArray[np.float64] = np.zeros((n_rows, 3, 2), dtype=np.float64)
    payoffs[:, 1, 0] = -yes_cost
    payoffs[:, 1, 1] = 1.0 - yes_cost
    payoffs[:, 2, 0] = 1.0 - no_cost
    payoffs[:, 2, 1] = -no_cost
    return payoff_table(CONTRACT_ACTIONS, payoffs)


def choose_actions(proba: NDArray[np.float64], table: PayoffTable) -> NDArray[np.int64]:
    """Take the action with the highest expected payoff on each row.

    Args:
        proba: Positive-class probability per row.
        table: The payoffs.

    Returns:
        Chosen action index per row; ties go to the lower index.

    Raises:
        ValueError: If the probabilities do not cover the table's rows.
    """
    n_rows = _row_count(table)
    if len(proba) != n_rows:
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] probabilities must cover {n_rows} rows, got {len(proba)}"
        )
    positive: NDArray[np.float64] = proba[:, np.newaxis]
    if_negative: NDArray[np.float64] = table.payoffs[:, :, 0]
    if_positive: NDArray[np.float64] = table.payoffs[:, :, 1]
    expected: NDArray[np.float64] = (1.0 - positive) * if_negative + positive * if_positive
    chosen: NDArray[np.intp] = np.argmax(expected, axis=1)
    return chosen.astype(np.int64)


def realized_utility(
    actions: NDArray[np.int64],
    labels: NDArray[np.int64],
    table: PayoffTable,
) -> NDArray[np.float64]:
    """What each row's chosen action actually paid.

    Args:
        actions: Action index per row.
        labels: Binary outcome per row.
        table: The payoffs.

    Returns:
        Realised payoff per row.

    Raises:
        ValueError: If the actions or labels do not cover the table's rows.
    """
    n_rows = _row_count(table)
    if len(actions) != n_rows or len(labels) != n_rows:
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] actions and labels must cover {n_rows} rows, "
            f"got {len(actions)} and {len(labels)}"
        )
    rows: NDArray[np.int64] = np.arange(n_rows, dtype=np.int64)
    realised: NDArray[np.float64] = table.payoffs[rows, actions, labels]
    return realised


class PolicyEvaluation(NamedTuple):
    """One policy's outcome and the per-row payoffs behind it.

    Args:
        outcome: The recorded summary.
        utilities: Realised payoff per row, kept for paired comparison.
    """

    outcome: PolicyOutcome
    utilities: NDArray[np.float64]


def evaluate_actions(
    arm: str,
    actions: NDArray[np.int64],
    labels: NDArray[np.int64],
    table: PayoffTable,
    resampling: Resampling,
    confidence: float,
) -> PolicyEvaluation:
    """Score a fixed set of actions.

    Args:
        arm: The name the policy is reported under.
        actions: Action index per row.
        labels: Binary outcome per row.
        table: The payoffs.
        resampling: Which rows form one independent unit.
        confidence: Coverage of the interval on the mean payoff.

    Returns:
        The policy's outcome and per-row payoffs.

    Raises:
        ValueError: If the actions or labels do not cover the table's rows.
    """
    utilities = realized_utility(actions, labels, table)
    counts: NDArray[np.intp] = np.bincount(actions, minlength=len(table.action_names))
    return PolicyEvaluation(
        outcome={
            "arm": arm,
            "action_counts": [int(counts.flat[index].item()) for index in range(len(counts))],
            "value": paired_mean_difference(utilities, resampling, confidence),
        },
        utilities=utilities,
    )


def evaluate_policy(
    arm: str,
    proba: NDArray[np.float64],
    labels: NDArray[np.int64],
    table: PayoffTable,
    resampling: Resampling,
    confidence: float,
) -> PolicyEvaluation:
    """Score the decisions one model's probabilities imply.

    Args:
        arm: The model's name.
        proba: Its positive-class probability per row.
        labels: Binary outcome per row.
        table: The payoffs.
        resampling: Which rows form one independent unit.
        confidence: Coverage of the interval on the mean payoff.

    Returns:
        The policy's outcome and per-row payoffs.

    Raises:
        ValueError: If the arrays do not cover the table's rows.
    """
    return evaluate_actions(
        arm, choose_actions(proba, table), labels, table, resampling, confidence
    )


__all__ = [
    "ALERT_ACTIONS",
    "CONTRACT_ACTIONS",
    "PayoffTable",
    "PolicyEvaluation",
    "alert_payoffs",
    "binary_contract_payoffs",
    "choose_actions",
    "evaluate_actions",
    "evaluate_policy",
    "payoff_table",
    "realized_utility",
]
