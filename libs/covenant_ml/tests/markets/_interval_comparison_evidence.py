"""Re-fit captured contracts and independently recount the published comparison."""

from __future__ import annotations

import math
from hashlib import sha256
from pathlib import Path
from statistics import NormalDist

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_float,
    require_dict,
    require_float,
    require_int,
    require_list,
)

from covenant_ml.markets.interval_comparison import compare_intervals
from covenant_ml.markets.validation import read_document


def require_numeric_reproduction(actual: JSONValue, expected: JSONValue) -> None:
    """Compare all fields, allowing only explicit cross-platform float tolerance.

    Args:
        actual: Real reconstruction from native evidence.
        expected: Hash-pinned original result.

    Raises:
        AssertionError: Structure, discrete values or numerical results differ.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            require_numeric_reproduction(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            require_numeric_reproduction(value, reference)
    elif isinstance(expected, float):
        assert isinstance(actual, float)
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
    else:
        assert actual == expected


def require_independent_interval(values: list[float], report: JSONObject) -> None:
    """Recount seven-lag daily uncertainty without the production estimator.

    Args:
        values: Thirty equal-weight calendar-date means.
        report: Published interval to check.

    Raises:
        AssertionError: Mean, standard error, interval or reported units differ.
    """
    assert len(values) == 30
    mean = math.fsum(values) / 30
    residual = [value - mean for value in values]
    variance = math.fsum(value * value for value in residual) / 30
    for lag in range(1, 8):
        covariance = math.fsum(residual[i] * residual[i - lag] for i in range(lag, 30)) / 30
        variance += 2 * (1 - lag / 8) * covariance
    error = math.sqrt(variance / 30)
    quantile = NormalDist().inv_cdf(0.975)
    for field, expected in (
        ("estimate", mean),
        ("std_error", error),
        ("ci_low", mean - quantile * error),
        ("ci_high", mean + quantile * error),
    ):
        assert math.isclose(require_float(report, field), expected, abs_tol=1e-12)
    assert report["method"] == "newey_west"
    assert report["n_units"] == 4  # Four bookkeeping blocks; thirty correlated dates.


def verify_published_comparison(cohort: JSONObject, quotes: JSONObject) -> None:
    """Execute the real frozen fit using the already reconstructed native panels.

    Args:
        cohort: Full cohort rebuilt from hash-verified exchange envelopes.
        quotes: Full sixty-day quotes rebuilt from the same native archive.

    Raises:
        AssertionError: Pinned evidence, predictions, scores or cash flows differ.
    """
    root = Path(__file__).parent / "fixtures"
    plan_path = root / "interval-comparison-plan-20261015.json"
    result_path = root / "interval-comparison-results-20261015.json"
    assert sha256(plan_path.read_bytes()).hexdigest() == (
        "dea2e2db959a30371c10aae1b69b240c78bbb7bfd430cbe9aa0485fc76a5c47b"
    )
    assert sha256(result_path.read_bytes()).hexdigest() == (
        "985e3dd51ee3b9c7f7dd2fd6d4ee571f78e58fef01bbe80505b29c0f75147f94"
    )
    expected = read_document(result_path)
    actual = compare_intervals(cohort, quotes, read_document(plan_path))
    metadata = {"sources", "fingerprint", "implementation_sha256"}
    require_numeric_reproduction(
        actual, {key: value for key, value in expected.items() if key not in metadata}
    )
    assert (actual["training_count"], actual["evaluation_count"]) == (5694, 5696)
    assert actual["exclusions"] == actual["purged_training_labels"] == []
    rows = [narrow_json_to_dict(value) for value in require_list(actual, "evaluation_rows")]
    market_losses = [
        (require_int(row, "yes") - (require_int(row, "bid") + require_int(row, "ask")) / 20000) ** 2
        for row in rows
    ]
    for name, document in require_dict(actual, "arms").items():
        arm = narrow_json_to_dict(document)
        probabilities = [narrow_json_to_float(p) for p in require_list(arm, "predictions")]
        loss = [
            (require_int(row, "yes") - p) ** 2 for row, p in zip(rows, probabilities, strict=True)
        ]
        assert math.isclose(require_float(arm, "brier"), math.fsum(loss) / len(rows), abs_tol=1e-12)
        if name != "raw_market":
            daily: dict[int, list[float]] = {}
            for row, own, market in zip(rows, loss, market_losses, strict=True):
                daily.setdefault(require_int(row, "close_ms") // 86400000, []).append(own - market)
            means = [math.fsum(daily[day]) / len(daily[day]) for day in sorted(daily)]
            paired = require_dict(require_dict(actual, "paired_daily_brier_minus_market"), name)
            require_independent_interval(means, require_dict(paired, "interval"))
        for value in require_list(arm, "cost_scenarios"):
            policy = narrow_json_to_dict(value)
            friction = require_int(policy, "friction")
            actions: list[JSONValue] = []
            payouts: list[JSONValue] = []
            fees: list[JSONValue] = []
            total, trades = 0, 0
            for row, p in zip(rows, probabilities, strict=True):
                yes_price, no_price = require_int(row, "ask"), 10000 - require_int(row, "bid")
                costs = [
                    ((700 * price * (10000 - price) + 9999999999) // 10000000000) * 100
                    for price in (yes_price, no_price)
                ]
                utilities = (
                    0.0,
                    p * 10000 - yes_price - costs[0] - friction - 200,
                    (1 - p) * 10000 - no_price - costs[1] - friction - 200,
                )
                choice = utilities.index(max(utilities))
                cash = (
                    0,
                    require_int(row, "yes") * 10000 - yes_price - costs[0] - friction,
                    (1 - require_int(row, "yes")) * 10000 - no_price - costs[1] - friction,
                )[choice]
                actions.append(("pass", "buy_yes", "buy_no")[choice])
                payouts.append(cash)
                fees.append((0, *costs)[choice])
                total += cash
                trades += choice != 0
            assert policy["actions"] == actions
            assert policy["fees"] == fees
            assert policy["pnl_by_row"] == payouts
            assert (policy["pnl"], policy["trades"]) == (total, trades)
