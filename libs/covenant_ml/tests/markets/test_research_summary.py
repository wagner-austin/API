"""Independent block-index oracle and strict evidence checks for seed summaries."""

from hashlib import sha256
from statistics import fmean, quantiles, stdev

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import dump_json_str, narrow_json_to_dict, require_dict, require_list
from platform_core.run_record import encode_run_record

from covenant_ml.markets.research_ranking import winning_attempt
from covenant_ml.markets.research_records import candidate_digest, encode_research_proposal
from covenant_ml.markets.research_summary import validation_summary

from ._research_summary_fixtures import summary_evidence


def test_seed_average_matches_independent_moving_block_index_oracle() -> None:
    policy, evaluation, rows = summary_evidence()
    document, observations = validation_summary(policy, evaluation, rows)
    assert validation_summary(policy, evaluation, tuple(reversed(rows))) == (document, observations)
    assert winning_attempt(policy, rows) == 1
    assert document["selection_uses_interval"] is False
    assert document["interpretation"] == (
        "Descriptive after-search validation uncertainty; no multiplicity adjustment."
    )
    assert document["final_scoring"] == (
        "The selected candidate uses its fixed final seed, not this seed average."
    )
    results = [narrow_json_to_dict(value) for value in require_list(document, "results")]
    assert [item["arm"] for item in results] == ["cleargbm", "lightgbm", "xgboost", "equal_weight"]
    assert len(observations) == 20
    for arm_index, result in enumerate(results):
        expected = [((month // 6 - 2) * 4 + arm_index) / 1024 for month in range(36)]
        assert result["mean_paired_differences"] == expected
        assert (result["n_months"], result["n_seeds"], result["cost_bps"]) == (36, 5, 10)
        assert result["candidate"] == candidate_digest(rows[0].proposal)
        assert result["reference"] == "momentum"
        assert len(require_list(result, "calendar")) == 36
        assert narrow_json_to_dict(require_list(result, "calendar")[0]) == {
            "decision": "2019-01-31",
            "entry": "2019-02-01",
            "exit": "2019-03-01",
        }
        # Sample actual contiguous indexes independently, without production weights.
        rng = np.random.default_rng(evaluation["design"]["bootstrap_seed"])
        starts: NDArray[np.int64] = rng.integers(0, 25, size=(10000, 3), dtype=np.int64)
        draws: list[float] = []
        for draw in range(10000):
            sampled: list[float] = []
            for block in range(3):
                start: np.int64 = starts[draw, block]
                sampled.extend(expected[int(start) : int(start) + 12])
            draws.append(fmean(sampled))
        interval = require_dict(result, "interval")
        assert interval["estimate"] == fmean(expected)
        assert interval["std_error"] == pytest.approx(stdev(draws), abs=1e-15)
        bounds = quantiles(draws, n=40, method="inclusive")
        assert interval["ci_low"] == pytest.approx(bounds[0], abs=1e-15)
        assert interval["ci_high"] == pytest.approx(bounds[-1], abs=1e-15)
        tail = min(sum(value <= 0 for value in draws), sum(value >= 0 for value in draws))
        assert interval["p_value"] == min(1.0, 2 * (tail + 1) / 10001)
        assert (interval["confidence"], interval["n_units"]) == (0.95, 3)
        inputs = [narrow_json_to_dict(item) for item in require_list(result, "inputs")]
        assert inputs == [
            {
                "attempt": row.number,
                "proposal": encode_research_proposal(row.proposal),
                "record_sha256": sha256(
                    dump_json_str(encode_run_record(row.record)).encode()
                ).hexdigest(),
                "payload_sha256": row.record["payload_digest"],
            }
            for row in rows
        ]
        emitted = observations[arm_index * 5 : (arm_index + 1) * 5]
        assert [item["value"] for item in emitted] == [
            interval["estimate"],
            interval["std_error"],
            interval["ci_low"],
            interval["ci_high"],
            interval["p_value"],
        ]
        assert [item["name"] for item in emitted] == [
            f"{result['candidate']}.{result['arm']}_vs_momentum.{metric}"
            for metric in ("estimate", "std_error", "ci_low", "ci_high", "p_value")
        ]


def test_changed_frozen_evaluation_is_refused() -> None:
    policy, evaluation, rows = summary_evidence()
    with pytest.raises(ValueError, match="RESEARCH_SUMMARY_PROTOCOL"):
        validation_summary(policy, {**evaluation, "confidence": 0.9}, rows)


@pytest.mark.parametrize(
    "fault", ["missing_seed", "duplicate_seed", "missing_arm", "calendar", "nonfinite", "reference"]
)
def test_uncertainty_rejects_incomplete_or_changed_seed_evidence(fault: str) -> None:
    policy, evaluation, rows = summary_evidence()
    if fault == "missing_seed":
        rows = rows[:-1]
    elif fault == "duplicate_seed":
        rows = (*rows, rows[0])
    else:
        item = rows[-1]
        values = list(item.record["observations"])
        index = next(i for i, value in enumerate(values) if ".cleargbm." in value["name"])
        if fault == "missing_arm":
            values.pop(index)
        elif fault == "calendar":
            values[index] = {
                **values[index],
                "name": values[index]["name"].replace("2019-01-31", "2019-01-30"),
            }
        elif fault == "nonfinite":
            values[index] = {**values[index], "value": float("nan")}
        else:
            reference = next(i for i, value in enumerate(values) if ".momentum." in value["name"])
            values[reference] = {**values[reference], "value": 0.5}
        rows = (*rows[:-1], item._replace(record={**item.record, "observations": tuple(values)}))
    with pytest.raises(ValueError, match=r"RESEARCH_SELECTION_(SEEDS|CALENDAR)"):
        validation_summary(policy, evaluation, rows)
