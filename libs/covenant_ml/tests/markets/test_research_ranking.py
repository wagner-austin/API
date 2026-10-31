"""Seed aggregation and ranking use real shared records and exact calendar pairs."""

import pytest
from platform_core.json_utils import JSONValue, require_dict
from platform_core.run_record import Observation, decode_run_record

from covenant_ml.benchmarking.types import BenchmarkConfig
from covenant_ml.markets.research_ranking import SeedResult, winning_attempt
from covenant_ml.markets.research_records import (
    candidate_digest,
    decode_research_policy,
    decode_research_proposal,
    encode_research_policy,
)
from covenant_ml.markets.research_selection_records import (
    decode_research_selection,
    encode_research_selection,
)

from ._research_fixtures import completion, policy, proposal


def evidence(number: int, seed: int, trees: int = 200) -> SeedResult:
    """Construct two chronological months under one admitted model configuration."""
    request = proposal()
    request["seed"] = seed
    require_dict(request, "config")["n_estimators"] = trees
    return SeedResult(
        number,
        decode_research_proposal(request),
        decode_run_record(require_dict(completion(number), "record")),
    )


def _returns(item: SeedResult, first: float, second: float) -> SeedResult:
    """Replace the two model observations while retaining the measured calendar."""
    rows: list[Observation] = []
    for row in item.record["observations"]:
        value = row["value"]
        if row["name"].startswith("monthly_net_return.lightgbm"):
            value = first if ".2019-01-31." in row["name"] else second
        rows.append(Observation(name=row["name"], value=value))
    return item._replace(record={**item.record, "observations": tuple(rows)})


def test_complete_seed_mean_wins_not_best_seed_and_order_does_not_matter() -> None:
    baseline = (_returns(evidence(1, 11), 0.02, 0.04), _returns(evidence(2, 23), 0.04, 0.02))
    alternative = (
        _returns(evidence(3, 11, 100), 0.5, 0.5),
        _returns(evidence(4, 23, 100), -0.5, -0.5),
    )
    assert winning_attempt(policy(), (*baseline, *alternative)) == 1
    assert winning_attempt(policy(), (*reversed(alternative), *reversed(baseline))) == 1


def test_incomplete_optional_candidate_is_ineligible_and_negative_winner_is_preserved() -> None:
    complete = (_returns(evidence(1, 11), -0.2, -0.1), _returns(evidence(2, 23), -0.1, -0.2))
    incomplete = _returns(evidence(3, 11, 100), 0.5, 0.5)
    assert winning_attempt(policy(), (*complete, incomplete)) == 1


def test_exact_tie_prefers_fewer_leaves() -> None:
    original = evidence(3, 11, 100)
    config: BenchmarkConfig = {**original.proposal["config"], "num_leaves": 16}
    lower = original._replace(proposal={**original.proposal, "config": config})
    lower_second = lower._replace(
        number=4,
        proposal={**lower.proposal, "seed": 23},
        record={**lower.record, "label": "4"},
    )
    rows = tuple(
        _returns(item, 0.1, 0.2) for item in (evidence(1, 11), evidence(2, 23), lower, lower_second)
    )
    assert winning_attempt(policy(), rows) == 3


def test_exact_tie_uses_candidate_identity_after_leaf_count() -> None:
    rows = tuple(
        _returns(item, 0.1, 0.2)
        for item in (evidence(1, 11), evidence(2, 23), evidence(3, 11, 100), evidence(4, 23, 100))
    )
    winner = winning_attempt(policy(), rows)
    selected = rows[winner - 1]
    other = rows[2] if winner == 1 else rows[0]
    assert winner in (1, 3)
    assert candidate_digest(selected.proposal) < candidate_digest(other.proposal)
    assert winning_attempt(policy(), tuple(reversed(rows))) == winner


@pytest.mark.parametrize(
    "rows", [(), (evidence(1, 11),), (evidence(1, 11, 100), evidence(2, 23, 100))]
)
def test_required_candidates_need_every_seed(rows: tuple[SeedResult, ...]) -> None:
    with pytest.raises(ValueError, match="RESEARCH_SELECTION_SEEDS"):
        winning_attempt(policy(), rows)


def test_duplicate_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="RESEARCH_SELECTION_SEEDS"):
        winning_attempt(policy(), (evidence(1, 11), evidence(2, 11)))


def test_all_five_frozen_seeds_are_required_and_final_seed_is_not_best_seed() -> None:
    document = encode_research_policy(policy())
    document["seeds"] = [11, 23, 37, 53, 71]
    fixed = decode_research_policy(document)
    rows = tuple(evidence(number, seed) for number, seed in enumerate(fixed["seeds"], start=1))
    with pytest.raises(ValueError, match="RESEARCH_SELECTION_SEEDS"):
        winning_attempt(fixed, rows[:-1])
    assert winning_attempt(fixed, rows) == 1


def test_consecutive_windows_must_share_the_same_execution_close() -> None:
    item = evidence(1, 11)
    rows = tuple(
        Observation(
            name=row["name"].replace("2019-02-28.2019-03-01", "2019-02-28.2019-03-04"),
            value=row["value"],
        )
        for row in item.record["observations"]
    )
    item = item._replace(record={**item.record, "observations": rows})
    with pytest.raises(ValueError, match="RESEARCH_SELECTION_CALENDAR"):
        winning_attempt(policy(), (item,))


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate_month",
        "bad_entry",
        "bad_exit",
        "shape",
        "nonfinite",
        "paired_date",
        "reference",
        "overflow",
    ],
)
def test_incomplete_or_changed_monthly_evidence_is_refused(fault: str) -> None:
    first, second = evidence(1, 11), evidence(2, 23)
    observations = list(second.record["observations"])
    index = next(
        i
        for i, row in enumerate(observations)
        if row["name"].startswith("monthly_net_return.lightgbm")
    )
    chosen = observations[index]
    if fault == "missing":
        observations.pop(index)
    elif fault == "duplicate_month":
        observations.append({**chosen, "name": chosen["name"].replace("2019-01-31", "2019-01-30")})
    elif fault == "bad_entry":
        observations[index] = {**chosen, "name": chosen["name"].replace("2019-02-01", "2019-04-01")}
    elif fault == "bad_exit":
        observations[index] = {**chosen, "name": chosen["name"].replace("2019-03-01", "2019-04-01")}
    elif fault == "shape":
        observations[index] = {**chosen, "name": chosen["name"] + ".extra"}
    elif fault == "nonfinite":
        observations[index] = {**chosen, "value": float("inf")}
    elif fault == "paired_date":
        observations[index] = {**chosen, "name": chosen["name"].replace("2019-01-31", "2019-01-30")}
    elif fault == "reference":
        target = next(
            i
            for i, row in enumerate(observations)
            if row["name"].startswith("monthly_net_return.momentum")
        )
        observations[target] = {**observations[target], "value": 0.001}
    else:
        observations[index] = {**chosen, "value": 1e308}
        target = next(
            i
            for i, row in enumerate(observations)
            if row["name"].startswith("monthly_net_return.momentum")
        )
        observations[target] = {**observations[target], "value": -1e308}
    second = second._replace(record={**second.record, "observations": tuple(observations)})
    with pytest.raises(ValueError, match=r"RESEARCH_SELECTION_(CALENDAR|SCORE)"):
        winning_attempt(policy(), (first, second))


@pytest.mark.parametrize("field", ["experiment", "label", "image", "payload", "protocol"])
def test_changed_result_lineage_is_refused(field: str) -> None:
    item = evidence(1, 11)
    record = item.record
    if field == "image":
        record = {**record, "fingerprint": {**record["fingerprint"], "image_digest": "f" * 64}}
    elif field == "payload":
        record = {**record, "payload_digest": ""}
    elif field == "label":
        record = {**record, "label": "99"}
    elif field == "experiment":
        record = {**record, "experiment": "f" * 64}
    else:
        item = item._replace(proposal={**item.proposal, "protocol_digest": "f" * 64})
    with pytest.raises(ValueError, match="RESEARCH_SELECTION_LINEAGE"):
        winning_attempt(policy(), (item._replace(record=record),))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("extra", True),
        ("primary", "momentum"),
        ("reference", "equal_weight"),
        ("required_candidates", []),
        ("required_candidates", ["a" * 64, "a" * 64]),
    ],
)
def test_selection_codec_refuses_ambiguous_settings(key: str, value: JSONValue) -> None:
    encoded = encode_research_selection(policy()["selection"])
    assert decode_research_selection(encoded) == policy()["selection"]
    encoded[key] = value
    with pytest.raises(ValueError, match="RESEARCH_SELECTION_POLICY"):
        decode_research_selection(encoded)


@pytest.mark.parametrize("fault", ["seed", "ceiling"])
def test_selection_must_fit_the_same_admission_policy(fault: str) -> None:
    encoded = encode_research_policy(policy())
    selection = require_dict(encoded, "selection")
    if fault == "seed":
        selection["final_seed"] = 999
    else:
        selection["required_candidates"] = ["a" * 64, "b" * 64, "c" * 64]
    with pytest.raises(ValueError, match="RESEARCH_SELECTION_POLICY"):
        decode_research_policy(encoded)


__all__ = ["evidence"]
