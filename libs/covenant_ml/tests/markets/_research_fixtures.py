"""Typed fixtures for research admission; all persistence uses real SQLite."""

from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject
from platform_core.run_record import Observation, encode_run_record, run_record

from covenant_ml.benchmarking.factory import make_benchmark_config
from covenant_ml.benchmarking.types_codec import encode_benchmark_config
from covenant_ml.markets.provenance import market_fingerprint
from covenant_ml.markets.research_records import ResearchPolicy
from tests.benchmarking._cli_fixtures import stand_in_pin


def policy() -> ResearchPolicy:
    return ResearchPolicy(
        protocol_digest="a" * 64,
        dataset_digest="b" * 64,
        code_digest="c" * 64,
        image_digest="d" * 64,
        features=("return_1", "volume_60"),
        seeds=(11, 23),
        max_candidates=2,
        max_seconds=60,
        max_trees=200,
        max_leaves=31,
    )


def proposal() -> JSONObject:
    return {
        "hypothesis": "Lagged return predicts the next observation.",
        "parent": "root",
        "protocol_digest": "a" * 64,
        "partition": "validation",
        "features": ["return_1"],
        "seed": 11,
        "seconds": 60,
        "config": encode_benchmark_config(make_benchmark_config(repeats=1, warmups=0)),
    }


def completion(number: int = 1) -> JSONObject:
    fingerprint = market_fingerprint(stand_in_pin(), config_test_hooks.get_env)
    fingerprint["image_digest"] = "d" * 64
    record = run_record(
        "a" * 64,
        str(number),
        fingerprint,
        (Observation(name="candidate_minus_reference", value=-0.02),),
        "e" * 64,
    )
    return {
        "action": "complete",
        "number": number,
        "observed_ms": 2000,
        "job_id": "slurm:123",
        "dataset_digest": "b" * 64,
        "code_digest": "c" * 64,
        "record": encode_run_record(record),
    }
