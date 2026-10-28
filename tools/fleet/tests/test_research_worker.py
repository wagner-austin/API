"""Typed placement and observation boundaries reject misleading completion metadata."""

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from fleet.contracts.research_worker import (
    decode_observation,
    decode_worker,
    encode_observation,
    encode_worker,
)
from tests._research_fixtures import worker


@pytest.mark.parametrize("state", ["running", "failed", "complete"])
def test_observation_roundtrip(state: str) -> None:
    wire: JSONObject = {
        "state": state,
        "native_job_id": "123",
        "record_sha256": "a" * 64 if state == "complete" else None,
    }
    assert encode_observation(decode_observation(wire)) == wire


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"state": "complete", "native_job_id": "", "record_sha256": "a" * 64},
        {"state": "failed", "native_job_id": "123", "record_sha256": "a" * 64},
    ],
)
def test_inconsistent_observation_refuses(value: JSONValue) -> None:
    with pytest.raises(ValueError, match="RESEARCH_OBSERVATION"):
        decode_observation(value)


def test_private_placement_roundtrip_and_invalid_boundaries() -> None:
    placed = worker()
    assert decode_worker(encode_worker(placed)) == placed
    for field, value in (
        ("host", "-oProxyCommand=bad"),
        ("node", "bad;node"),
        ("root", "relative"),
        ("tenant", ""),
    ):
        changed = encode_worker(placed)
        changed[field] = value
        with pytest.raises(ValueError, match="RESEARCH_WORKER"):
            decode_worker(changed)
    with pytest.raises(ValueError, match="RESEARCH_WORKER"):
        decode_worker({**encode_worker(placed), "shell": "forbidden"})
