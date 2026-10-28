"""Private placement and public observations for the existing research queue lane."""

from __future__ import annotations

import pathlib
import re
from enum import StrEnum

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_str,
)
from platform_core.members import require_member
from typing_extensions import ReadOnly, TypedDict

from fleet.contracts.research import require_digest


class ResearchState(StrEnum):
    """Native states the queue consumer can act on."""

    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ResearchObservation(TypedDict):
    """Verified metadata, excluding research data and private process output."""

    state: ReadOnly[ResearchState]
    native_job_id: ReadOnly[str]
    record_sha256: ReadOnly[str | None]


def decode_observation(value: JSONValue) -> ResearchObservation:
    """Validate a native adapter's answer before changing queue state.

    Args:
        value: Untrusted native process output.

    Returns:
        Strict lifecycle and evidence fields.

    Raises:
        ValueError: RESEARCH_OBSERVATION for incomplete or inconsistent evidence.
    """
    row = narrow_json_to_dict(value)
    if set(row) != {"state", "native_job_id", "record_sha256"}:
        raise ValueError("RESEARCH_OBSERVATION: unexpected observation fields")
    state = require_member(row, "state", ResearchState)
    native = require_str(row, "native_job_id")
    digest = row["record_sha256"]
    if state is ResearchState.COMPLETE:
        if not native:
            raise ValueError("RESEARCH_OBSERVATION: completion requires a native allocation")
        pin: str | None = require_digest(require_str(row, "record_sha256"))
    else:
        if digest is not None:
            raise ValueError("RESEARCH_OBSERVATION: unfinished work cannot claim a result digest")
        pin = None
    return ResearchObservation(state=state, native_job_id=native, record_sha256=pin)


def encode_observation(value: ResearchObservation) -> JSONObject:
    """Encode verified metadata for the queue trail.

    Args:
        value: Valid observation.

    Returns:
        JSON metadata without payload data.
    """
    return {
        "state": value["state"].value,
        "native_job_id": value["native_job_id"],
        "record_sha256": value["record_sha256"],
    }


class ResearchWorker(TypedDict):
    """Operator-provisioned placement; none of these paths come from a queue row."""

    tenant: ReadOnly[str]
    node: ReadOnly[str]
    host: ReadOnly[str]
    python: ReadOnly[str]
    root: ReadOnly[str]
    env_file: ReadOnly[str]
    script: ReadOnly[str]


def decode_worker(value: JSONValue) -> ResearchWorker:
    """Require an explicit private placement and safe SSH alias.

    Args:
        value: Operator-owned configuration document.

    Returns:
        Validated placement for one tenant.

    Raises:
        ValueError: RESEARCH_WORKER for undeclared fields, relative paths or unsafe aliases.
    """
    row = narrow_json_to_dict(value)
    if set(row) != {"tenant", "node", "host", "python", "root", "env_file", "script"}:
        raise ValueError("RESEARCH_WORKER: expected an explicit tenant and controller placement")
    for key in ("node", "host"):
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", require_str(row, key)) is None:
            raise ValueError(f"RESEARCH_WORKER: unsafe {key} alias")
    for key in ("python", "root", "env_file", "script"):
        if not pathlib.PurePosixPath(require_str(row, key)).is_absolute():
            raise ValueError(f"RESEARCH_WORKER: {key} must be absolute")
    tenant = require_str(row, "tenant")
    if not tenant:
        raise ValueError("RESEARCH_WORKER: tenant must be explicit")
    return ResearchWorker(
        tenant=tenant,
        node=require_str(row, "node"),
        host=require_str(row, "host"),
        python=require_str(row, "python"),
        root=require_str(row, "root"),
        env_file=require_str(row, "env_file"),
        script=require_str(row, "script"),
    )


def encode_worker(value: ResearchWorker) -> JSONObject:
    """Encode operator placement for provisioning and round-trip verification.

    Args:
        value: Validated private placement.

    Returns:
        Configuration with no embedded credentials.
    """
    return {
        "tenant": value["tenant"],
        "node": value["node"],
        "host": value["host"],
        "python": value["python"],
        "root": value["root"],
        "env_file": value["env_file"],
        "script": value["script"],
    }


__all__ = [
    "ResearchObservation",
    "ResearchState",
    "ResearchWorker",
    "decode_observation",
    "decode_worker",
    "encode_observation",
    "encode_worker",
]
