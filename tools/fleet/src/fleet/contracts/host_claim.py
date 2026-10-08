"""A claimed job whose launch is under way, recorded for every runner of its host.

MCPs board task a85ef09e. A node that declares ``elevated`` has two runner
identities (:mod:`fleet.cli.node_agent`), and they are two processes on the
hub. A run reaches the shared ledger only once it is staged, 40 to 80 s after
its claim on serendipity, so until then only the claiming process knew its
grant. On 2026-10-07 at 22:57:23 PDT serendipity's elevated runner claimed
MCPs/scripts/ps-harness, its ordinary runner claimed MCPs/sms-gateway at
22:59:57 and the elevated runner MCPs/execution-elevated at 23:01:34, and
the host held three runs and 6 workers against a pool of 4 cores; the
sms-gateway check (fleet job c858b46b) took 321 s of its 300 s budget. A
:class:`HostClaim` is that grant written where the host's other runner reads
it (:mod:`fleet.core.host_claims`).
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict


class HostClaim(TypedDict):
    """One claimed job's grant, charged to its host until its run is on the ledger.

    Attributes:
        job_id: The queue job.
        runner: The label of the runner identity launching it.
        project: Its project, which every runner of the host leaves out of
            its fits while the launch is under way, as it leaves out a
            project a lease holds.
        workers: The test workers it was granted.
        ram_gb: The memory those workers may hold.
        until_unix: When the claim stops counting whatever became of it: the
            queue job's own claim lease, so a runner that died mid-launch
            stops charging its host when the queue stops holding its job.
    """

    job_id: str
    runner: str
    project: str
    workers: int
    ram_gb: float
    until_unix: int


def encode_host_claim(claim: HostClaim) -> JSONObject:
    """Encode a host claim.

    Args:
        claim: The claim to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "job_id": claim["job_id"],
        "runner": claim["runner"],
        "project": claim["project"],
        "workers": claim["workers"],
        "ram_gb": claim["ram_gb"],
        "until_unix": claim["until_unix"],
    }


def decode_host_claim(value: JSONValue) -> HostClaim:
    """Decode and validate a host claim.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated claim.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or the grant is not positive. A claim of no workers
            charges its host nothing while it holds a launch's project, so
            it is refused rather than read as free room.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"host claim must be a JSON object, got {type(value).__name__}")
    workers = require_int(value, "workers")
    if workers <= 0:
        raise JSONTypeError(
            f"host claim grants {workers} workers; a claimed job holds at least one, and a "
            "claim of none would charge its host nothing while its launch runs"
        )
    return HostClaim(
        job_id=require_str(value, "job_id"),
        runner=require_str(value, "runner"),
        project=require_str(value, "project"),
        workers=workers,
        ram_gb=require_float(value, "ram_gb"),
        until_unix=require_int(value, "until_unix"),
    )


__all__ = ["HostClaim", "decode_host_claim", "encode_host_claim"]
