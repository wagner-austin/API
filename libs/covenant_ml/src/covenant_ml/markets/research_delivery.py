"""Move one journal-authorized final grant to evaluator-private native storage."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import dump_json_str
from platform_core.stage_manifest import (
    StagedFile,
    StageManifest,
    decode_stage_manifest,
    encode_stage_manifest,
)

from . import _test_hooks
from .research_cluster import ResearchCluster, attempt_identity
from .research_execution import require_trial_attempt
from .research_journal import ResearchAttempt
from .research_release import consume_final_grant, encode_final_grant
from .research_trial_records import ResearchTrial


def deliver_final_grant(
    ledger: Path,
    trial: ResearchTrial,
    cluster: ResearchCluster,
    directory: Path,
    final_manifest: Path,
    destination: str,
) -> ResearchAttempt:
    """Consume the local released grant before its sole private transport attempt.

    Args:
        ledger: Original evaluator admission journal; never copied to compute.
        trial: Exact selected request.
        cluster: Existing evaluator-owned controller and workspace configuration.
        directory: Private source grant; metadata and consumption marker are retained.
        final_manifest: Exact encrypted final manifest bound by selection.
        destination: New directory beneath existing private native evaluator storage.

    Returns:
        Original released journal row after verified delivery; no key material.

    Raises:
        ValueError: RESEARCH_FINAL_DELIVERY for an unreleased or different final bundle.
        RuntimeError: Transport failed; grant remains consumed and must not be resent.
        OSError: Grant consumption, metadata persistence or subprocess launch fails.
    """
    row = require_trial_attempt(ledger, trial)
    if row["selection"] != "released" or (
        sha256(final_manifest.read_bytes()).hexdigest() != row["final_manifest_digest"]
    ):
        raise ValueError("RESEARCH_FINAL_DELIVERY: require the journal's released final bundle")
    grant, identity = consume_final_grant(directory, trial, final_manifest)
    parts = (
        ("release.json", dump_json_str(encode_final_grant(grant)).encode()),
        ("final.agekey", (str(identity) + "\n").encode()),
    )
    manifest = decode_stage_manifest(
        encode_stage_manifest(
            StageManifest(
                destination=destination,
                files=[
                    StagedFile(name=name, size_bytes=len(data), sha256=sha256(data).hexdigest())
                    for name, data in parts
                ],
                provenance={
                    "research_attempt": attempt_identity(trial),
                    "request_sha256": grant["request_digest"],
                    "final_manifest_sha256": grant["manifest_digest"],
                },
            )
        )
    )
    manifest_path = directory / "delivery.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(encode_stage_manifest(manifest)))
    command = (
        cluster["python"],
        "-m",
        "hpc3.cli.private_stage",
        "--config",
        cluster["workspace"],
        "--manifest",
        str(manifest_path),
    )
    code = _test_hooks.campaign_runner(
        command,
        directory,
        directory / "delivery.log",
        stdin_bytes=b"".join(data for _, data in parts),
    )
    if code != 0:
        raise RuntimeError(
            "RESEARCH_FINAL_DELIVERY: private transport failed; grant consumed, do not resend"
        )
    return row


__all__ = ["deliver_final_grant"]
