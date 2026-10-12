"""Shared market research report persistence and complete RunRecord provenance."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint, encode_run_fingerprint
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from platform_core.run_record import Observation, encode_run_record, run_record, run_record_sidecar


def write_market_report(
    report: JSONObject,
    fingerprint: RunFingerprint,
    output: Path,
    archive_paths: Sequence[Path],
    experiment: str,
    label: str,
    observations: tuple[Observation, ...],
) -> Path:
    """Bind exact report bytes, all source files and arithmetic posture to a RunRecord.

    Args:
        report: Complete experiment output.
        fingerprint: Actual arithmetic environment.
        output: New destination path; existing outputs are refused.
        archive_paths: Evidence files to hash and include.
        experiment: Registered experiment name.
        label: Frozen configuration identity.
        observations: Named measured quantities for cross-run comparison.

    Returns:
        Persisted run-record sidecar path.

    Raises:
        FileExistsError: MARKET-027 if either output already exists.
        OSError: Any source or output operation fails.
    """
    sidecar = run_record_sidecar(output)
    if output.exists() or sidecar.exists():
        raise FileExistsError("[MARKET-027] report already exists; choose a new run path")
    sources: list[JSONValue] = [
        {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
        for path in sorted(archive_paths)
    ]
    payload: JSONObject = dict(
        report, sources=sources, fingerprint=encode_run_fingerprint(fingerprint)
    )
    text = dump_json_str(payload, indent=2)
    record = run_record(
        experiment, label, fingerprint, observations, sha256(text.encode()).hexdigest()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(text)
    with sidecar.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(dump_json_str(encode_run_record(record), indent=2))
    return sidecar


__all__ = ["write_market_report"]
