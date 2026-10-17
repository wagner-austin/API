"""Shared market research report persistence and complete RunRecord provenance."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint, encode_run_fingerprint
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)
from platform_core.run_record import (
    Observation,
    decode_run_record,
    encode_run_record,
    run_record,
    run_record_sidecar,
)

from . import _test_hooks
from .evidence_location import locate_evidence


def read_verified_market_report(
    path: Path,
    error_code: str,
) -> tuple[JSONObject, tuple[Path, ...], str]:
    """Verify a report's exact bytes, RunRecord payload and every referenced source.

    Args:
        path: Report beside its required RunRecord sidecar.
        error_code: Calling workflow's traceable evidence error code.

    Returns:
        Decoded report, all verified paths including report and sidecar, and payload digest.

    Raises:
        ValueError: A report or source differs from its recorded identity.
        OSError: Required evidence cannot be read.
        TypeError: Required JSON fields are malformed.
    """
    payload = path.read_bytes()
    report = narrow_json_to_dict(load_json_str(payload.decode("utf-8")))
    sidecar = run_record_sidecar(path)
    record = decode_run_record(narrow_json_to_dict(load_json_str(sidecar.read_text("utf-8"))))
    digest = sha256(payload).hexdigest()
    if record["payload_digest"] != digest:
        raise ValueError(f"[{error_code}] report differs from its RunRecord")
    sources = [path, sidecar]
    relocations = _test_hooks.evidence_relocations()
    for value in require_list(report, "sources"):
        source = narrow_json_to_dict(value)
        source_path = locate_evidence(require_str(source, "path"), relocations)
        if sha256(source_path.read_bytes()).hexdigest() != require_str(source, "sha256"):
            raise ValueError(f"[{error_code}] original source evidence changed")
        sources.append(source_path)
    return report, tuple(sources), digest


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


__all__ = ["read_verified_market_report", "write_market_report"]
