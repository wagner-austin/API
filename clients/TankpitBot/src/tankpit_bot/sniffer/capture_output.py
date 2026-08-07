"""Write a captured session to disk.

The per-run outputs and the grouped archive write the sniffer emits at
teardown.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot.runtime_artifacts import SniffRunArtifactsDict


def _write_capture_outputs(
    output_path: Path,
    capture_json: str,
    summary_json: str,
    *,
    runtime_artifacts: SniffRunArtifactsDict | None,
) -> None:
    """Persist requested and canonical sniffer outputs.

    Args:
        output_path: Requested capture session path.
        capture_json: Serialized capture session JSON.
        summary_json: Serialized session summary JSON.
        runtime_artifacts: Optional canonical latest/archive artifact bundle.
    """
    output_dir = output_path.parent
    raw_path = output_dir / "raw_capture.json"
    summary_path = output_dir / "session_summary.json"
    _write_capture_group(output_path, raw_path, summary_path, capture_json, summary_json)
    if runtime_artifacts is None:
        return
    _write_capture_group(
        Path(runtime_artifacts["latest_capture_path"]),
        Path(runtime_artifacts["latest_raw_capture_path"]),
        Path(runtime_artifacts["latest_summary_path"]),
        capture_json,
        summary_json,
    )
    _write_capture_group(
        Path(runtime_artifacts["archive_capture_path"]),
        Path(runtime_artifacts["archive_raw_capture_path"]),
        Path(runtime_artifacts["archive_summary_path"]),
        capture_json,
        summary_json,
    )


def _write_capture_group(
    capture_path: Path,
    raw_path: Path,
    summary_path: Path,
    capture_json: str,
    summary_json: str,
) -> None:
    """Write one complete capture/session-summary output group.

    Args:
        capture_path: Capture session JSON path.
        raw_path: Raw capture mirror path.
        summary_path: Session summary path.
        capture_json: Serialized capture session JSON.
        summary_json: Serialized session summary JSON.
    """
    _test_hooks.write_text(raw_path, capture_json)
    _test_hooks.write_text(summary_path, summary_json)
    _test_hooks.write_text(capture_path, capture_json)


__all__ = []
