"""Canonical runtime artifact paths for bot and sniffer runs.

This module centralizes the on-disk locations for human logs, structured
event streams, and sniffer capture outputs so both CLI entry points and
tests use the same path model.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from platform_core.json_utils import JSONObject, require_str
from typing_extensions import TypedDict

_RUNS_DIR = Path("runs")
_BOT_DIR = _RUNS_DIR / "bot"
_SNIFF_DIR = _RUNS_DIR / "sniff"


class BotRunArtifactsDict(TypedDict):
    """Canonical artifact paths for one bot run.

    Attributes:
        log_dir: Directory containing bot run artifacts.
        latest_log_path: Stable latest human-readable bot log path.
        archive_log_path: Timestamped archived bot log path.
        latest_events_path: Stable latest structured bot event stream path.
        archive_events_path: Timestamped archived bot event stream path.
        latest_capture_path: Stable latest capture session path for replay.
        archive_capture_path: Timestamped archived capture session path.
    """

    log_dir: str
    latest_log_path: str
    archive_log_path: str
    latest_events_path: str
    archive_events_path: str
    latest_capture_path: str
    archive_capture_path: str


class SniffRunArtifactsDict(TypedDict):
    """Canonical artifact paths for one sniffer run.

    Attributes:
        log_dir: Directory containing sniffer run artifacts.
        latest_log_path: Stable latest human-readable sniffer log path.
        archive_log_path: Timestamped archived sniffer log path.
        latest_events_path: Stable latest structured sniffer event stream path.
        archive_events_path: Timestamped archived sniffer event stream path.
        latest_capture_path: Stable latest capture session path.
        latest_raw_capture_path: Stable latest raw capture path.
        latest_summary_path: Stable latest session summary path.
        archive_capture_path: Timestamped archived capture session path.
        archive_raw_capture_path: Timestamped archived raw capture path.
        archive_summary_path: Timestamped archived session summary path.
    """

    log_dir: str
    latest_log_path: str
    archive_log_path: str
    latest_events_path: str
    archive_events_path: str
    latest_capture_path: str
    latest_raw_capture_path: str
    latest_summary_path: str
    archive_capture_path: str
    archive_raw_capture_path: str
    archive_summary_path: str


def make_run_stamp(now: datetime | None = None) -> str:
    """Return the canonical timestamp stamp for run archives.

    Args:
        now: Optional datetime override for deterministic tests.

    Returns:
        Timestamp string in ``YYYYMMDD-HHMMSS`` format.
    """
    current = now if now is not None else datetime.now()
    return current.strftime("%Y%m%d-%H%M%S")


def build_bot_run_artifacts(stamp: str) -> BotRunArtifactsDict:
    """Build canonical artifact paths for a bot run.

    Args:
        stamp: Timestamp stamp from :func:`make_run_stamp`.

    Returns:
        Bot artifact path bundle.
    """
    return BotRunArtifactsDict(
        log_dir=str(_BOT_DIR),
        latest_log_path=str(_BOT_DIR / "latest.log"),
        archive_log_path=str(_BOT_DIR / f"bot-{stamp}.log"),
        latest_events_path=str(_BOT_DIR / "latest.events.jsonl"),
        archive_events_path=str(_BOT_DIR / f"bot-{stamp}.events.jsonl"),
        latest_capture_path=str(_BOT_DIR / "latest.capture_session.json"),
        archive_capture_path=str(_BOT_DIR / f"bot-{stamp}.capture_session.json"),
    )


def build_sniff_run_artifacts(stamp: str) -> SniffRunArtifactsDict:
    """Build canonical artifact paths for a sniffer run.

    Args:
        stamp: Timestamp stamp from :func:`make_run_stamp`.

    Returns:
        Sniffer artifact path bundle.
    """
    return SniffRunArtifactsDict(
        log_dir=str(_SNIFF_DIR),
        latest_log_path=str(_SNIFF_DIR / "latest.log"),
        archive_log_path=str(_SNIFF_DIR / f"sniff-{stamp}.log"),
        latest_events_path=str(_SNIFF_DIR / "latest.events.jsonl"),
        archive_events_path=str(_SNIFF_DIR / f"sniff-{stamp}.events.jsonl"),
        latest_capture_path=str(_SNIFF_DIR / "latest.capture_session.json"),
        latest_raw_capture_path=str(_SNIFF_DIR / "latest.raw_capture.json"),
        latest_summary_path=str(_SNIFF_DIR / "latest.session_summary.json"),
        archive_capture_path=str(_SNIFF_DIR / f"sniff-{stamp}.capture_session.json"),
        archive_raw_capture_path=str(_SNIFF_DIR / f"sniff-{stamp}.raw_capture.json"),
        archive_summary_path=str(_SNIFF_DIR / f"sniff-{stamp}.session_summary.json"),
    )


def encode_bot_run_artifacts(artifacts: BotRunArtifactsDict) -> JSONObject:
    """Encode bot run artifacts to JSON-compatible data.

    Args:
        artifacts: Bot run artifacts.

    Returns:
        JSON-compatible representation.
    """
    return {
        "log_dir": artifacts["log_dir"],
        "latest_log_path": artifacts["latest_log_path"],
        "archive_log_path": artifacts["archive_log_path"],
        "latest_events_path": artifacts["latest_events_path"],
        "archive_events_path": artifacts["archive_events_path"],
        "latest_capture_path": artifacts["latest_capture_path"],
        "archive_capture_path": artifacts["archive_capture_path"],
    }


def decode_bot_run_artifacts(data: JSONObject) -> BotRunArtifactsDict:
    """Decode bot run artifacts from JSON-compatible data.

    Args:
        data: JSON object to decode.

    Returns:
        Validated bot run artifacts.
    """
    return BotRunArtifactsDict(
        log_dir=require_str(data, "log_dir"),
        latest_log_path=require_str(data, "latest_log_path"),
        archive_log_path=require_str(data, "archive_log_path"),
        latest_events_path=require_str(data, "latest_events_path"),
        archive_events_path=require_str(data, "archive_events_path"),
        latest_capture_path=require_str(data, "latest_capture_path"),
        archive_capture_path=require_str(data, "archive_capture_path"),
    )


def encode_sniff_run_artifacts(artifacts: SniffRunArtifactsDict) -> JSONObject:
    """Encode sniffer run artifacts to JSON-compatible data.

    Args:
        artifacts: Sniffer run artifacts.

    Returns:
        JSON-compatible representation.
    """
    return {
        "log_dir": artifacts["log_dir"],
        "latest_log_path": artifacts["latest_log_path"],
        "archive_log_path": artifacts["archive_log_path"],
        "latest_events_path": artifacts["latest_events_path"],
        "archive_events_path": artifacts["archive_events_path"],
        "latest_capture_path": artifacts["latest_capture_path"],
        "latest_raw_capture_path": artifacts["latest_raw_capture_path"],
        "latest_summary_path": artifacts["latest_summary_path"],
        "archive_capture_path": artifacts["archive_capture_path"],
        "archive_raw_capture_path": artifacts["archive_raw_capture_path"],
        "archive_summary_path": artifacts["archive_summary_path"],
    }


def decode_sniff_run_artifacts(data: JSONObject) -> SniffRunArtifactsDict:
    """Decode sniffer run artifacts from JSON-compatible data.

    Args:
        data: JSON object to decode.

    Returns:
        Validated sniffer run artifacts.
    """
    return SniffRunArtifactsDict(
        log_dir=require_str(data, "log_dir"),
        latest_log_path=require_str(data, "latest_log_path"),
        archive_log_path=require_str(data, "archive_log_path"),
        latest_events_path=require_str(data, "latest_events_path"),
        archive_events_path=require_str(data, "archive_events_path"),
        latest_capture_path=require_str(data, "latest_capture_path"),
        latest_raw_capture_path=require_str(data, "latest_raw_capture_path"),
        latest_summary_path=require_str(data, "latest_summary_path"),
        archive_capture_path=require_str(data, "archive_capture_path"),
        archive_raw_capture_path=require_str(data, "archive_raw_capture_path"),
        archive_summary_path=require_str(data, "archive_summary_path"),
    )


__all__ = [
    "BotRunArtifactsDict",
    "SniffRunArtifactsDict",
    "build_bot_run_artifacts",
    "build_sniff_run_artifacts",
    "decode_bot_run_artifacts",
    "decode_sniff_run_artifacts",
    "encode_bot_run_artifacts",
    "encode_sniff_run_artifacts",
    "make_run_stamp",
]
