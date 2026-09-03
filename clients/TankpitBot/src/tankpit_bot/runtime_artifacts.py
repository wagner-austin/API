"""Canonical runtime artifact paths for bot and sniffer runs.

This module centralizes the on-disk locations for human logs, structured
event streams, and sniffer capture outputs so both CLI entry points and
tests use the same path model.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from platform_core.json_utils import JSONObject, require_str
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks

TANK_REGISTRY_PATH = Path("data") / "tank_registry.json"
"""Measured per-colour tank ranks, keyed account -> world -> colour.

Operator state like ``accounts.json``, not a build artifact: it is
filled by entering each colour once and reading the rank off the wire,
because nothing reports the rank of a colour the account is not
currently playing. ``make release`` copies it into the snapshot for
the same reason it copies ``accounts.json``."""

_RUNS_DIR = Path("runs")

_BOT_DIR = _RUNS_DIR / "bot"
_SNIFF_DIR = _RUNS_DIR / "sniff"
_PROBE_DIR = _RUNS_DIR / "probe"


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


class ProbeRunArtifactsDict(TypedDict):
    """Canonical artifact paths for one action_lab probe run.

    Attributes:
        log_dir: Directory containing probe run artifacts.
        probe_name: Probe identifier (``fuel``, ``equipment``, ``movement``,
            ``teleport``, ``enemy_teleport``, ``fuel_drill``) used in the
            archive filenames so multiple probe kinds can coexist under
            ``runs/probe/``.
        latest_log_path: Stable latest human-readable probe log path.
        archive_log_path: Timestamped archived probe log path.
        latest_events_path: Stable latest structured probe event stream path.
        archive_events_path: Timestamped archived probe event stream path.
    """

    log_dir: str
    probe_name: str
    latest_log_path: str
    archive_log_path: str
    latest_events_path: str
    archive_events_path: str


def make_run_stamp(now: datetime | None = None) -> str:
    """Return the canonical timestamp stamp for run archives.

    Args:
        now: Optional datetime override for deterministic tests.

    Returns:
        Timestamp string in ``YYYYMMDD-HHMMSS`` format.
    """
    current = now if now is not None else datetime.now()
    return current.strftime("%Y%m%d-%H%M%S")


_INSTANCE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def resolve_bot_instance() -> str:
    """Resolve this process's bot-instance namespace from the environment.

    ``TANKPIT_BOT_INSTANCE`` names one of several bots sharing the
    machine (2026-08-06, the two-bots-one-map lift): each instance
    gets its own artifact directory (``runs/bot/<instance>/``) so
    parallel processes never overwrite each other's latest.* files or
    captures. Unset or empty means THE sole-bot namespace
    (``runs/bot/`` directly) — the single-bot layout is the primary
    configuration, not a fallback.

    Returns:
        The validated instance name, or ``""`` for the sole-bot
        namespace.

    Raises:
        ValueError: If the name is not lowercase alphanumeric with
            ``-``/``_`` (max 32 chars) — path separators and dots
            must never reach the filesystem layer.
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_INSTANCE")
    if raw is None or raw == "":
        return ""
    if not _INSTANCE_NAME.match(raw):
        raise ValueError(
            f"TANKPIT_BOT_INSTANCE {raw!r} is not a valid instance name "
            "(lowercase alphanumeric plus -_, max 32 chars)"
        )
    return raw


def bot_run_dir(instance: str) -> Path:
    """Return the artifact directory for one bot instance.

    Args:
        instance: Validated instance name, or ``""`` for the sole-bot
            namespace.

    Returns:
        ``runs/bot`` or ``runs/bot/<instance>``.
    """
    return _BOT_DIR / instance if instance else _BOT_DIR


def build_bot_run_artifacts(stamp: str, instance: str) -> BotRunArtifactsDict:
    """Build canonical artifact paths for a bot run.

    Args:
        stamp: Timestamp stamp from :func:`make_run_stamp`.
        instance: Instance namespace from :func:`resolve_bot_instance`
            (``""`` for the sole-bot layout).

    Returns:
        Bot artifact path bundle.
    """
    run_dir = bot_run_dir(instance)
    return BotRunArtifactsDict(
        log_dir=str(run_dir),
        latest_log_path=str(run_dir / "latest.log"),
        archive_log_path=str(run_dir / f"bot-{stamp}.log"),
        latest_events_path=str(run_dir / "latest.events.jsonl"),
        archive_events_path=str(run_dir / f"bot-{stamp}.events.jsonl"),
        latest_capture_path=str(run_dir / "latest.capture_session.json"),
        archive_capture_path=str(run_dir / f"bot-{stamp}.capture_session.json"),
    )


def build_probe_run_artifacts(
    probe_name: str, stamp: str, runs_root: Path | None = None
) -> ProbeRunArtifactsDict:
    """Build canonical artifact paths for one action_lab probe run.

    Args:
        probe_name: Probe identifier embedded in the archive filenames
            (e.g. ``fuel``, ``equipment``, ``movement``, ``teleport``,
            ``enemy_teleport``, ``fuel_drill``). Must be non-empty.
        stamp: Timestamp stamp from :func:`make_run_stamp`.
        runs_root: Directory the probe artifacts land under, replacing
            the fixed ``runs/`` root. None keeps that root, which is
            right for a workstation and WRONG for a cluster array:
            ``latest.<probe>.log`` is a fixed path, so N tasks sharing a
            node overwrite each other's, and the archive paths collide
            too whenever a sweep holds the stamp still to stop it
            varying the world ([[sim-world-parameterization]]).

    Returns:
        Probe artifact path bundle.

    Raises:
        ValueError: When ``probe_name`` is empty.
    """
    if not probe_name:
        raise ValueError("probe_name must be non-empty")
    probe_dir = _PROBE_DIR if runs_root is None else runs_root / "probe"
    return ProbeRunArtifactsDict(
        log_dir=str(probe_dir),
        probe_name=probe_name,
        latest_log_path=str(probe_dir / f"latest.{probe_name}.log"),
        archive_log_path=str(probe_dir / f"{probe_name}-{stamp}.log"),
        latest_events_path=str(probe_dir / f"latest.{probe_name}.events.jsonl"),
        archive_events_path=str(probe_dir / f"{probe_name}-{stamp}.events.jsonl"),
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


def encode_probe_run_artifacts(artifacts: ProbeRunArtifactsDict) -> JSONObject:
    """Encode probe run artifacts to JSON-compatible data.

    Args:
        artifacts: Probe run artifacts.

    Returns:
        JSON-compatible representation.
    """
    return {
        "log_dir": artifacts["log_dir"],
        "probe_name": artifacts["probe_name"],
        "latest_log_path": artifacts["latest_log_path"],
        "archive_log_path": artifacts["archive_log_path"],
        "latest_events_path": artifacts["latest_events_path"],
        "archive_events_path": artifacts["archive_events_path"],
    }


def decode_probe_run_artifacts(data: JSONObject) -> ProbeRunArtifactsDict:
    """Decode probe run artifacts from JSON-compatible data.

    Args:
        data: JSON object to decode.

    Returns:
        Validated probe run artifacts.
    """
    return ProbeRunArtifactsDict(
        log_dir=require_str(data, "log_dir"),
        probe_name=require_str(data, "probe_name"),
        latest_log_path=require_str(data, "latest_log_path"),
        archive_log_path=require_str(data, "archive_log_path"),
        latest_events_path=require_str(data, "latest_events_path"),
        archive_events_path=require_str(data, "archive_events_path"),
    )


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
    "TANK_REGISTRY_PATH",
    "BotRunArtifactsDict",
    "ProbeRunArtifactsDict",
    "SniffRunArtifactsDict",
    "bot_run_dir",
    "build_bot_run_artifacts",
    "build_probe_run_artifacts",
    "build_sniff_run_artifacts",
    "decode_bot_run_artifacts",
    "decode_probe_run_artifacts",
    "decode_sniff_run_artifacts",
    "encode_bot_run_artifacts",
    "encode_probe_run_artifacts",
    "encode_sniff_run_artifacts",
    "make_run_stamp",
    "resolve_bot_instance",
]
