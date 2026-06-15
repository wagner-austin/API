"""Analyze command-response timing from capture session files.

Extracts every sent command, correlates it with the server's response,
and computes timing statistics: command→response latency, inter-command
gaps, and server-side cooldowns.  Works on both bot and human captures.
"""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.json_utils import JSONObject, load_json_str
from typing_extensions import TypedDict

from scripts import _test_hooks
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.types import CapturedMessage, decode_captured_message


class CommandTimingDict(TypedDict):
    """Timing record for a single sent command.

    Attributes:
        sent_ms: Timestamp when the command was sent.
        label: Bot-side label for the command.
        response_ms: Timestamp of the first server response after the command.
        latency_ms: Time between send and first response.
    """

    sent_ms: int
    label: str
    response_ms: int
    latency_ms: int


class TimingReportDict(TypedDict):
    """Aggregate timing report for a capture session.

    Attributes:
        session_file: Path to the analyzed capture file.
        total_commands: Number of sent commands with labels.
        timings: Per-command timing records.
        shoot_gaps_ms: Gaps between consecutive shoot commands.
        avg_shoot_gap_ms: Average gap between shoot commands.
        avg_latency_ms: Average command→response latency.
    """

    session_file: str
    total_commands: int
    timings: list[CommandTimingDict]
    shoot_gaps_ms: list[int]
    avg_shoot_gap_ms: int
    avg_latency_ms: int


def _load_messages(capture_path: Path) -> list[CapturedMessage]:
    """Load and decode messages from a capture session file.

    Args:
        capture_path: Path to a .capture_session.json file.

    Returns:
        Decoded message list.
    """
    text = capture_path.read_text(encoding="utf-8")
    parsed = load_json_str(text)
    if not isinstance(parsed, dict):
        raise ValueError("capture session must be a JSON object")
    raw: JSONObject = parsed
    raw_messages = raw["messages"]
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be a list")
    messages: list[CapturedMessage] = []
    for raw_msg in raw_messages:
        if not isinstance(raw_msg, dict):
            continue
        msg_obj: JSONObject = raw_msg
        messages.append(decode_captured_message(msg_obj))
    return messages


def _extract_timings(messages: list[CapturedMessage]) -> list[CommandTimingDict]:
    """Extract per-command timing records from decoded messages.

    Args:
        messages: Decoded capture messages.

    Returns:
        Timing records for each labeled sent command.
    """
    timings: list[CommandTimingDict] = []
    for i, msg in enumerate(messages):
        if msg["direction"] != "sent":
            continue
        label = msg.get("sent_label", "")
        if not label:
            continue
        response_ms = 0
        latency_ms = 0
        for j in range(i + 1, min(i + 50, len(messages))):
            resp = messages[j]
            if resp["direction"] == "received":
                response_ms = resp["timestamp_ms"]
                latency_ms = response_ms - msg["timestamp_ms"]
                break
        timings.append(
            CommandTimingDict(
                sent_ms=msg["timestamp_ms"],
                label=label,
                response_ms=response_ms,
                latency_ms=latency_ms,
            )
        )
    return timings


def analyze_timing(capture_path: Path) -> TimingReportDict:
    """Extract command-response timing from a capture session.

    Args:
        capture_path: Path to a .capture_session.json file.

    Returns:
        Timing report with per-command latencies and shoot gaps.
    """
    messages = _load_messages(capture_path)
    timings = _extract_timings(messages)

    shoot_timings = [t for t in timings if "shoot" in t["label"]]
    shoot_gaps: list[int] = []
    for i in range(1, len(shoot_timings)):
        shoot_gaps.append(shoot_timings[i]["sent_ms"] - shoot_timings[i - 1]["sent_ms"])

    avg_shoot_gap = sum(shoot_gaps) // len(shoot_gaps) if shoot_gaps else 0
    avg_latency = sum(t["latency_ms"] for t in timings) // len(timings) if timings else 0

    return TimingReportDict(
        session_file=str(capture_path),
        total_commands=len(timings),
        timings=timings,
        shoot_gaps_ms=shoot_gaps,
        avg_shoot_gap_ms=avg_shoot_gap,
        avg_latency_ms=avg_latency,
    )


def render_timing_report(report: TimingReportDict) -> str:
    """Render a timing report as a human-readable table.

    Args:
        report: Timing report to render.

    Returns:
        Multi-line string with timing summary.
    """
    lines: list[str] = [
        f"SESSION TIMING: {report['session_file']}",
        f"  commands: {report['total_commands']}",
        f"  avg latency: {report['avg_latency_ms']}ms",
        f"  avg shoot gap: {report['avg_shoot_gap_ms']}ms",
        "",
        "SHOOT GAPS (ms between consecutive shots):",
    ]
    for gap in report["shoot_gaps_ms"]:
        lines.append(f"  {gap}ms")
    lines.append("")
    lines.append("COMMAND LATENCIES (top 20 by latency):")

    def _latency_key(t: CommandTimingDict) -> int:
        return t["latency_ms"]

    sorted_by_latency = sorted(report["timings"], key=_latency_key, reverse=True)
    for t in sorted_by_latency[:20]:
        lines.append(f"  {t['latency_ms']:>6}ms  {t['label']}")
    return "\n".join(lines)


def main() -> int:
    """Entry point for session timing analysis.

    Returns:
        Exit code (0 for success).
    """
    _test_hooks.setup_rich_logging("INFO")
    full_argv = list(core_hooks.get_argv())
    user_args = full_argv[1:] if full_argv else []
    if not user_args:
        capture_path = Path("runs/bot/latest.capture_session.json")
    else:
        capture_path = Path(user_args[0])
    report = analyze_timing(capture_path)
    sys.stdout.write(render_timing_report(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
