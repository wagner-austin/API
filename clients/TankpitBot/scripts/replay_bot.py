"""Replay a captured WebSocket session through the bot planner offline.

Usage: poetry run python -m scripts.replay_bot [session.json] [--json]

Loads a capture session, feeds received frames through the decode/world-state
pipeline, runs the planner tick-by-tick, and prints the decision trace. With
``--json``, outputs the full result as a JSON document to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.replay.engine import replay_session
from tankpit_bot.replay.types import ReplayTickTraceDict, encode_replay_session_result
from tankpit_bot.sniffer.decoders import set_protocol_frame_logging
from tankpit_bot.types import decode_capture_session

log = get_logger(__name__)


def _format_threat_summary(trace: ReplayTickTraceDict) -> str:
    """Format the visible threat list as a compact summary string.

    Args:
        trace: Tick trace containing visible_threats.

    Returns:
        Compact threat summary like ``2:[Tank1@d=5,Tank2@d=12]`` or ``0:[]``.
    """
    threats = trace["visible_threats"]
    if not threats:
        return "0:[]"
    entries = [f"{t['name']}@d={t['distance']}" for t in threats[:3]]
    suffix = f"+{len(threats) - 3}" if len(threats) > 3 else ""
    return f"{len(threats)}:[{','.join(entries)}{suffix}]"


def _format_trace_line(trace: ReplayTickTraceDict) -> str:
    """Format a single tick trace as a human-readable log line.

    Args:
        trace: Tick trace to format.

    Returns:
        Formatted string for console output.
    """
    ai_mode = trace["ai_mode"]
    ai_mode_state = trace["ai_mode_state"]
    ai_info = f" ai={ai_mode}/{ai_mode_state}" if ai_mode_state else f" ai={ai_mode}"
    resource = trace["resource_target_kind"]
    resource_info = f" resource={resource}" if resource else ""
    return (
        f"[{trace['tick_index']:4d}] "
        f"pos=({trace['self_x']},{trace['self_y']}) "
        f"fuel={trace['fuel']:5d} "
        f"{trace['behavior_mode']:<20s} "
        f"score={trace['behavior_score']:4d} "
        f"cmd={trace['command_type']:<16s} "
        f"target=({trace['target_x']},{trace['target_y']}) "
        f"threats={_format_threat_summary(trace)} "
        f"containers={trace['container_count']}"
        f"{ai_info}{resource_info} "
        f"reason={trace['behavior_reason']}"
    )


def main() -> int:
    """Run the replay and print decision traces.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    setup_rich_logging(level="WARNING")
    set_protocol_frame_logging(False)

    argv = _test_hooks.get_argv()
    args = argv[1:]

    json_output = "--json" in args
    path_args = [a for a in args if a != "--json"]
    session_path = Path(path_args[0]) if path_args else Path("capture_session.json")

    if not _test_hooks.path_exists(session_path):
        log.error("File not found: %s", session_path)
        return 1

    session_text = _test_hooks.read_text(session_path)
    session_json = narrow_json_to_dict(load_json_str(session_text))
    session = decode_capture_session(session_json)

    if session["magic"] is None:
        log.error("No magic key in session — cannot XOR-decode binary messages")
        return 1

    result = replay_session(session)

    if json_output:
        encoded = encode_replay_session_result(result)
        sys.stdout.write(dump_json_str(encoded) + "\n")
        return 0

    log.warning(
        "Replay: session=%s messages=%d ticks=%d",
        result["session_id"],
        result["total_messages"],
        result["total_ticks"],
    )
    for trace in result["traces"]:
        log.warning(_format_trace_line(trace))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
]
