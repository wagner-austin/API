"""Replay tooling for offline bot decision analysis.

Loads a captured WebSocket session and re-runs the planner tick-by-tick
against the decoded world state, recording structured decision traces
without launching a live browser.
"""

from __future__ import annotations

from tankpit_bot.replay.engine import replay_session
from tankpit_bot.replay.types import (
    ReplaySessionResultDict,
    ReplayTickTraceDict,
    decode_replay_session_result,
    decode_replay_tick_trace,
    encode_replay_session_result,
    encode_replay_tick_trace,
)

__all__ = [
    "ReplaySessionResultDict",
    "ReplayTickTraceDict",
    "decode_replay_session_result",
    "decode_replay_tick_trace",
    "encode_replay_session_result",
    "encode_replay_tick_trace",
    "replay_session",
]
