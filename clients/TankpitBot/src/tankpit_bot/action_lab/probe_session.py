"""Shared session-envelope helpers for live action-lab probes."""

from __future__ import annotations

from typing import NamedTuple, Protocol

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    build_probe_startup_timing,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict


class ProbeSessionEnvelopeSourceProtocol(Protocol):
    """Probe state required to build one session envelope."""

    _start_timestamp_ms: int
    _target_url: str

    @property
    def session_id(self) -> str:
        """Return the stable session identifier."""


class ProbeSessionEnvelope(NamedTuple):
    """Common metadata shared by live action-lab session payloads."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    startup_timing: TeleportStartupTimingDict


def build_probe_session_envelope(
    probe: ProbeSessionEnvelopeSourceProtocol,
    *,
    context: ProbeCommandReadyContextDict,
    first_attempt_started_ms: int | None,
) -> ProbeSessionEnvelope:
    """Build shared session metadata for one command-ready probe run.

    Args:
        probe: Probe instance that owns the session.
        context: Command-ready runtime context.
        first_attempt_started_ms: Timestamp of the first attempt, if any.

    Returns:
        Shared session envelope with spawn coordinates and startup timing.
    """
    spawn = context["spawn"]
    return ProbeSessionEnvelope(
        session_id=probe.session_id,
        start_timestamp_ms=probe._start_timestamp_ms,
        end_timestamp_ms=action_hooks.get_current_time_ms(),
        base_url=probe._target_url,
        spawn_x=spawn["x"],
        spawn_y=spawn["y"],
        startup_timing=build_probe_startup_timing(
            game_ready_timestamp_ms=context["game_ready_timestamp_ms"],
            intel_ready_timestamp_ms=context["intel_ready_timestamp_ms"],
            initial_sync_started_ms=context["initial_sync_started_ms"],
            initial_world_timestamp_ms=context["initial_world_timestamp_ms"],
            command_ready_timestamp_ms=context["command_ready_timestamp_ms"],
            first_attempt_started_ms=first_attempt_started_ms,
        ),
    )


__all__ = [
    "ProbeSessionEnvelope",
    "ProbeSessionEnvelopeSourceProtocol",
    "build_probe_session_envelope",
]
