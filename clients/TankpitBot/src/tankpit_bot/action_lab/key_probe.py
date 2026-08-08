"""Live keyboard probe: press physical keys, capture what the client sends.

The key→command mapping is client-side, so no wire capture can reveal
which KEY produced a frame — only pressing the key can. This probe
answers the 2026-07-24 three-way R-key discrepancy (site help panel:
"R: Radar"; this build's JS default keymap: KeyS→radar, KeyR→Top-10
red; user experience: R is radar) and verifies the whole default map
empirically: for each key it records the captured-message window
around one ``page.keyboard.press``, so offline analysis attributes
every sent frame (``sent_origin == "page_client"``) to its key.

Run as a GUEST so a radar press can never consume account extras.
Map keys ('f', 'm') are pressed LAST — the map-open contract
([[client-commands]]) means an open map changes later keys' behavior.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONObject, JSONValue
from platform_core.logging import get_logger

from tankpit_bot.action_lab.probe_base import ProbeBase, ProbeError
from tankpit_bot.action_lab.probe_entrypoint import run_and_save_standard_probe_session
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.action_lab.types_codecs import encode_teleport_startup_timing

log = get_logger(__name__)

DEFAULT_KEYS: tuple[str, ...] = (
    "t",
    "r",
    "p",
    "b",
    "o",
    "s",
    "e",
    "i",
    "c",
    "x",
    "/",
    "l",
    "n",
    "h",
    "z",
    "a",
    "f",
    "m",
)
"""Safe press order: Top-10 family, queries, toggles — map keys last.

Excluded on purpose: 'q' (exit), Space (fires), 'd' (mine placement),
digits (equipment toggles mutate persistent account state), arrows
(movement)."""


class KeyPressWindowDict(TypedDict):
    """Captured-message window around one key press."""

    key: str
    pressed_at_ms: int
    message_start_index: int
    message_end_index: int


class KeyProbeSessionDict(TypedDict):
    """One live keyboard-probe session."""

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    inter_key_delay_ms: int
    presses: list[KeyPressWindowDict]


def encode_key_probe_session(session: KeyProbeSessionDict) -> JSONObject:
    """Encode a keyboard-probe session to a JSON object.

    Args:
        session: Session to encode.

    Returns:
        JSON-serializable object representation.
    """
    presses: list[JSONValue] = [
        {
            "key": press["key"],
            "pressed_at_ms": press["pressed_at_ms"],
            "message_start_index": press["message_start_index"],
            "message_end_index": press["message_end_index"],
        }
        for press in session["presses"]
    ]
    return {
        "session_id": session["session_id"],
        "start_timestamp_ms": session["start_timestamp_ms"],
        "end_timestamp_ms": session["end_timestamp_ms"],
        "base_url": session["base_url"],
        "spawn_x": session["spawn_x"],
        "spawn_y": session["spawn_y"],
        "capture_session_path": session["capture_session_path"],
        "initial_sync_timeout_ms": session["initial_sync_timeout_ms"],
        "startup_timing": encode_teleport_startup_timing(session["startup_timing"]),
        "inter_key_delay_ms": session["inter_key_delay_ms"],
        "presses": presses,
    }


def format_key_probe_summary(session: KeyProbeSessionDict) -> str:
    """Format a compact human-readable summary for a keyboard-probe session.

    Args:
        session: Session to summarize.

    Returns:
        One-line summary string.
    """
    keys = ",".join(press["key"] for press in session["presses"])
    return (
        f"Key probe complete: presses={len(session['presses'])} "
        f"delay_ms={session['inter_key_delay_ms']} keys={keys}"
    )


class KeyProbe(ProbeBase):
    """Live keyboard probe pressing physical keys on the game page."""

    def _press_keys(
        self,
        keys: tuple[str, ...],
        inter_key_delay_ms: int,
    ) -> list[KeyPressWindowDict]:
        """Press each key once, recording its captured-message window.

        Args:
            keys: Keys to press, in order.
            inter_key_delay_ms: Wait after each press.

        Returns:
            One window record per pressed key.

        Raises:
            ProbeError: If the page is unavailable.
        """
        page = self._page
        if page is None:
            raise ProbeError("page is unavailable")
        from tankpit_bot.action_lab import _test_hooks as action_hooks

        presses: list[KeyPressWindowDict] = []
        for key in keys:
            start_index = len(self.messages)
            pressed_at_ms = action_hooks.get_current_time_ms()
            page.keyboard.press(key)
            page.wait_for_timeout(float(inter_key_delay_ms))
            action_hooks.drain_buffered_messages(self, self.world)
            presses.append(
                KeyPressWindowDict(
                    key=key,
                    pressed_at_ms=pressed_at_ms,
                    message_start_index=start_index,
                    message_end_index=len(self.messages),
                )
            )
            log.info(
                "Key press %r: %d captured frames",
                key,
                presses[-1]["message_end_index"] - start_index,
            )
        return presses

    def execute_probe(
        self,
        *,
        keys: tuple[str, ...],
        initial_sync_timeout_ms: int,
        inter_key_delay_ms: int,
    ) -> KeyProbeSessionDict:
        """Run the live keyboard-probe session."""
        if not keys:
            raise ProbeError("keys must not be empty")

        def _run_ready_session(context: ProbeCommandReadyContextDict) -> KeyProbeSessionDict:
            presses = self._press_keys(keys, inter_key_delay_ms)
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=presses[0]["pressed_at_ms"],
            )
            return KeyProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=envelope.startup_timing,
                inter_key_delay_ms=inter_key_delay_ms,
                presses=presses,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def _create_key_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> KeyProbe:
    """Factory for KeyProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        KeyProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, KeyProbe)
    return probe


def run_key_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    keys: tuple[str, ...] = DEFAULT_KEYS,
    initial_sync_timeout_ms: int = 10000,
    inter_key_delay_ms: int = 1500,
) -> KeyProbeSessionDict:
    """Run a live keyboard probe and save the session JSON."""

    def _run_session(probe: KeyProbe) -> KeyProbeSessionDict:
        return probe.execute_probe(
            keys=keys,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            inter_key_delay_ms=inter_key_delay_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_key_probe,
        run_session=_run_session,
        encoder=encode_key_probe_session,
        summary_formatter=format_key_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "DEFAULT_KEYS",
    "KeyProbe",
    "KeyProbeSessionDict",
    "encode_key_probe_session",
    "format_key_probe_summary",
    "run_key_probe",
]
