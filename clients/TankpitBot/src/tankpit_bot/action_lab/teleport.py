"""Live teleport probe harness."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.capture import save_capture_session
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportProbeSessionDict,
    TeleportStartupTimingDict,
    TeleportTargetDict,
    encode_teleport_probe_session,
)
from tankpit_bot.bot import Bot
from tankpit_bot.bot.states import make_no_action, transition_to
from tankpit_bot.browser import PlaywrightNotInstalledError, reset_cdp_time_offset
from tankpit_bot.sniffer import reset_all_trackers, reset_world_state
from tankpit_bot.sniffer.viewport import reset_viewport_tracking
from tankpit_bot.state import SelfStateDict

log = get_logger(__name__)
_TELEPORT_POLL_INTERVAL_MS = 100.0


class TeleportProbeError(Exception):
    """Raised when the teleport probe cannot proceed."""


def _clamp_tile(value: int) -> int:
    """Clamp a world coordinate to the valid tile range.

    Args:
        value: Raw tile coordinate.

    Returns:
        Clamped tile coordinate in the inclusive ``0..255`` range.
    """
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


def build_box_targets(
    origin_x: int,
    origin_y: int,
    step_x: int,
    step_y: int,
) -> list[TeleportTargetDict]:
    """Build the default 2x5 teleport target box around an origin.

    Args:
        origin_x: Spawn X coordinate.
        origin_y: Spawn Y coordinate.
        step_x: Horizontal spacing between columns.
        step_y: Vertical spacing between rows.

    Returns:
        Ten teleport targets in row-major order.

    Raises:
        ValueError: If either step is not positive.
    """
    if step_x <= 0:
        raise ValueError("step_x must be positive")
    if step_y <= 0:
        raise ValueError("step_y must be positive")
    x_offsets = (-2 * step_x, -step_x, 0, step_x, 2 * step_x)
    y_offsets = (-step_y, step_y)
    targets: list[TeleportTargetDict] = []
    for row_index, y_offset in enumerate(y_offsets):
        for col_index, x_offset in enumerate(x_offsets):
            targets.append(
                TeleportTargetDict(
                    label=f"box_r{row_index}_c{col_index}",
                    x=_clamp_tile(origin_x + x_offset),
                    y=_clamp_tile(origin_y + y_offset),
                )
            )
    return targets


def parse_targets_arg(raw: str) -> list[TeleportTargetDict]:
    """Parse a CLI target list into teleport targets.

    The accepted format is ``x:y[,x:y...]``.

    Args:
        raw: Raw CLI argument string.

    Returns:
        Parsed teleport targets with deterministic labels.

    Raises:
        ValueError: If the argument is empty, malformed, or outside tile bounds.
    """
    stripped = raw.strip()
    if not stripped:
        raise ValueError("targets argument must not be empty")
    targets: list[TeleportTargetDict] = []
    for index, part in enumerate(stripped.split(",")):
        piece = part.strip()
        coords = piece.split(":")
        if len(coords) != 2:
            raise ValueError(f"invalid target '{piece}'; expected x:y")
        x = int(coords[0])
        y = int(coords[1])
        if x < 0 or x > 255 or y < 0 or y > 255:
            raise ValueError(f"target '{piece}' is outside 0..255 tile bounds")
        targets.append(TeleportTargetDict(label=f"target_{index}", x=x, y=y))
    return targets


def _wait_for_teleport_outcome(
    page: action_session.WaitPageProtocol,
    provider: action_session.BufferedWorldStateProviderProtocol,
    target: TeleportTargetDict,
    *,
    map_open_started_ms: int,
    map_sync_timestamp_ms: int | None,
    teleport_started_ms: int,
    fuel_before: int,
    world_timestamp_before: int,
    timeout_ms: int,
) -> TeleportAttemptResultDict:
    """Wait for a teleport to land or time out.

    Args:
        page: Page-like object used for short waits.
        provider: World-state provider to poll.
        target: Requested teleport destination.
        map_open_started_ms: Timestamp when map-open was sent.
        map_sync_timestamp_ms: Timestamp when the map-open fresh sync arrived.
        teleport_started_ms: Timestamp when the teleport command was sent.
        fuel_before: Fuel before the attempt started.
        world_timestamp_before: World-state timestamp before the attempt started.
        timeout_ms: Maximum time to wait for landing.

    Returns:
        Terminal attempt result for the teleport.

    Raises:
        TeleportProbeError: If self state disappears during the wait.
    """
    while action_hooks.get_current_time_ms() - teleport_started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(provider)
        if action_hooks.check_and_clear_teleport_landed():
            completion_timestamp_ms = action_hooks.get_current_time_ms()
            world = provider.get_world_state()
            self_state = world["self_state"]
            if self_state is None:
                raise TeleportProbeError("self state disappeared after teleport landed")
            status: Literal["landed_exact", "landed_offset", "map_sync_timeout", "teleport_timeout"]
            if self_state["x"] == target["x"] and self_state["y"] == target["y"]:
                status = "landed_exact"
            else:
                status = "landed_offset"
            return TeleportAttemptResultDict(
                target=target,
                status=status,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                completion_timestamp_ms=completion_timestamp_ms,
                map_sync_elapsed_ms=(
                    None
                    if map_sync_timestamp_ms is None
                    else map_sync_timestamp_ms - map_open_started_ms
                ),
                teleport_elapsed_ms=completion_timestamp_ms - teleport_started_ms,
                fuel_before=fuel_before,
                fuel_after=self_state["fuel"],
                world_timestamp_before=world_timestamp_before,
                world_timestamp_after=world["timestamp_ms"],
                landed_signal_received=True,
                landed_x=self_state["x"],
                landed_y=self_state["y"],
                message_start_index=0,
                message_end_index=0,
            )
        page.wait_for_timeout(_TELEPORT_POLL_INTERVAL_MS)
    completion_timestamp_ms = action_hooks.get_current_time_ms()
    world = provider.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        raise TeleportProbeError("self state disappeared while waiting for teleport timeout")
    return TeleportAttemptResultDict(
        target=target,
        status="teleport_timeout",
        map_open_started_ms=map_open_started_ms,
        map_sync_timestamp_ms=map_sync_timestamp_ms,
        teleport_started_ms=teleport_started_ms,
        completion_timestamp_ms=completion_timestamp_ms,
        map_sync_elapsed_ms=(
            None if map_sync_timestamp_ms is None else map_sync_timestamp_ms - map_open_started_ms
        ),
        teleport_elapsed_ms=completion_timestamp_ms - teleport_started_ms,
        fuel_before=fuel_before,
        fuel_after=self_state["fuel"],
        world_timestamp_before=world_timestamp_before,
        world_timestamp_after=world["timestamp_ms"],
        landed_signal_received=False,
        landed_x=self_state["x"],
        landed_y=self_state["y"],
        message_start_index=0,
        message_end_index=0,
    )


def format_teleport_probe_summary(session: TeleportProbeSessionDict) -> str:
    """Format a compact human-readable summary line for a probe session.

    Args:
        session: Teleport probe session to summarize.

    Returns:
        One-line summary string with outcome counts.
    """
    exact = 0
    offset = 0
    map_timeouts = 0
    teleport_timeouts = 0
    for attempt in session["attempts"]:
        if attempt["status"] == "landed_exact":
            exact += 1
        elif attempt["status"] == "landed_offset":
            offset += 1
        elif attempt["status"] == "map_sync_timeout":
            map_timeouts += 1
        else:
            teleport_timeouts += 1
    startup_timing = session["startup_timing"]
    bootstrap_ms = (
        startup_timing["command_ready_timestamp_ms"] - startup_timing["initial_sync_started_ms"]
    )
    return (
        "Teleport probe complete: "
        f"strategy={session['teleport_strategy']} "
        f"attempts={len(session['attempts'])} "
        f"exact={exact} offset={offset} "
        f"map_sync_timeout={map_timeouts} teleport_timeout={teleport_timeouts} "
        "session_to_initial_sync_ms="
        f"{startup_timing['initial_sync_started_ms'] - session['start_timestamp_ms']} "
        f"initial_sync_to_command_ready_ms={bootstrap_ms}"
    )


def _limit_targets(
    targets: list[TeleportTargetDict],
    max_targets: int | None,
) -> list[TeleportTargetDict]:
    """Limit the target list when a maximum is configured.

    Args:
        targets: Candidate target list.
        max_targets: Maximum number of targets to keep, or None for no limit.

    Returns:
        Original targets, or the leading subset constrained by max_targets.

    Raises:
        ValueError: If max_targets is not positive when provided.
    """
    if max_targets is None:
        return targets
    if max_targets <= 0:
        raise ValueError("max_targets must be positive")
    return targets[:max_targets]


class TeleportProbe(Bot):
    """Live teleport probe that isolates map-open and teleport behavior."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the current self state or raise when absent.

        Returns:
            Current self tank state.

        Raises:
            TeleportProbeError: If self state is not yet available.
        """
        self_state = self.get_self_state()
        if self_state is None:
            raise TeleportProbeError("self state is unavailable")
        return self_state

    def _require_page(self) -> action_session.WaitPageProtocol:
        """Return the current Playwright page or raise when absent.

        Returns:
            Current page handle.

        Raises:
            TeleportProbeError: If the page has not been initialized.
        """
        if self._page is None:
            raise TeleportProbeError("page is unavailable")
        return self._page

    def _clear_in_flight_action(self) -> None:
        """Clear any pending action record without inferring success or failure."""
        self._state_data = transition_to(
            self._state_data,
            self._state_data["state"],
            in_flight_action=make_no_action(),
        )

    def _reset_probe_state_to_idle(self) -> None:
        """Reset the probe to an executable idle state between attempts."""
        self._state_data = transition_to(
            self._state_data,
            "IDLE",
            in_flight_action=make_no_action(),
        )

    def _probe_single_target(
        self,
        target: TeleportTargetDict,
        *,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> TeleportAttemptResultDict:
        """Run one teleport attempt against the live server.

        Args:
            target: Requested destination.
            map_sync_timeout_ms: Maximum wait for the map-open fresh sync.
            teleport_timeout_ms: Maximum wait for teleport confirmation.
            settle_delay_ms: Delay after completion before the next attempt.

        Returns:
            Terminal attempt result for the target.

        Raises:
            TeleportProbeError: If command dispatch fails.
        """
        page = self._require_page()
        world_before = self.get_world_state()
        self_state_before = self._require_self_state()
        fuel_before = self_state_before["fuel"]
        world_timestamp_before = world_before["timestamp_ms"]

        self._reset_probe_state_to_idle()
        message_start_index = len(self.messages)
        map_open_started_ms = action_hooks.get_current_time_ms()
        if not self.open_map():
            raise TeleportProbeError("map_open command dispatch failed")
        map_sync_timestamp_ms: int | None = None
        if teleport_strategy == "sync_before_teleport":
            map_sync_timestamp_ms = action_session.wait_for_world_sync(
                page,
                self,
                map_open_started_ms,
                map_sync_timeout_ms,
            )
            if map_sync_timestamp_ms is None:
                completion_timestamp_ms = action_hooks.get_current_time_ms()
                self._reset_probe_state_to_idle()
                self_state_after = self._require_self_state()
                result = TeleportAttemptResultDict(
                    target=target,
                    status="map_sync_timeout",
                    map_open_started_ms=map_open_started_ms,
                    map_sync_timestamp_ms=None,
                    teleport_started_ms=None,
                    completion_timestamp_ms=completion_timestamp_ms,
                    map_sync_elapsed_ms=None,
                    teleport_elapsed_ms=None,
                    fuel_before=fuel_before,
                    fuel_after=self_state_after["fuel"],
                    world_timestamp_before=world_timestamp_before,
                    world_timestamp_after=self.get_world_state()["timestamp_ms"],
                    landed_signal_received=False,
                    landed_x=self_state_after["x"],
                    landed_y=self_state_after["y"],
                    message_start_index=message_start_index,
                    message_end_index=len(self.messages),
                )
                if settle_delay_ms > 0:
                    page.wait_for_timeout(float(settle_delay_ms))
                return result

        teleport_started_ms = action_hooks.get_current_time_ms()
        if not self.teleport_to(target["x"], target["y"]):
            raise TeleportProbeError("teleport command dispatch failed")
        result = _wait_for_teleport_outcome(
            page,
            self,
            target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            fuel_before=fuel_before,
            world_timestamp_before=world_timestamp_before,
            timeout_ms=teleport_timeout_ms,
        )
        result["message_start_index"] = message_start_index
        result["message_end_index"] = len(self.messages)
        self._reset_probe_state_to_idle()
        if settle_delay_ms > 0:
            page.wait_for_timeout(float(settle_delay_ms))
        return result

    def execute(
        self,
        *,
        explicit_targets: list[TeleportTargetDict] | None,
        box_step_x: int,
        box_step_y: int,
        max_targets: int | None,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> TeleportProbeSessionDict:
        """Run the live teleport probe session.

        Args:
            explicit_targets: Absolute requested targets, or None to use the default box.
            box_step_x: Horizontal spacing for the default box.
            box_step_y: Vertical spacing for the default box.
            max_targets: Maximum number of targets to run, or None for all.
            teleport_strategy: Teleport sequencing strategy for each attempt.
            initial_sync_timeout_ms: Maximum wait for the initial self-state sync.
            map_sync_timeout_ms: Maximum wait for the map-open fresh sync.
            teleport_timeout_ms: Maximum wait for teleport confirmation.
            settle_delay_ms: Delay after each attempt.

        Returns:
            Complete teleport probe session.

        Raises:
            PlaywrightNotInstalledError: If Playwright is not installed.
            TeleportProbeError: If bootstrap or command dispatch fails.
        """
        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError("Playwright is not installed.")

        self._start_timestamp_ms = action_hooks.get_current_time_ms()
        self._messages = []
        self._ws_urls = {}
        self._magic = None
        self._cdp_message_buffer = []

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()
            cdp = context.new_cdp_session(page)

            self._cdp = cdp
            self._page = page

            reset_world_state()
            reset_all_trackers()
            reset_cdp_time_offset()
            reset_viewport_tracking()

            self._setup_console_listener(cdp)
            self._setup_cdp_handlers(cdp)
            self._navigate_and_login(page, cdp, tank_name_prefix="TP", auto_join_room=True)
            self._wait_for_game_ready(page)
            game_ready_timestamp_ms = action_hooks.get_current_time_ms()
            self._gather_intel(page, cdp)
            intel_ready_timestamp_ms = action_hooks.get_current_time_ms()

            try:
                initial_sync_started_ms = action_hooks.get_current_time_ms()
                initial_world_timestamp_ms, spawn = action_session.wait_for_initial_self_state(
                    page,
                    self,
                    initial_sync_started_ms,
                    initial_sync_timeout_ms,
                )
                action_session.advance_startup_state(self)
                command_ready_timestamp_ms = action_hooks.get_current_time_ms()
                targets = _limit_targets(
                    (
                        explicit_targets
                        if explicit_targets is not None
                        else build_box_targets(spawn["x"], spawn["y"], box_step_x, box_step_y)
                    ),
                    max_targets,
                )
                if not targets:
                    raise TeleportProbeError("teleport probe requires at least one target")
                attempts: list[TeleportAttemptResultDict] = []
                for target in targets:
                    attempts.append(
                        self._probe_single_target(
                            target,
                            teleport_strategy=teleport_strategy,
                            map_sync_timeout_ms=map_sync_timeout_ms,
                            teleport_timeout_ms=teleport_timeout_ms,
                            settle_delay_ms=settle_delay_ms,
                        )
                    )
                first_attempt_started_ms = attempts[0]["map_open_started_ms"] if attempts else None
                startup_timing = TeleportStartupTimingDict(
                    game_ready_timestamp_ms=game_ready_timestamp_ms,
                    intel_ready_timestamp_ms=intel_ready_timestamp_ms,
                    initial_sync_started_ms=initial_sync_started_ms,
                    initial_world_timestamp_ms=initial_world_timestamp_ms,
                    command_ready_timestamp_ms=command_ready_timestamp_ms,
                    first_attempt_started_ms=first_attempt_started_ms,
                    game_ready_to_intel_ready_ms=intel_ready_timestamp_ms - game_ready_timestamp_ms,
                    intel_ready_to_initial_world_ms=(
                        initial_world_timestamp_ms - intel_ready_timestamp_ms
                    ),
                    initial_world_to_command_ready_ms=(
                        command_ready_timestamp_ms - initial_world_timestamp_ms
                    ),
                    command_ready_to_first_attempt_ms=(
                        None
                        if first_attempt_started_ms is None
                        else first_attempt_started_ms - command_ready_timestamp_ms
                    ),
                )
                return TeleportProbeSessionDict(
                    session_id=self.session_id,
                    start_timestamp_ms=self._start_timestamp_ms,
                    end_timestamp_ms=action_hooks.get_current_time_ms(),
                    base_url=self._target_url,
                    spawn_x=spawn["x"],
                    spawn_y=spawn["y"],
                    teleport_strategy=teleport_strategy,
                    max_targets=max_targets,
                    capture_session_path="",
                    initial_sync_timeout_ms=initial_sync_timeout_ms,
                    startup_timing=startup_timing,
                    map_sync_timeout_ms=map_sync_timeout_ms,
                    teleport_timeout_ms=teleport_timeout_ms,
                    settle_delay_ms=settle_delay_ms,
                    targets=targets,
                    attempts=attempts,
                )
            finally:
                self._cdp = None
                self._page = None
                self._cleanup(cdp, page, context, browser)


def run_teleport_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    explicit_targets: list[TeleportTargetDict] | None = None,
    box_step_x: int = 8,
    box_step_y: int = 8,
    max_targets: int | None = None,
    teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"] = (
        "sync_before_teleport"
    ),
    initial_sync_timeout_ms: int = 10000,
    map_sync_timeout_ms: int = 3000,
    teleport_timeout_ms: int = 10000,
    settle_delay_ms: int = 500,
) -> TeleportProbeSessionDict:
    """Run a live teleport probe and save the session JSON.

    Args:
        target_url: URL to navigate to.
        output_path: Output path for the session JSON.
        headless: Whether to run the browser headlessly.
        prefer_account: Whether to use account login instead of guest login.
        explicit_targets: Absolute targets to test, or None for the default box.
        box_step_x: Horizontal spacing for the default box.
        box_step_y: Vertical spacing for the default box.
        max_targets: Maximum number of targets to run, or None for all.
        teleport_strategy: Teleport sequencing strategy for each attempt.
        initial_sync_timeout_ms: Maximum wait for the initial self-state sync.
        map_sync_timeout_ms: Maximum wait for the map-open fresh sync.
        teleport_timeout_ms: Maximum wait for teleport confirmation.
        settle_delay_ms: Delay after each attempt.

    Returns:
        Completed teleport probe session.
    """
    probe = TeleportProbe(
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    session = probe.execute(
        explicit_targets=explicit_targets,
        box_step_x=box_step_x,
        box_step_y=box_step_y,
        max_targets=max_targets,
        teleport_strategy=teleport_strategy,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        settle_delay_ms=settle_delay_ms,
    )
    capture_session_path = save_capture_session(
        session_id=session["session_id"],
        start_timestamp_ms=session["start_timestamp_ms"],
        end_timestamp_ms=session["end_timestamp_ms"],
        base_url=session["base_url"],
        messages=probe.messages,
        magic=probe.magic,
        output_path=output_path,
    )
    session["capture_session_path"] = capture_session_path
    encoded = encode_teleport_probe_session(session)
    json_str = dump_json_str(encoded, compact=False, indent=2)
    _test_hooks.write_text(Path(output_path), json_str)
    log.info(format_teleport_probe_summary(session))
    return session


__all__ = [
    "TeleportProbe",
    "TeleportProbeError",
    "build_box_targets",
    "format_teleport_probe_summary",
    "parse_targets_arg",
    "run_teleport_probe",
]
