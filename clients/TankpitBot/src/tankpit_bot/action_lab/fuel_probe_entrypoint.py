"""Shared entrypoint helpers for the live fuel probe."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeSessionDict,
    encode_fuel_probe_session,
)
from tankpit_bot.action_lab.probe_entrypoint import (
    run_and_save_standard_probe_session,
)
from tankpit_bot.types import CapturedMessage


class FuelProbeEntrypointProtocol(Protocol):
    """Minimal probe interface required to run and persist a fuel session."""

    def execute_probe(
        self,
        *,
        target_pickups: int,
        max_attempts: int,
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelProbeSessionDict:
        """Execute one live fuel-probe session."""

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return captured wire messages."""

    @property
    def magic(self) -> str | None:
        """Return the active capture magic, when available."""


class FuelProbeFactoryProtocol(Protocol):
    """Typed callable used to build one live fuel probe."""

    def __call__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
    ) -> FuelProbeEntrypointProtocol:
        """Create one probe instance."""


def run_and_save_fuel_probe_session(
    *,
    probe_factory: FuelProbeFactoryProtocol,
    summary_formatter: Callable[[FuelProbeSessionDict], str],
    target_url: str,
    output_path: str,
    headless: bool,
    prefer_account: bool,
    target_pickups: int,
    max_attempts: int,
    initial_sync_timeout_ms: int,
    map_sync_timeout_ms: int,
    teleport_timeout_ms: int,
    radar_timeout_ms: int,
    pickup_timeout_ms: int,
    settle_delay_ms: int,
) -> FuelProbeSessionDict:
    """Run one live fuel probe and persist its artifacts.

    Args:
        probe_factory: Typed probe constructor.
        summary_formatter: Session summary formatter for terminal logging.
        target_url: Browser target URL.
        output_path: JSON output path for the structured session artifact.
        headless: Whether the browser should run headless.
        prefer_account: Whether to prefer pre-selected account login.
        target_pickups: Number of successful pickups required before exit.
        max_attempts: Maximum number of probe attempts.
        initial_sync_timeout_ms: Initial world-sync timeout in milliseconds.
        map_sync_timeout_ms: Map-sync timeout in milliseconds.
        teleport_timeout_ms: Teleport timeout in milliseconds.
        radar_timeout_ms: Radar timeout in milliseconds.
        pickup_timeout_ms: Pickup timeout in milliseconds.
        settle_delay_ms: Optional post-attempt settle delay in milliseconds.

    Returns:
        Completed and persisted session payload.
    """

    def _run_session(probe: FuelProbeEntrypointProtocol) -> FuelProbeSessionDict:
        return probe.execute_probe(
            target_pickups=target_pickups,
            max_attempts=max_attempts,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            radar_timeout_ms=radar_timeout_ms,
            pickup_timeout_ms=pickup_timeout_ms,
            settle_delay_ms=settle_delay_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=probe_factory,
        run_session=_run_session,
        encoder=encode_fuel_probe_session,
        summary_formatter=summary_formatter,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = ["run_and_save_fuel_probe_session"]
