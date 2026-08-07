"""Typed module handles for the shared teleport-attempt DI seams.

The probe modules import their collaborators by name, so a test swaps
the attribute ON THE CONSUMER module to intercept the call — that
module attribute IS the seam. Reading it from outside would otherwise
trip mypy's no-implicit-reexport, and the fix used to be an alias
re-exported through ``__all__``. The alias was a rename with no other
purpose; typing the surface here removes it while keeping every stub
checked against the real signature, so a signature change fails
type-checking instead of leaving a test driving a shape the production
code no longer has ([[testing-patterns]]).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from tests.action_lab._combat_probe_harness import (
    AcquisitionPhaseFn,
    TeleportCommandFn,
)

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    TrackedTeleportAttempt,
)
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import TeleportTargetDict


class TrackedTeleportAttemptFn(Protocol):
    """The shared acquisition-plus-teleport attempt, spelled out for stubs.

    Written in full rather than as ``*args: object`` so every stub is
    checked against the real signature — a loose Protocol would accept
    a stub the production caller could never invoke.
    """

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: TeleportAttemptProbeProtocol,
        target: TeleportTargetDict,
        *,
        cdp: CDPSessionProtocol | None,
        attempt_label: str,
        fuel_before: int,
        world_timestamp_before: int,
        send_acquisition_command: Callable[[], bool],
        acquisition_command_name: str,
        capture_before_map_open: bool,
        wait_for_acquisition_sync: bool,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
        unexpected_result_error: type[Exception],
        unexpected_result_message: str,
        reset_to_idle_before_start: bool = True,
    ) -> TrackedTeleportAttempt:
        """Run one tracked teleport attempt."""


class _TeleportAttemptModuleProtocol(Protocol):
    """The two phase functions ``teleport_attempt`` calls through."""

    run_tracked_acquisition_phase: AcquisitionPhaseFn
    run_tracked_teleport_command: TeleportCommandFn


class _RepositionModuleProtocol(Protocol):
    """The attempt runner a target-phase module calls through."""

    run_tracked_teleport_attempt: TrackedTeleportAttemptFn


_teleport_attempt_import = __import__(
    "tankpit_bot.action_lab.teleport_attempt",
    fromlist=["teleport_attempt"],
)
_equipment_target_phase_import = __import__(
    "tankpit_bot.action_lab.equipment_target_phase",
    fromlist=["equipment_target_phase"],
)
_fuel_target_phase_import = __import__(
    "tankpit_bot.action_lab.fuel_target_phase",
    fromlist=["fuel_target_phase"],
)


teleport_attempt_module: _TeleportAttemptModuleProtocol = _teleport_attempt_import
equipment_target_phase_module: _RepositionModuleProtocol = _equipment_target_phase_import
fuel_target_phase_module: _RepositionModuleProtocol = _fuel_target_phase_import
