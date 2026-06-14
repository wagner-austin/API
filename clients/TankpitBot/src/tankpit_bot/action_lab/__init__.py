"""Live action probes for protocol-level diagnostics.

This package isolates transport and server-acceptance behavior from planner
behavior. The initial implementation focuses on teleport probing, but the
package structure is intended to host additional live action probes over time.
"""

from tankpit_bot.action_lab.capture import build_capture_output_path, save_capture_session
from tankpit_bot.action_lab.enemy_teleport import (
    EnemyTeleportProbe,
    format_enemy_teleport_probe_summary,
    run_enemy_teleport_probe,
)
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportAttemptResultDict,
    EnemyTeleportProbeSessionDict,
    decode_enemy_teleport_attempt_result,
    decode_enemy_teleport_probe_session,
    encode_enemy_teleport_attempt_result,
    encode_enemy_teleport_probe_session,
)
from tankpit_bot.action_lab.fuel_dot_probe import (
    format_fuel_dot_probe_summary,
    run_fuel_dot_probe,
)
from tankpit_bot.action_lab.fuel_dot_probe_types import (
    FuelDotProbeSessionDict,
)
from tankpit_bot.action_lab.fuel_locations import build_distinct_ground_targets
from tankpit_bot.action_lab.fuel_probe import (
    FuelProbe,
    FuelProbeError,
    format_fuel_probe_summary,
    run_fuel_probe,
)
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
    decode_fuel_probe_attempt_result,
    decode_fuel_probe_session,
    encode_fuel_probe_attempt_result,
    encode_fuel_probe_session,
)
from tankpit_bot.action_lab.movement_probe import (
    MovementProbe,
    MovementProbeError,
    format_movement_probe_summary,
    run_movement_probe,
)
from tankpit_bot.action_lab.movement_probe_types import (
    MovementProbeAttemptResultDict,
    MovementProbeSessionDict,
    decode_movement_probe_attempt_result,
    decode_movement_probe_session,
    encode_movement_probe_attempt_result,
    encode_movement_probe_session,
)
from tankpit_bot.action_lab.session import (
    ActionLabSessionError,
    advance_startup_state,
    wait_for_initial_self_state,
    wait_for_radar_sync,
    wait_for_world_sync,
)
from tankpit_bot.action_lab.teleport import (
    DEFAULT_TELEPORT_STRATEGY,
    TeleportProbe,
    TeleportProbeError,
    build_box_targets,
    format_teleport_probe_summary,
    parse_targets_arg,
    run_teleport_probe,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportProbeSessionDict,
    TeleportTargetDict,
    decode_teleport_attempt_result,
    decode_teleport_probe_session,
    decode_teleport_target,
    encode_teleport_attempt_result,
    encode_teleport_probe_session,
    encode_teleport_target,
)

__all__ = [
    "DEFAULT_TELEPORT_STRATEGY",
    "ActionLabSessionError",
    "EnemyTeleportAttemptResultDict",
    "EnemyTeleportProbe",
    "EnemyTeleportProbeSessionDict",
    "FuelDotProbeSessionDict",
    "FuelProbe",
    "FuelProbeAttemptResultDict",
    "FuelProbeError",
    "FuelProbeSessionDict",
    "MovementProbe",
    "MovementProbeAttemptResultDict",
    "MovementProbeError",
    "MovementProbeSessionDict",
    "TeleportAttemptResultDict",
    "TeleportProbe",
    "TeleportProbeError",
    "TeleportProbeSessionDict",
    "TeleportTargetDict",
    "advance_startup_state",
    "build_box_targets",
    "build_capture_output_path",
    "build_distinct_ground_targets",
    "decode_enemy_teleport_attempt_result",
    "decode_enemy_teleport_probe_session",
    "decode_fuel_probe_attempt_result",
    "decode_fuel_probe_session",
    "decode_movement_probe_attempt_result",
    "decode_movement_probe_session",
    "decode_teleport_attempt_result",
    "decode_teleport_probe_session",
    "decode_teleport_target",
    "encode_enemy_teleport_attempt_result",
    "encode_enemy_teleport_probe_session",
    "encode_fuel_probe_attempt_result",
    "encode_fuel_probe_session",
    "encode_movement_probe_attempt_result",
    "encode_movement_probe_session",
    "encode_teleport_attempt_result",
    "encode_teleport_probe_session",
    "encode_teleport_target",
    "format_enemy_teleport_probe_summary",
    "format_fuel_dot_probe_summary",
    "format_fuel_probe_summary",
    "format_movement_probe_summary",
    "format_teleport_probe_summary",
    "parse_targets_arg",
    "run_enemy_teleport_probe",
    "run_fuel_dot_probe",
    "run_fuel_probe",
    "run_movement_probe",
    "run_teleport_probe",
    "save_capture_session",
    "wait_for_initial_self_state",
    "wait_for_radar_sync",
    "wait_for_world_sync",
]
