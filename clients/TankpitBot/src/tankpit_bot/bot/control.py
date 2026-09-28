"""Control verbs for a RUNNING bot: the operator steering a live tank.

The need (operator, 95e5da95 item 7): peel one bot off a long human
fight without taking it out of the world. Stopping is not that --
``STOP`` ends the session -- so a running bot needs verbs short of it.

ONE CHANNEL, THE STOP CHANNEL'S. A fleet bot is an independent child
process; the manager cannot reach into its memory, which is exactly why
stop is a file. The verbs ride the same shape: one ``CONTROL`` file
beside ``STOP`` in ``runs/bot/<instance>/``, written atomically by the
fleet manager (:func:`write_control`), read at the top of every tick
and consumed on ingest (:func:`take_pending_control`). The in-process
``ModeBridge`` steers only the single-session service and is not a
second fleet channel.

EVERY VERB IS A STATE THE AI ALREADY HONOURS. Nothing here adds a hunt
gate or bypasses one; each verb writes a field an existing path reads:

* ``hold <HUNT|COLLECT>`` pins ``AIStateDict.manual_mode``, which the
  mode controller already respects.
* ``disengage`` drops the combat lock (:func:`clear_combat_target`) and
  pins COLLECT until ``release``, so the bot stays in the world and
  forages. The wartime floor composes with it unchanged.
* ``wind_down`` sets ``AIStateDict.wind_down``, the flag the session
  clock and the kill target already set: no new engagements, top off,
  exit ``session_complete``. It also clears any pin, because a pinned
  owner skips the arbitration that holds the stocked exit; a live
  locked fight still finishes first, as it does for the clock, so
  ``disengage`` then ``wind_down`` leaves one now.
* ``release`` clears the pin: auto-arbitration again.
* ``doctrine <name>`` rewrites ``AIConfigDict.doctrine`` in place,
  because a bot reads its environment config exactly once, at
  ``Bot.__init__``.

An unknown verb, a missing argument, or an argument on a verb that takes
none is a hard error: the file came from the manager, which validated
it with this same decoder, so anything else is corruption.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_str,
)
from platform_core.logging import get_logger
from platform_core.members import find_member, require_member
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.bot.ai.combat_target import clear_combat_target
from tankpit_bot.bot.ai.types import AIConfigDict, AIStateDict
from tankpit_bot.fleetshare.types import EngagementDoctrine
from tankpit_bot.types.modes import AIMode

log = get_logger(__name__)

#: The control file's name, beside the ``STOP`` sentinel in a run directory.
CONTROL_FILE_NAME = "CONTROL"


class ControlVerb(StrEnum):
    """What the operator can ask of a running bot."""

    HOLD = "hold"
    DISENGAGE = "disengage"
    WIND_DOWN = "wind_down"
    RELEASE = "release"
    DOCTRINE = "doctrine"


#: The modes ``hold`` may pin. ``UNSET`` is not a mode to hold; ``release``
#: is how the pin goes away.
HOLDABLE_MODES: tuple[AIMode, ...] = (AIMode.HUNT, AIMode.COLLECT)


class ControlCommandDict(TypedDict):
    """One control verb and its argument.

    Attributes:
        verb: What to do.
        argument: The mode for ``hold``, the doctrine word for
            ``doctrine``, and the empty string for every other verb.
    """

    verb: ControlVerb
    argument: str


def _validate_argument(verb: ControlVerb, argument: str) -> None:
    """Refuse an argument the verb does not take, or a missing one it needs.

    Args:
        verb: The verb.
        argument: Its argument.

    Raises:
        JSONTypeError: When ``hold`` names no holdable mode, ``doctrine``
            names no doctrine, or any other verb carries an argument.
    """
    if verb is ControlVerb.HOLD:
        mode = find_member(argument, AIMode)
        if mode is None or mode not in HOLDABLE_MODES:
            holdable = ", ".join(held.value for held in HOLDABLE_MODES)
            raise JSONTypeError(f"hold needs a mode to hold ({holdable}), got {argument!r}")
        return
    if verb is ControlVerb.DOCTRINE:
        if find_member(argument, EngagementDoctrine) is None:
            known = ", ".join(doctrine.value for doctrine in EngagementDoctrine)
            raise JSONTypeError(
                f"doctrine needs an engagement doctrine ({known}), got {argument!r}"
            )
        return
    if argument != "":
        raise JSONTypeError(f"{verb.value} takes no argument, got {argument!r}")


def make_control_command(verb: ControlVerb, argument: str) -> ControlCommandDict:
    """Build a validated control command.

    Args:
        verb: The verb.
        argument: Its argument, the empty string for a verb without one.

    Returns:
        The command.

    Raises:
        JSONTypeError: As :func:`_validate_argument`.
    """
    _validate_argument(verb, argument)
    return ControlCommandDict(verb=verb, argument=argument)


def encode_control_command(command: ControlCommandDict) -> JSONObject:
    """Encode a control command as JSON.

    Args:
        command: The command.

    Returns:
        ``{"verb": ..., "argument": ...}``.
    """
    return {"verb": command["verb"].value, "argument": command["argument"]}


def decode_control_command(data: JSONObject) -> ControlCommandDict:
    """Decode and validate a control command.

    Args:
        data: The decoded JSON object.

    Returns:
        The command.

    Raises:
        JSONTypeError: When a field is missing, the verb is unknown, or
            the argument does not fit the verb.
    """
    return make_control_command(
        require_member(data, "verb", ControlVerb), require_str(data, "argument")
    )


def control_file_path(run_dir: Path) -> Path:
    """Where a bot's control file lives.

    Args:
        run_dir: The bot's run directory, the one holding ``STOP``.

    Returns:
        ``<run_dir>/CONTROL``.
    """
    return run_dir / CONTROL_FILE_NAME


def control_pending(run_dir: Path) -> bool:
    """Whether a written verb has not been consumed yet.

    Args:
        run_dir: The bot's run directory.

    Returns:
        True while the control file exists.
    """
    return _test_hooks.path_exists(control_file_path(run_dir))


def write_control(run_dir: Path, command: ControlCommandDict) -> None:
    """Write a verb for the bot to take at its next tick, atomically.

    Args:
        run_dir: The bot's run directory.
        command: The validated command.
    """
    _test_hooks.replace_text(
        control_file_path(run_dir), dump_json_str(encode_control_command(command))
    )


def take_pending_control(run_dir: Path) -> ControlCommandDict | None:
    """Read and consume the pending verb, if one was written.

    Args:
        run_dir: The bot's run directory.

    Returns:
        The command, or ``None`` when no verb is pending.

    Raises:
        InvalidJsonError: When the file is not JSON.
        JSONTypeError: When it is not a valid command.
    """
    path = control_file_path(run_dir)
    if not _test_hooks.path_exists(path):
        return None
    raw = _test_hooks.read_text(path)
    _test_hooks.remove_file(path)
    return decode_control_command(narrow_json_to_dict(load_json_str(raw)))


def apply_control(ai_state: AIStateDict, command: ControlCommandDict) -> AIStateDict:
    """The AI state after a verb, every field written one an existing path reads.

    Args:
        ai_state: The AI state before the verb.
        command: A validated command.

    Returns:
        The new AI state.
    """
    verb = command["verb"]
    if verb is ControlVerb.HOLD:
        return AIStateDict(**{**ai_state, "manual_mode": AIMode(command["argument"])})
    if verb is ControlVerb.DISENGAGE:
        return AIStateDict(**{**clear_combat_target(ai_state), "manual_mode": AIMode.COLLECT})
    if verb is ControlVerb.WIND_DOWN:
        return AIStateDict(**{**ai_state, "wind_down": True, "manual_mode": None})
    if verb is ControlVerb.RELEASE:
        return AIStateDict(**{**ai_state, "manual_mode": None})
    config = AIConfigDict(
        **{**ai_state["config"], "doctrine": EngagementDoctrine(command["argument"])}
    )
    return AIStateDict(**{**ai_state, "config": config})


def apply_pending_control(ai_state: AIStateDict, run_dir: Path) -> AIStateDict:
    """Take the pending verb, if any, and return the state after it.

    Called at the top of every tick, beside the stop-file check's
    channel, so a verb lands at a tick boundary.

    Args:
        ai_state: The AI state before the tick.
        run_dir: The bot's run directory.

    Returns:
        The AI state after the verb, or ``ai_state`` unchanged when none
        is pending.

    Raises:
        InvalidJsonError: As :func:`take_pending_control`.
        JSONTypeError: As :func:`take_pending_control`.
    """
    command = take_pending_control(run_dir)
    if command is None:
        return ai_state
    log.info("Control verb %s %s applied", command["verb"].value, command["argument"])
    return apply_control(ai_state, command)


__all__ = [
    "CONTROL_FILE_NAME",
    "HOLDABLE_MODES",
    "ControlCommandDict",
    "ControlVerb",
    "apply_control",
    "apply_pending_control",
    "control_file_path",
    "control_pending",
    "decode_control_command",
    "encode_control_command",
    "make_control_command",
    "take_pending_control",
    "write_control",
]
