"""Re-attaching a restarted manager to the bots that kept playing.

The fleet spawns bots as child processes precisely so that losing the
manager cannot kill a live tank. The cost of that choice was paid on
every restart: the new manager started with an empty registry, so
bots that were still fighting became unreachable -- not stoppable, not
inspectable, not even visible -- and the only way to end one was to
find its pid by hand.

Adoption closes that. Every spawn leaves a record
(:mod:`tankpit_bot.service.fleet_record`); every boot reads those
records back and re-attaches to whichever processes are still alive,
discarding the records of the ones that finished while nobody was
watching.

What makes it safe is that a record names an IDENTITY, not just a pid.
Windows recycles pids, and a manager restarted minutes later could
otherwise "adopt" some unrelated program that happened to inherit the
number -- then refuse to restart the instance forever, because its
imaginary bot is always running. The creation time recorded next to
the pid is compared exactly, so a recycled pid is simply not a match.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet_bot import _ManagedBot
from tankpit_bot.service.fleet_record import (
    FleetProcessRecordDict,
    forget_process_record,
    read_process_record,
    recorded_instances,
)

log = get_logger(__name__)


def _adopt_one(record: FleetProcessRecordDict) -> _ManagedBot | None:
    """Re-attach to one recorded bot, if it is still the same process.

    Args:
        record: The spawn record read from disk.

    Returns:
        The re-attached registry entry, or ``None`` when nothing is
        running under the recorded identity.
    """
    instance = record["instance"]
    process = service_hooks.open_adopted_process(record["pid"], record["created_at"])
    if process is None:
        return None
    log.info(
        "Fleet: adopted instance %r pid %d (role=%s account=%s room=%s)",
        instance,
        record["pid"],
        record["role"],
        record["account"] or "default",
        record["room"] or "Practice",
    )
    return _ManagedBot(
        instance=instance,
        account=record["account"],
        role=record["role"],
        room=record["room"],
        troop=record["troop"],
        doctrine=record["doctrine"],
        kills=record["kills"],
        seconds=record["seconds"],
        started_ms=record["started_ms"],
        process=process,
    )


def adopt_recorded_bots() -> list[_ManagedBot]:
    """Re-attach to every recorded bot that is still running.

    Records whose process is gone are DELETED here rather than left to
    be re-checked on every future boot: the run they describe is over,
    and its other artifacts (log, events, scorecard) remain untouched
    as the record of what happened.

    A record that fails to decode is NOT swallowed. Records are
    written atomically, so a malformed one is real corruption rather
    than a torn write, and starting a manager on top of it would mean
    silently forgetting a tank that may well still be playing.

    Returns:
        Registry entries for the surviving bots, in instance-name
        order.

    Raises:
        OSError: If a listed record cannot be read.
        InvalidJsonError: If a record is not valid JSON.
        JSONTypeError: If a record is missing fields or malformed.
    """
    adopted: list[_ManagedBot] = []
    for instance in recorded_instances():
        record = read_process_record(instance)
        bot = _adopt_one(record)
        if bot is None:
            log.info(
                "Fleet: instance %r finished while unsupervised (pid %d); record cleared",
                instance,
                record["pid"],
            )
            forget_process_record(instance)
            continue
        adopted.append(bot)
    return adopted


__all__ = [
    "adopt_recorded_bots",
]
