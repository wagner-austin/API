"""A node's CI runners and the action archive cache, as its probe reads them (MCPs ce2b92e0).

A runner host's root timer (:mod:`fleet.core.runner_cache_adopt`) publishes
four counts in a file any account can read, and the node runner's capacity
probe forwards them beside free memory and disk, so each tick says whether
the host's runners take their workflow actions from the cache. The tick
carries them to MCPs ``dispatch_tick`` (MCPs mig 767), where a query can
read them; until then, whether each runner read the cache was visible only
in its own ``_diag`` Worker log inside the distro, which no check could
reach.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, require_int
from typing_extensions import TypedDict

#: Where a runner host's root timer publishes the counts, one ``key=value``
#: line each, readable by every account and holding nothing else: no log
#: line and no token leaves the runner account's home.
STATUS_PATH: Final = "/var/lib/fleet-action-cache/status"

#: Minutes after which the probe ignores the file: the timer rewrites it every
#: minute, so an older one says the timer stopped, and its counts no longer
#: describe the runners.
STATUS_FRESH_MINUTES: Final = 5

#: The probe's keys for the four counts, in the order the status file holds them.
PROBE_FIELDS: Final[tuple[str, str, str, str]] = (
    "action_cache_runners",
    "action_cache_reading",
    "action_cache_served",
    "action_cache_missed",
)


class ActionCacheCounts(TypedDict):
    """Whether a host's runners take their actions from its archive cache.

    ``served`` and ``missed`` read each runner's newest job since its process
    started; a runner with no job since then is in neither.

    Attributes:
        runners: The runner units the host's timer examines.
        reading: Those whose running process carries
            ``ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE``.
        served: Those whose newest job looked up at least one action and
            found every one in the cache.
        missed: Those whose newest job looked up an action the cache did not
            hold, so fetched it from codeload.
    """

    runners: int
    reading: int
    served: int
    missed: int


def inconsistency(counts: ActionCacheCounts) -> str | None:
    """What makes counts unable to describe one set of runners, if anything.

    Args:
        counts: The four counts.

    Returns:
        None when they can, or a sentence naming all four when a count is
        negative, more runners read the cache than were counted, or more
        were served and missed together.
    """
    runners, reading = counts["runners"], counts["reading"]
    served, missed = counts["served"], counts["missed"]
    if min(runners, reading, served, missed) < 0 or reading > runners or served + missed > runners:
        return (
            f"action cache counts do not describe one set of runners: {runners} runner(s), "
            f"{reading} reading, {served} served, {missed} missed"
        )
    return None


def require_consistent(counts: ActionCacheCounts) -> ActionCacheCounts:
    """Refuse counts that cannot describe one set of runners.

    Args:
        counts: The four counts.

    Returns:
        The same counts.

    Raises:
        JSONTypeError: As :func:`inconsistency` describes.
    """
    problem = inconsistency(counts)
    if problem is not None:
        raise JSONTypeError(problem)
    return counts


def encode_action_cache(counts: ActionCacheCounts | None) -> JSONObject | None:
    """The counts as ``dispatch_tick``'s load carries them.

    Args:
        counts: The counts, or None when the node publishes none.

    Returns:
        ``{"runners", "reading", "served", "missed"}``, or None.
    """
    if counts is None:
        return None
    return {
        "runners": counts["runners"],
        "reading": counts["reading"],
        "served": counts["served"],
        "missed": counts["missed"],
    }


def decode_action_cache(value: JSONValue) -> ActionCacheCounts | None:
    """Decode the counts from the shape :func:`encode_action_cache` writes.

    Args:
        value: The encoded counts, or None.

    Returns:
        The counts, or None.

    Raises:
        JSONTypeError: If it is neither an object nor None, a count is
            missing or not a whole number, or the counts are inconsistent.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"actionCache must be a JSON object or null, got {type(value).__name__}"
        )
    return require_consistent(
        ActionCacheCounts(
            runners=require_int(value, "runners"),
            reading=require_int(value, "reading"),
            served=require_int(value, "served"),
            missed=require_int(value, "missed"),
        )
    )


__all__ = [
    "PROBE_FIELDS",
    "STATUS_FRESH_MINUTES",
    "STATUS_PATH",
    "ActionCacheCounts",
    "decode_action_cache",
    "encode_action_cache",
    "inconsistency",
    "require_consistent",
]
