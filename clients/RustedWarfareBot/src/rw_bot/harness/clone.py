"""Per-worker copies of the pinned game directory.

A headless match is already isolated from every other one in all but a single
respect. The agent jar is stamped with a GUID per invocation, the channel port
is drawn per invocation, and log paths are named by the caller. What is *not*
isolated is the game's own directory: a running match writes three fixed-name
paths inside it -- ``preferences.ini``, ``saves/autosave.rwsave.tmp2``, and the
``cache/mods-info.cachedata`` tree -- so two matches launched from one directory
race on all three.

Copying the directory is therefore the whole of what concurrent matches require,
and at roughly 0.44 GB a copy that is a cheap trade for playing several matches
at once ([[harness-nodisplay]]).

Everything here is pure: which entries a copy needs, which it must not take, and
what proves a copy usable. The copying itself belongs to the entry point.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot import RwBotError

_INCOMPLETE = "RW-CLONE-001"
_NOT_POSITIVE = "RW-CLONE-002"

#: Directories a clone creates empty rather than copying.
#:
#: These are the two trees the game rewrites. Copying them risks reading one
#: while a match already in flight is part-way through writing it, and the game
#: rebuilds both on boot, so there is nothing in them worth the risk.
VOLATILE_DIRS = ("saves", "cache")

#: Directories a headless match never reads.
#:
#: The launcher names ``jvm64`` for the game, for ``javac`` and for ``jar``, and
#: nothing anywhere names the 32-bit tree beside it. At 118 MB it is the single
#: largest thing in the directory after the JVM that *is* used, and it would
#: otherwise be copied once per worker for nothing.
UNUSED_DIRS = ("jvm",)

#: Files a match starts from the pinned copy of, every time.
#:
#: The game rewrites ``preferences.ini`` on every boot, so a clone carries the
#: previous match's copy into the next one and the clones drift apart from each
#: other. Measured, the only key that moves is ``nextBackgroundMap`` -- a main
#: menu counter, and the harness runs ``-nodisplay`` -- so the drift observed so
#: far cannot reach the simulation.
#:
#: It is reset anyway, because the property an experiment needs is not "the
#: state that differs happens to be harmless" but "the state does not differ".
#: Nothing guarantees the next key the engine writes here is a cosmetic one, and
#: a settings difference between workers would show up as unexplainable variance
#: between arms rather than as an error ([[policy-determinism]]).
VOLATILE_FILES = ("preferences.ini",)

#: Paths that must exist in a clone before it is handed to a worker.
#:
#: The launcher runs ``<GAME_DIR>/jvm64/bin/java.exe`` and compiles the agent
#: with the JDK beside it. A clone missing any of them fails ninety seconds
#: later as "the agent never opened port N", which reads like a fault in the
#: agent rather than a truncated copy -- so the copy is checked where the
#: message can still be accurate.
REQUIRED_ENTRIES = (
    "jvm64/bin/java.exe",
    "jvm64/bin/javac.exe",
    "jvm64/bin/jar.exe",
    "game-lib.jar",
    "libs",
    "assets",
)


class CloneError(RwBotError):
    """A worker's copy of the game directory is unusable.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what is missing.
    """


def clone_name(prefix: str, index: int) -> str:
    """Return the directory name for one worker's copy.

    Args:
        prefix: Shared leading part of every clone's name.
        index: Which worker this is, from zero.

    Returns:
        The directory name, numbered from one so it reads like the worker
        labels in a sweep's output.

    Raises:
        CloneError: ``RW-CLONE-002`` when the index is negative.
    """
    if index < 0:
        raise CloneError(_NOT_POSITIVE, f"a worker index cannot be negative, got {index}")
    return f"{prefix}{index + 1}"


def entries_to_copy(present: Sequence[str]) -> tuple[str, ...]:
    """Return the top-level entries a clone takes from the source directory.

    Everything except the trees the game rewrites and the one it never reads,
    listed by exclusion rather than by an allow-list: a directory added to the
    game by a future patch is then copied rather than silently dropped, and a
    dropped directory shows up as a match that will not boot.

    Args:
        present: Top-level entry names in the source directory.

    Returns:
        The names to copy, in the order given.
    """
    skip = (*VOLATILE_DIRS, *UNUSED_DIRS)
    return tuple(name for name in present if name not in skip)


def missing_requirements(present: Sequence[str]) -> tuple[str, ...]:
    """Return the required paths a finished clone does not have.

    Args:
        present: Paths that exist in the clone, relative to it, with forward
            slashes.

    Returns:
        The required paths that are absent, in declaration order.
    """
    have = set(present)
    return tuple(needed for needed in REQUIRED_ENTRIES if needed not in have)


def verify(name: str, present: Sequence[str]) -> None:
    """Raise unless a clone carries everything a match needs.

    Args:
        name: The clone's directory name, for the message.
        present: Paths that exist in the clone, relative to it, with forward
            slashes.

    Raises:
        CloneError: ``RW-CLONE-001`` when anything required is absent, naming
            every missing path rather than only the first.
    """
    missing = missing_requirements(present)
    if missing:
        raise CloneError(
            _INCOMPLETE,
            f"clone {name!r} is missing {', '.join(missing)}: a match launched from it "
            "would fail as a channel that never opens rather than as a bad copy",
        )


__all__ = [
    "REQUIRED_ENTRIES",
    "UNUSED_DIRS",
    "VOLATILE_DIRS",
    "VOLATILE_FILES",
    "CloneError",
    "clone_name",
    "entries_to_copy",
    "missing_requirements",
    "verify",
]
