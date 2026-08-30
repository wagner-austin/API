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
from rw_bot.harness.jvm import bundled_tools, tool_path

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

#: The one tool every match needs, whether or not it builds an agent.
JAVA_TOOL = "java"

#: Paths a clone must have whatever it runs on.
#:
#: The game's own code, the libraries it loads beside it, and the assets a map
#: is read from. None of the three is named differently on any platform, which
#: is what separates them from the JDK tools listed with them.
REQUIRED_CONTENT = ("game-lib.jar", "libs", "assets")


def required_entries(platform: str, *, compiles_agent: bool) -> tuple[str, ...]:
    """Return the paths that must exist in a clone before a worker gets it.

    A clone missing any of them fails ninety seconds later as "the agent never
    opened port N", which reads like a fault in the agent rather than a
    truncated copy -- so the copy is checked here, where the message can still
    be accurate.

    THE COMPILER IS NOT ALWAYS REQUIRED, and demanding it was wrong. Every
    match runs ``java`` out of the clone; only a match that BUILDS its agent
    needs ``javac`` and ``jar`` beside it. The Linux depot ships a JRE with
    neither, so requiring them unconditionally rejected every valid Linux tree
    -- and a batch does not need them there anyway, because it compiles its
    agent once and carries the jar inside its frozen tree.

    Args:
        platform: A ``sys.platform`` value.
        compiles_agent: Whether this run builds its own agent. False for a
            batch reusing a frozen snapshot, which is every cluster member.

    Returns:
        The required paths, JDK tools first, in declaration order. Only the
        tools the platform's bundled runtime actually carries are ever asked
        for (:func:`~rw_bot.harness.jvm.bundled_tools`).
    """
    tools = bundled_tools(platform) if compiles_agent else (JAVA_TOOL,)
    return (*(tool_path(tool, platform) for tool in tools), *REQUIRED_CONTENT)


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


#: First channel port of the leased band; clone ordinal N binds BASE + N.
#: Above the fleet page's 27500 and the match service's 27501, below the
#: recipe's own 27600-27999 random band so a leased port and a drawn one
#: can never collide either.
PLAY_PORT_BASE = 27510

#: Leading part of every worker copy's directory name.
#:
#: Here rather than beside the sweep entry point that used to hold it: the
#: cluster's single-match entry runs from the installed package, which cannot
#: import ``scripts/`` because ``scripts/`` is not in the wheel. Two spellings
#: of one prefix would have meant a clone leasing a port under one name and
#: being looked for under another.
CLONE_PREFIX = ".game-w"


def leased_port(game_dir: str, prefix: str) -> int:
    """Return the channel port a cloned game directory owns, or zero.

    The play recipe draws a random port per invocation, and the first time
    eight matches launched in one instant the draws collided -- two agents
    bound the same port and both matches died (imp-creep12, 2026-08-08).
    Random draws are not leases. A clone's ordinal IS a lease -- the
    allocator guarantees no two concurrent matches share one -- so a port
    derived from it inherits that exclusivity for free.

    Args:
        game_dir: The game directory the match plays in.
        prefix: Shared leading part of every clone's name.

    Returns:
        ``PLAY_PORT_BASE`` plus the clone's ordinal, or zero when the
        directory is not a numbered clone -- the single-match entry points
        play in the pinned directory itself, and their recipe keeps its
        random draw.
    """
    ordinal = _clone_ordinal(game_dir, prefix)
    if ordinal is None:
        return 0
    return PLAY_PORT_BASE + ordinal


#: First X display of the leased band; clone ordinal N runs on BASE + N.
#:
#: Well clear of ``:0``, which is a physical console. A headless match must
#: never take that one -- on a workstation it is somebody's desktop, and on a
#: login node it is whatever the last interactive session left behind.
DISPLAY_BASE = 90

#: What :func:`leased_display` returns for a directory that is not a numbered
#: clone: no X server is to be started.
#:
#: Zero rather than a separate flag because display ``:0`` is never a legal
#: answer here anyway (see :data:`DISPLAY_BASE`), so the number has a spare
#: value and a second field would be a second thing to keep in step.
NO_DISPLAY = 0


def leased_display(game_dir: str, prefix: str) -> int:
    """Return the X display a cloned game directory owns, or zero.

    The same argument as :func:`leased_port`, for the same reason. Under
    ``-nodisplay`` the engine still opens a Slick2D display and creates
    framebuffer objects, so on a machine with no X server every concurrent
    match needs one of its own ([[harness-nodisplay]]). Two matches sharing a
    display number race exactly as two matches sharing a port do, and the
    clone's ordinal is already an exclusive lease.

    Args:
        game_dir: The game directory the match plays in.
        prefix: Shared leading part of every clone's name.

    Returns:
        :data:`DISPLAY_BASE` plus the clone's ordinal, or :data:`NO_DISPLAY`
        when the directory is not a numbered clone -- the single-match entry
        points run wherever the caller already has a display, which on a
        workstation is the desktop and needs no server started for it.
    """
    ordinal = _clone_ordinal(game_dir, prefix)
    if ordinal is None:
        return NO_DISPLAY
    return DISPLAY_BASE + ordinal


def _clone_ordinal(game_dir: str, prefix: str) -> int | None:
    """Return the worker number a clone directory names.

    Args:
        game_dir: The game directory.
        prefix: Shared leading part of every clone's name.

    Returns:
        The ordinal, or None when the directory is not a numbered clone.
    """
    if not game_dir.startswith(prefix):
        return None
    ordinal = game_dir[len(prefix) :]
    if not ordinal.isdigit():
        return None
    return int(ordinal)


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


def missing_requirements(
    present: Sequence[str], platform: str, *, compiles_agent: bool
) -> tuple[str, ...]:
    """Return the required paths a finished clone does not have.

    Args:
        present: Paths that exist in the clone, relative to it, with forward
            slashes.
        platform: A ``sys.platform`` value, which decides what the JDK's tools
            are called.
        compiles_agent: Whether this run builds its own agent.

    Returns:
        The required paths that are absent, in declaration order.
    """
    have = set(present)
    return tuple(
        needed
        for needed in required_entries(platform, compiles_agent=compiles_agent)
        if needed not in have
    )


def verify(name: str, present: Sequence[str], platform: str, *, compiles_agent: bool) -> None:
    """Raise unless a clone carries everything a match needs.

    Args:
        name: The clone's directory name, for the message.
        present: Paths that exist in the clone, relative to it, with forward
            slashes.
        platform: A ``sys.platform`` value.
        compiles_agent: Whether this run builds its own agent.

    Raises:
        CloneError: ``RW-CLONE-001`` when anything required is absent, naming
            every missing path rather than only the first.
    """
    missing = missing_requirements(present, platform, compiles_agent=compiles_agent)
    if missing:
        raise CloneError(
            _INCOMPLETE,
            f"clone {name!r} is missing {', '.join(missing)}: a match launched from it "
            "would fail as a channel that never opens rather than as a bad copy",
        )


__all__ = [
    "CLONE_PREFIX",
    "DISPLAY_BASE",
    "JAVA_TOOL",
    "NO_DISPLAY",
    "PLAY_PORT_BASE",
    "REQUIRED_CONTENT",
    "UNUSED_DIRS",
    "VOLATILE_DIRS",
    "VOLATILE_FILES",
    "CloneError",
    "clone_name",
    "entries_to_copy",
    "leased_display",
    "leased_port",
    "missing_requirements",
    "required_entries",
    "verify",
]
