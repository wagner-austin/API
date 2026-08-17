"""The match fleet: spawn, observe, and stop headless matches.

Built 2026-08-06, the RustedWarfare side of the two-game fleet ruling
("i want to ensure we can easily run multiple bots on the desktop.
with a simple ui, easy for me or the ai to use"). One user-owned
manager process; each match is a ``make play`` child, so every knob
the Makefile owns — seed, map, opponents, difficulty, the
fast-forward multiplier, a frozen tree — is a spawn parameter and the
launch plumbing is never duplicated. ``PLAY_PORT`` self-randomizes
per invocation (27600-27999), which is what makes parallel matches
safe by design.

Per-instance artifacts live under ``runs/fleet/``: ``<instance>.out``
is the child's combined transcript (the planner prints its match
report there at the end — the stats source), and the game log is
directed to ``runs/fleet/<instance>.log``.

Unlike the tankpit fleet, STOP here kills the process tree outright:
a headless simulation owes no teardown, holds no account, and loses
nothing but its own wall-clock when felled mid-match.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks

_BAD_NAME = "RW-FLEET-001"
_BAD_BOUND = "RW-FLEET-002"
_ALREADY_RUNNING = "RW-FLEET-003"
_UNKNOWN = "RW-FLEET-004"
_STILL_RUNNING = "RW-FLEET-005"
_NOT_RUNNING = "RW-FLEET-006"

_INSTANCE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

_FLEET_DIR = Path("runs") / "fleet"

#: The report line the planner prints first at match end; its presence
#: in a transcript is the finished-match signal the stats read.
_VERDICT_PREFIX = "verdict"


class FleetError(RwBotError):
    """A fleet operation the HTTP layer maps to a 4xx response.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description.
    """


class FleetMatchRow(TypedDict):
    """One managed match, as reported by ``GET /bots``.

    Attributes:
        instance: Validated instance name (artifact namespace).
        seed: ``PLAY_SEED`` the match was spawned with.
        map: ``PLAY_MAP`` (empty means the Makefile default).
        opponents: ``PLAY_OPPONENTS``.
        difficulty: ``PLAY_DIFFICULTY``.
        fastforward: ``PLAY_FASTFORWARD`` (0 leaves the wall clock).
        tree: ``PLAY_TREE`` frozen-tree path (empty runs the worktree).
        pid: Child process id (the ``make`` root of the tree).
        alive: Whether the process is still running at report time.
        returncode: Exit code once dead; ``None`` while alive.
    """

    instance: str
    seed: int
    map: str
    opponents: int
    difficulty: int
    fastforward: int
    tree: str
    pid: int
    alive: bool
    returncode: int | None


class FleetStats(TypedDict):
    """A match transcript's tail, reduced for the control page.

    Attributes:
        available: Whether the transcript held any lines yet.
        finished: Whether the planner's match report was found.
        verdict: The report's verdict line, empty until finished.
        report: Every report line from the verdict onward, in order.
    """

    available: bool
    finished: bool
    verdict: str
    report: list[str]


class _ManagedMatch:
    """Registry entry pairing spawn parameters with the live process."""

    def __init__(
        self,
        *,
        instance: str,
        seed: int,
        map_name: str,
        opponents: int,
        difficulty: int,
        fastforward: int,
        tree: str,
        process: _test_hooks.SpawnedMatchProto,
    ) -> None:
        """Bind one spawned match to its parameters.

        Args:
            instance: Validated instance name.
            seed: ``PLAY_SEED`` the child received.
            map_name: ``PLAY_MAP`` the child received.
            opponents: ``PLAY_OPPONENTS`` the child received.
            difficulty: ``PLAY_DIFFICULTY`` the child received.
            fastforward: ``PLAY_FASTFORWARD`` the child received.
            tree: ``PLAY_TREE`` the child received.
            process: The spawned child process handle.
        """
        self.instance = instance
        self.seed = seed
        self.map_name = map_name
        self.opponents = opponents
        self.difficulty = difficulty
        self.fastforward = fastforward
        self.tree = tree
        self.process = process

    def row(self) -> FleetMatchRow:
        """Return the match's current state for ``GET /bots``.

        Returns:
            The typed report row.
        """
        returncode = self.process.poll()
        return FleetMatchRow(
            instance=self.instance,
            seed=self.seed,
            map=self.map_name,
            opponents=self.opponents,
            difficulty=self.difficulty,
            fastforward=self.fastforward,
            tree=self.tree,
            pid=self.process.pid,
            alive=returncode is None,
            returncode=returncode,
        )


def transcript_path(instance: str) -> Path:
    """Return the transcript file for one instance.

    Args:
        instance: Validated instance name.

    Returns:
        ``runs/fleet/<instance>.out``.
    """
    return _FLEET_DIR / f"{instance}.out"


def _make_argv(
    *,
    instance: str,
    seed: int,
    map_name: str,
    opponents: int,
    difficulty: int,
    fastforward: int,
    tree: str,
) -> list[str]:
    """Compose the ``make play`` invocation for one match.

    The Makefile stays the single owner of the launch plumbing — the
    fleet only sets the per-match variables it exposes.

    Args:
        instance: Validated instance name.
        seed: ``PLAY_SEED``.
        map_name: ``PLAY_MAP`` (empty keeps the Makefile default).
        opponents: ``PLAY_OPPONENTS``.
        difficulty: ``PLAY_DIFFICULTY``.
        fastforward: ``PLAY_FASTFORWARD``.
        tree: ``PLAY_TREE`` (empty runs the worktree).

    Returns:
        Argument vector, program first.
    """
    argv = [
        "make",
        "play",
        f"PLAY_SEED={seed}",
        f"PLAY_OPPONENTS={opponents}",
        f"PLAY_DIFFICULTY={difficulty}",
        f"PLAY_FASTFORWARD={fastforward}",
        f"PLAY_LOG=runs/fleet/{instance}.log",
    ]
    if map_name:
        argv.append(f"PLAY_MAP={map_name}")
    if tree:
        argv.append(f"PLAY_TREE={tree}")
    return argv


class FleetManager:
    """Spawn and track one ``make play`` child per instance name."""

    def __init__(self) -> None:
        """Start with an empty registry."""
        self._matches: dict[str, _ManagedMatch] = {}

    def spawn(
        self,
        *,
        instance: str,
        seed: int,
        map_name: str,
        opponents: int,
        difficulty: int,
        fastforward: int,
        tree: str,
    ) -> FleetMatchRow:
        """Spawn one match child process.

        Args:
            instance: Instance name; rejected here rather than by a
                crashed child.
            seed: ``PLAY_SEED``; non-negative.
            map_name: ``PLAY_MAP`` (empty keeps the Makefile default).
            opponents: ``PLAY_OPPONENTS``; non-negative.
            difficulty: ``PLAY_DIFFICULTY``; non-negative.
            fastforward: ``PLAY_FASTFORWARD``; non-negative, 0 leaves
                the engine at the wall clock.
            tree: ``PLAY_TREE`` frozen-tree path (empty runs the
                worktree).

        Returns:
            The spawned match's report row.

        Raises:
            FleetError: If the name is invalid, a number is negative,
                or the instance is already registered and alive.
        """
        if not _INSTANCE_NAME.match(instance):
            raise FleetError(
                _BAD_NAME,
                f"instance {instance!r} is not a valid instance name "
                "(lowercase alphanumeric plus -_, max 32 chars)",
            )
        for label, value in (
            ("seed", seed),
            ("opponents", opponents),
            ("difficulty", difficulty),
            ("fastforward", fastforward),
        ):
            if value < 0:
                raise FleetError(_BAD_BOUND, f"{label} must be non-negative, got {value}")
        existing = self._matches.get(instance)
        if existing is not None and existing.process.poll() is None:
            raise FleetError(
                _ALREADY_RUNNING,
                f"instance {instance!r} is already running (pid {existing.process.pid})",
            )
        argv = _make_argv(
            instance=instance,
            seed=seed,
            map_name=map_name,
            opponents=opponents,
            difficulty=difficulty,
            fastforward=fastforward,
            tree=tree,
        )
        process = _test_hooks.spawn_match(argv, transcript_path(instance))
        match = _ManagedMatch(
            instance=instance,
            seed=seed,
            map_name=map_name,
            opponents=opponents,
            difficulty=difficulty,
            fastforward=fastforward,
            tree=tree,
            process=process,
        )
        self._matches[instance] = match
        _test_hooks.write_line(
            f"[fleet] spawned {instance!r} pid {process.pid}"
            f" (seed={seed} ff={fastforward} opponents={opponents})"
        )
        return match.row()

    def report(self) -> list[FleetMatchRow]:
        """Return every registered match's current state.

        Returns:
            Report rows sorted by instance name.
        """
        return [self._matches[name].row() for name in sorted(self._matches)]

    def stats(self, instance: str) -> FleetStats:
        """Reduce a registered match's transcript for the control page.

        Args:
            instance: Registered instance name.

        Returns:
            The transcript tail: ``available`` is False until the child
            has written anything; ``finished`` and the report lines
            appear once the planner printed its match report.

        Raises:
            FleetError: If the instance is not registered.
        """
        if instance not in self._matches:
            raise FleetError(_UNKNOWN, f"unknown instance {instance!r}")
        try:
            lines = _test_hooks.read_text_lines(transcript_path(instance))
        except FileNotFoundError:
            # Absence IS the answer: the child has not written yet, which is
            # the state ``available=False`` exists to report, and it recurs
            # every poll until the first write. Only absence -- a permission
            # or disk failure is not "not started yet" and propagates.
            return FleetStats(available=False, finished=False, verdict="", report=[])
        report_lines: list[str] = []
        for index, line in enumerate(lines):
            if line.startswith(_VERDICT_PREFIX):
                report_lines = list(lines[index:])
                break
        return FleetStats(
            available=bool(lines),
            finished=bool(report_lines),
            verdict=report_lines[0] if report_lines else "",
            report=report_lines,
        )

    def stop(self, instance: str) -> FleetMatchRow:
        """Kill a live match's whole process tree.

        A headless simulation owes no teardown — unlike the tankpit
        fleet's sentinel stop, this is immediate and forceful.

        Args:
            instance: Registered instance name.

        Returns:
            The match's report row after the kill was issued.

        Raises:
            FleetError: If the instance is unknown or already finished.
        """
        match = self._matches.get(instance)
        if match is None:
            raise FleetError(_UNKNOWN, f"unknown instance {instance!r}")
        if match.process.poll() is not None:
            raise FleetError(_NOT_RUNNING, f"instance {instance!r} already finished")
        _test_hooks.kill_tree(match.process.pid)
        _test_hooks.write_line(f"[fleet] killed {instance!r} (pid {match.process.pid})")
        return match.row()

    def restart(self, instance: str) -> FleetMatchRow:
        """Respawn a finished match with the parameters it had.

        Args:
            instance: Registered instance name.

        Returns:
            The respawned match's report row.

        Raises:
            FleetError: If the instance is unknown or still alive.
        """
        match = self._matches.get(instance)
        if match is None:
            raise FleetError(_UNKNOWN, f"unknown instance {instance!r}")
        if match.process.poll() is None:
            raise FleetError(
                _STILL_RUNNING, f"instance {instance!r} is still running; stop it first"
            )
        return self.spawn(
            instance=instance,
            seed=match.seed,
            map_name=match.map_name,
            opponents=match.opponents,
            difficulty=match.difficulty,
            fastforward=match.fastforward,
            tree=match.tree,
        )

    def remove(self, instance: str) -> FleetMatchRow:
        """Drop a finished match from the registry.

        Args:
            instance: Registered instance name.

        Returns:
            The removed match's final report row.

        Raises:
            FleetError: If the instance is unknown or still alive.
        """
        match = self._matches.get(instance)
        if match is None:
            raise FleetError(_UNKNOWN, f"unknown instance {instance!r}")
        if match.process.poll() is None:
            raise FleetError(
                _STILL_RUNNING, f"instance {instance!r} is still running; stop it first"
            )
        del self._matches[instance]
        return match.row()


__all__ = [
    "FleetError",
    "FleetManager",
    "FleetMatchRow",
    "FleetStats",
    "transcript_path",
]
