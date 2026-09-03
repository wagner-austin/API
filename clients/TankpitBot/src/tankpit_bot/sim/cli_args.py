"""The sim CLI's flags, and the gate that refuses an unnamed world.

Split from :mod:`tankpit_bot.sim.scenarios` 2026-09-02 (the 400-600 line
rule): that module owns WHAT world a session plays on, and it had reached
593 lines, so the flag parsing that had accumulated there moved here rather
than being squeezed under the ceiling.

The gate is the reason this is not merely a move. ``--layout`` and
``--population-seed`` are optional by design -- an interactive soak wants
the stamp to keep varying the room for free -- and that is exactly what
makes them forgettable on a sweep member, where the same defaulting turns
into a confound nothing warns about ([[sim-world-parameterization]]).
:func:`require_named_world` turns the omission into a refusal wherever the
run's numbers are going to be compared with another run's.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from tankpit_bot.sim.atlas_seed import DEFAULT_ATLAS_PATH

_SIM_DIR = Path("runs") / "sim"
_DEFAULT_ROUNDS = 150

ARRAY_TASK_ENV_VAR = "SLURM_ARRAY_TASK_ID"
"""What Slurm sets on every task of a job array.

The same variable :mod:`hpc3.contracts.array` reads, so the gate fires on
the real cluster rather than on a spelling of it this package invented.
Its presence means this process is one member of a set whose results will
be compared -- which is the condition the gate exists for, and one nobody
has to remember to declare.
"""


class UnnamedWorldError(RuntimeError):
    """Raised when a run whose numbers will be compared left its world to the stamp."""


class _CliArgsDict(TypedDict):
    """The sim CLI's parsed flags.

    ``out`` is the archive directory the session's capture and world
    land in; it defaults to the shared ``runs/sim`` archive and is
    pointed elsewhere by ``scripts.build_sim_baseline``, which needs a
    directory holding exactly one generation of the sim.

    ``layout`` and ``population_seed`` STATE the world instead of letting
    the stamp imply it. Both default to None, which means "derive from the
    stamp as before" and keeps interactive runs varying for free; a sweep
    member sets them and the stamp becomes a pure label
    ([[sim-world-parameterization]]).

    ``runs_root`` relocates the probe log and event artifacts, which are
    otherwise fixed under ``runs/probe``. N array tasks sharing a node
    share that path and clobber each other's ``latest.*``.
    """

    rounds: int
    opponent: bool
    practice: bool
    ferry: bool
    larder: bool
    atlas: str | None
    ghost: str | None
    stamp: str | None
    opponent_name: str
    out: str
    layout: str | None
    population_seed: int | None
    runs_root: str | None
    sweep: bool


def _apply_valued_flag(parsed: _CliArgsDict, token: str, value: str) -> bool:
    """Apply one flag whose meaning is carried by the NEXT token.

    Args:
        parsed: The bundle being filled (mutated on a match).
        token: The flag token.
        value: The token following it.

    Returns:
        True when ``token`` is a valued flag and both tokens are
        consumed; False when it is not one, leaving it for
        :func:`_apply_bare_flag`.

    Raises:
        ValueError: If ``--rounds`` or ``--population-seed`` names a
            non-integer. Neither a tick count nor a determinism seed is
            something to guess at when the caller mistyped it.
    """
    if token == "--rounds":
        parsed["rounds"] = int(value)
    elif token == "--ghost":
        parsed["ghost"] = value
    elif token == "--stamp":
        parsed["stamp"] = value
    elif token == "--human-opponent":
        parsed["opponent_name"] = value
    elif token == "--out":
        parsed["out"] = value
    elif token == "--layout":
        parsed["layout"] = value
    elif token == "--population-seed":
        parsed["population_seed"] = int(value)
    elif token == "--runs-root":
        parsed["runs_root"] = value
    else:
        return False
    return True


def _apply_bare_flag(parsed: _CliArgsDict, token: str, rest: list[str]) -> int:
    """Apply one flag that stands on its own.

    ``--from-atlas`` is the reason this takes ``rest`` rather than a
    single value: its path is OPTIONAL, so it reads the next token
    only when that token is not itself a flag.

    Args:
        parsed: The bundle being filled (mutated on a match).
        token: The flag token.
        rest: The tokens after it.

    Returns:
        How many tokens were consumed — 2 for ``--from-atlas PATH``,
        1 for everything else including tokens this does not
        recognise, which are skipped.
    """
    if token == "--no-opponent":
        parsed["opponent"] = False
    elif token == "--practice":
        parsed["practice"] = True
    elif token == "--ferry":
        parsed["ferry"] = True
    elif token == "--larder":
        parsed["larder"] = True
    elif token == "--sweep":
        parsed["sweep"] = True
    elif token == "--from-atlas":
        if rest and not rest[0].startswith("--"):
            parsed["atlas"] = rest[0]
            return 2
        parsed["atlas"] = str(DEFAULT_ATLAS_PATH)
    return 1


def _parse_cli(args: list[str]) -> _CliArgsDict:
    """Parse the manual flag loop into one typed bundle.

    The two flag SHAPES are parsed separately — a flag whose value is
    the next token, and a flag that stands alone — because a single
    chain covering both grew past the branch ceiling the moment a
    seventh flag arrived, and the two shapes have genuinely different
    consumption rules.

    Args:
        args: Raw CLI tokens.

    Returns:
        The parsed flags (unknown tokens are skipped).
    """
    parsed = _CliArgsDict(
        rounds=_DEFAULT_ROUNDS,
        opponent=True,
        practice=False,
        ferry=False,
        larder=False,
        atlas=None,
        ghost=None,
        stamp=None,
        opponent_name="",
        out=str(_SIM_DIR),
        layout=None,
        population_seed=None,
        runs_root=None,
        sweep=False,
    )
    index = 0
    while index < len(args):
        token = args[index]
        if index + 1 < len(args) and _apply_valued_flag(parsed, token, args[index + 1]):
            index += 2
            continue
        index += _apply_bare_flag(parsed, token, args[index + 1 :])
    return parsed


def require_named_world(
    parsed: _CliArgsDict,
    get_env: Callable[[str], str | None],
) -> None:
    """Refuse a stamp-derived world where the numbers will be compared.

    Omitting ``--layout`` or ``--population-seed`` is correct interactively
    and a confound in a measurement: the stamp then chooses the practice
    layout and the container field, so an array whose tasks stamp
    themselves varies the world along with whatever it meant to vary. That
    cost a retracted result on 2026-09-01, and a log line saying "derived
    from stamp" is not a check -- nobody reads 96 task logs.

    Fires on either of two conditions, deliberately:

    * ``--sweep`` was passed, which DECLARES the intent to compare.
    * :data:`ARRAY_TASK_ENV_VAR` is set, which BETRAYS it. This is the
      one that matters, because the failure mode being guarded is a
      forgotten flag, and a gate you must remember to arm does not guard
      against forgetting.

    Args:
        parsed: The parsed flags.
        get_env: Reader for a process environment variable, injected so a
            test can state a cluster without running on one.

    Raises:
        UnnamedWorldError: When comparison is intended and either world
            input was left to the stamp. Names the missing flags and the
            reason, because the reader is looking at a failed array task
            and needs to know what to add rather than that something was
            wrong.
    """
    array_task = get_env(ARRAY_TASK_ENV_VAR)
    in_array = array_task is not None and array_task != ""
    if not (parsed["sweep"] or in_array):
        return
    missing = [
        flag
        for flag, value in (
            ("--layout", parsed["layout"]),
            ("--population-seed", parsed["population_seed"]),
        )
        if value is None
    ]
    if not missing:
        return
    trigger = f"{ARRAY_TASK_ENV_VAR}={array_task}" if in_array else "--sweep"
    raise UnnamedWorldError(
        f"{trigger} means these numbers will be compared with another run's, "
        f"but {' and '.join(missing)} was not given, so the run stamp would "
        "choose the practice layout and the container field. Name the world "
        "explicitly, or drop --sweep if this run is not part of a set."
    )


__all__ = [
    "ARRAY_TASK_ENV_VAR",
    "_DEFAULT_ROUNDS",
    "_SIM_DIR",
    "UnnamedWorldError",
    "_CliArgsDict",
    "_parse_cli",
    "require_named_world",
]
