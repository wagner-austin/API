"""Playing a batch of matches: the service that turns jobs into results.

The layer between the job model and the command line. :mod:`rw_bot.harness.sweep`
decides *what* each match is and :mod:`rw_bot.harness.clone` decides what a
worker's copy of the game must contain; this module is what actually plays them,
and it is the only part of the three that touches anything outside the process.

Every such operation is reached through :mod:`rw_bot.harness._test_hooks`, so
the production and test paths are identical in shape and a test drives the real
control flow against fakes rather than a rehearsal of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from rw_bot.harness import _test_hooks
from rw_bot.harness.clone import (
    REQUIRED_ENTRIES,
    VOLATILE_FILES,
    clone_name,
    entries_to_copy,
    verify,
)
from rw_bot.harness.match import MatchConfig
from rw_bot.harness.sweep import (
    SweepJob,
    assigned,
    is_complete,
    job_name,
    make_argv,
    scorecard,
    trace_path,
)
from rw_bot.validation import require_int, require_non_empty_str, require_positive_int


class SweepConfig(TypedDict):
    """How one batch of matches is to be played.

    Attributes:
        out_dir: Where results are filed, one file per match.
        workers: How many matches to play at once. Bounded by memory rather
            than cores: a headless match holds about 430 MB and uses roughly
            one core.
        lockstep: Engine frames between samples. Never zero here -- free
            running, the exchange is paced by a wall clock, so parallel matches
            under CPU contention would sample at different game-times and the
            act of running them in parallel would change their results
            ([[policy-determinism]]).
        clone_prefix: Leading part of every worker copy's directory name.
        source_game_dir: The pinned game directory copies are taken from.
        tree: Where the batch's frozen copy of the code lives, under
            ``out_dir``. Every match imports from it instead of the working
            tree, so the tree is editable the moment the batch starts and the
            batch records exactly what its matches ran
            (:func:`prepare_tree`).
        match: Which match every job in the batch plays, or None for the
            engine's own default. Batch-level rather than per-job because the
            map decides the opponent count -- the engine caps teams by the
            map's own -- so an arm that varies the opponent varies the map, and
            that is what a batch *is* ([[policy-determinism]]).
        pin_delta: Constant frame delta in milliseconds, or zero to leave the
            engine on the wall clock. Batch-level like the match: a pinned and
            an unpinned run of one seed are different simulations, so mixing
            them inside a batch would be an uncontrolled arm
            ([[policy-determinism]]).
        fast_forward: Wall-clock multiple every match runs at, or zero for
            realtime. Batch-level because it is certified bit-exact -- a fast
            batch IS the realtime batch, only sooner (log 2026-08-06) -- and a
            knob that varied per job would suggest otherwise.
    """

    out_dir: str
    workers: int
    lockstep: int
    clone_prefix: str
    source_game_dir: str
    tree: str
    pin_delta: int
    fast_forward: int
    match: MatchConfig | None


class SweepOutcome(TypedDict):
    """What a batch achieved.

    Attributes:
        total: Matches the job file describes.
        already: Matches that had results before this run started.
        played: Matches this run finished.
    """

    total: int
    already: int
    played: int


def decode_sweep_config(
    payload: Mapping[str, str | int | float | bool], match: MatchConfig | None = None
) -> SweepConfig:
    """Read a batch configuration from a flat payload.

    Args:
        payload: Field values by name.
        match: Which match every job plays, already decoded because it has its
            own validation and its own error code.

    Returns:
        The configuration.

    Raises:
        DecodeError: ``RW-DECODE-001`` when a field is absent, ``RW-DECODE-002``
            when one carries the wrong type, ``RW-DECODE-003`` when a name is
            blank, ``RW-DECODE-004`` when a count is not positive.
    """
    return SweepConfig(
        out_dir=require_non_empty_str(payload, "out_dir"),
        workers=require_positive_int(payload, "workers"),
        lockstep=require_positive_int(payload, "lockstep"),
        clone_prefix=require_non_empty_str(payload, "clone_prefix"),
        source_game_dir=require_non_empty_str(payload, "source_game_dir"),
        tree=require_non_empty_str(payload, "tree"),
        pin_delta=require_int(payload, "pin_delta"),
        fast_forward=require_int(payload, "fast_forward"),
        match=match,
    )


def encode_sweep_config(config: SweepConfig) -> dict[str, str | int]:
    """Write a batch configuration back to a flat payload.

    Args:
        config: The configuration.

    Returns:
        Field values by name, as :func:`decode_sweep_config` reads them.
    """
    return {
        "out_dir": config["out_dir"],
        "workers": config["workers"],
        "lockstep": config["lockstep"],
        "clone_prefix": config["clone_prefix"],
        "source_game_dir": config["source_game_dir"],
        "tree": config["tree"],
        "pin_delta": config["pin_delta"],
        "fast_forward": config["fast_forward"],
    }


def decode_sweep_outcome(payload: Mapping[str, str | int | float | bool]) -> SweepOutcome:
    """Read a batch result from a flat payload.

    Args:
        payload: Field values by name.

    Returns:
        The outcome.

    Raises:
        DecodeError: ``RW-DECODE-001`` when a field is absent, ``RW-DECODE-002``
            when one is not an ``int``.
    """
    return SweepOutcome(
        total=require_positive_int(payload, "total"),
        already=require_positive_int(payload, "already"),
        played=require_positive_int(payload, "played"),
    )


def encode_sweep_outcome(outcome: SweepOutcome) -> dict[str, int]:
    """Write a batch result back to a flat payload.

    Args:
        outcome: The outcome.

    Returns:
        Field values by name.
    """
    return {
        "total": outcome["total"],
        "already": outcome["already"],
        "played": outcome["played"],
    }


#: What a batch freezes: everything a match imports or reads at launch that is
#: also under active development. The archived registry dumps (catalogue, type
#: flags) are deliberately absent -- they are generated artifacts pinned to the
#: game build, not code that changes between batches.
TREE_SOURCES = ("scripts", "doctrines", "agent/build/rw-agent.jar")

#: The frozen tree's directory name, under the batch's results directory.
TREE_DIR = ".tree"

#: Written into the tree last, so its presence certifies a complete freeze.
TREE_MARKER = ".complete"


def prepare_tree(config: SweepConfig) -> None:
    """Freeze the code the batch's matches will import, once, at launch.

    A match imports the source tree at launch, so before this existed an edit
    landed mid-batch meant later matches ran different code from earlier ones
    -- an arm's twelve seeds were only one experiment if nobody touched the
    tree for the batch's whole runtime, and the working tree was frozen for
    hours at a stretch. The batch copies what its matches need into its own
    results directory instead: the tree is editable the moment the sweep
    starts, and the batch carries a record of exactly what it ran.

    **An existing snapshot is reused, never refreshed.** That is what makes a
    resumed batch a continuation rather than a new experiment: the matches
    played after the interruption import the same frozen code as the ones
    played before it, whatever has happened to the working tree in between.

    Args:
        config: How the batch is being played.

    Raises:
        OSError: When a copy fails.
    """
    tree = Path(config["tree"])
    # Judged by the marker, not by the directory: a directory can survive a
    # partial delete -- Windows file locks kept one alive with its doctrines
    # gone -- and reusing a gutted tree fails ten matches at once with the
    # freeze reporting success (log: 2026-07-31). The marker is written last,
    # so its presence certifies every copy before it finished.
    if _test_hooks.path_exists(tree / TREE_MARKER):
        _test_hooks.write_line(f"[sweep] reusing the frozen tree at {config['tree']}")
        return
    _test_hooks.make_dirs(tree / "src")
    _test_hooks.copy_entry(Path("src/rw_bot"), tree / "src")
    for entry in TREE_SOURCES:
        _test_hooks.copy_entry(Path(entry), tree)
    _test_hooks.write_text_lines(tree / TREE_MARKER, ("frozen",))
    _test_hooks.write_line(f"[sweep] tree frozen at {config['tree']}")


def prepare_clone(index: int, config: SweepConfig) -> str:
    """Give one worker its own copy of the game directory.

    An existing copy is re-verified rather than rebuilt. A sweep is normally run
    several times while an experiment is refined, and re-copying 0.44 GB on each
    of those buys nothing.

    Args:
        index: Which worker this is, from zero.
        config: How the batch is being played.

    Returns:
        The clone's directory name.

    Raises:
        CloneError: ``RW-CLONE-001`` when the finished copy is missing anything
            a match needs, ``RW-CLONE-002`` when the index is negative.
        OSError: When the copy cannot be made.
    """
    name = clone_name(config["clone_prefix"], index)
    destination = Path(name)
    source = Path(config["source_game_dir"])
    if not _test_hooks.path_exists(destination):
        _test_hooks.make_dirs(destination)
        for entry in entries_to_copy(_test_hooks.list_names(source)):
            _test_hooks.copy_entry(source / entry, destination)
    verify(
        name,
        [needed for needed in REQUIRED_ENTRIES if _test_hooks.path_exists(destination / needed)],
    )
    return name


def reset_volatile_files(game_dir: str, config: SweepConfig) -> None:
    """Put a clone's rewritable settings back to the pinned copy's.

    Done before every match rather than once per clone, because the game
    rewrites these on each boot: without it, the second match a worker plays
    starts from the first one's leavings, and two workers that have played
    different numbers of matches start from different settings.

    The observed drift is a main-menu counter and cannot reach a headless
    simulation. Resetting is not about that key -- it is about the guarantee
    being "the state does not differ" rather than "the state that differs is
    currently harmless" ([[policy-determinism]]).

    Args:
        game_dir: The cloned game directory to reset.
        config: How the batch is being played.

    Raises:
        OSError: When a file cannot be copied.
    """
    source = Path(config["source_game_dir"])
    for name in VOLATILE_FILES:
        _test_hooks.copy_entry(source / name, Path(game_dir))


def play_job(job: SweepJob, game_dir: str, config: SweepConfig) -> bool:
    """Play one match and file its result.

    A match that printed no verdict did not finish. Its transcript is kept
    beside the results as ``.partial`` so the failure can be read, and no result
    file is written -- which leaves the job outstanding for the next run rather
    than filing a blank as though it were a measurement.

    Args:
        job: The match to play.
        game_dir: The cloned game directory this worker owns.
        config: How the batch is being played.

    Returns:
        True when the match finished and its result was filed.

    Raises:
        OSError: When the game cannot be started or the result cannot be
            written.
    """
    name = job_name(job)
    out_dir = Path(config["out_dir"])
    # Traces are namespaced by batch, and the batch is named by where its
    # results are filed -- one name for both artifacts, so a result and its
    # trace can never belong to different experiments.
    batch = out_dir.name
    # The planner opens the trace for writing and does not create its parent,
    # so the directory has to exist before the match starts rather than after
    # it has run for twenty minutes and failed to file anything.
    _test_hooks.make_dirs(Path(trace_path(job, batch)).parent)
    # The game redirects its stdout into the log path's directory at launch
    # and does not create it.
    _test_hooks.make_dirs(out_dir / "logs")
    reset_volatile_files(game_dir, config)
    _test_hooks.write_line(f"[sweep] {game_dir} playing {name}")
    _, output = _test_hooks.run_capture(
        make_argv(
            job,
            game_dir,
            config["lockstep"],
            batch,
            config["match"],
            config["tree"],
            config["pin_delta"],
            config["fast_forward"],
        )
    )
    card = scorecard(output)
    if not is_complete(card):
        _test_hooks.write_text_lines(out_dir / f"{name}.partial", (f"### {name} FAILED", *output))
        _test_hooks.write_line(f"[sweep] {name} FAILED, transcript kept as {name}.partial")
        return False
    _test_hooks.write_text_lines(out_dir / f"{name}.txt", (f"### {name}", *card))
    _test_hooks.write_line(f"[sweep] {name} done")
    return True


def run_worker(jobs: Sequence[SweepJob], index: int, config: SweepConfig) -> int:
    """Play every match assigned to one worker, one at a time.

    The worker takes its copy of the game only once it knows it has matches to
    play, so a batch smaller than the worker pool does not pay for copies it
    will not use.

    Args:
        jobs: Every match still outstanding, in order.
        index: Which worker this is, from zero.
        config: How the batch is being played.

    Returns:
        How many of this worker's matches finished.

    Raises:
        CloneError: When this worker's copy of the game is unusable.
        SweepError: ``RW-SWEEP-003`` or ``RW-SWEEP-004`` when the partition is
            not well formed.
        OSError: When a match cannot be started or its result written.
    """
    mine = assigned(jobs, index, config["workers"])
    if not mine:
        return 0
    game_dir = prepare_clone(index, config)
    return sum(1 for job in mine if play_job(job, game_dir, config))


def outstanding(jobs: Sequence[SweepJob], out_dir: Path) -> tuple[SweepJob, ...]:
    """Return the matches that do not have a result yet.

    Resumability comes from nothing more than this. A batch killed part way
    through is continued by issuing the same command, and because the result
    files *are* the progress record there is nothing that can disagree with it.

    Args:
        jobs: Every match the job file describes, in order.
        out_dir: Where results are filed.

    Returns:
        The matches still to play, in order.

    Raises:
        OSError: When the results directory cannot be read.
    """
    return tuple(
        job for job in jobs if not _test_hooks.path_exists(out_dir / f"{job_name(job)}.txt")
    )


__all__ = [
    "TREE_DIR",
    "TREE_MARKER",
    "TREE_SOURCES",
    "SweepConfig",
    "SweepOutcome",
    "decode_sweep_config",
    "decode_sweep_outcome",
    "encode_sweep_config",
    "encode_sweep_outcome",
    "outstanding",
    "play_job",
    "prepare_clone",
    "prepare_tree",
    "reset_volatile_files",
    "run_worker",
]
