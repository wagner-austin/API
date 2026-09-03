"""Batches of matches, described as data rather than written as a script.

An experiment is a list of matches that differ in one argument. Writing that
list as a shell loop makes it unrepeatable: the arm that ran last week cannot be
re-run, because the loop that defined it was edited to define the next one. So a
sweep is a **file of jobs**, one line per match, and the harness's whole job is
to decide which of them still need playing and what argument list each one
becomes.

All of that is pure, and all of it lives here. Spawning the game, reading the
clock and touching the filesystem belong to the entry point, reached through
:mod:`rw_bot.harness._test_hooks`.

Two properties are deliberate and both come from the same decision -- that a
match's result is a file named after the job:

* **Resumable.** A sweep killed half way through is re-run by issuing the same
  command; jobs whose result file exists are skipped. Nothing tracks progress
  separately, so nothing can disagree about it.
* **Crash-isolated.** A match that dies takes its own result with it and no
  other. A batch is never a single unit of work.

Parallelism is bounded by memory rather than cores: a headless match holds about
430 MB and uses roughly one core. The reason a match needs a *cloned* game
directory at all is that the engine writes three fixed-name paths inside its own
directory -- ``preferences.ini``, ``saves/autosave.rwsave.tmp2`` and the mod
cache -- so two matches sharing one directory race on all three. Everything else
the launcher needs was already per-invocation ([[harness-nodisplay]]).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TypedDict

from rw_bot import RwBotError
from rw_bot.harness import play_match_cli
from rw_bot.harness.launch import FROZEN_CATALOGUE, FROZEN_TYPE_DUMP
from rw_bot.harness.match import MatchConfig
from rw_bot.validation import require_int, require_non_empty_str, require_positive_int

#: The module a sweep launches each of its matches by.
#:
#: Read off the module itself rather than written out, so the name a sweep
#: invokes cannot drift from the module that answers. The import is also what
#: wires the launcher into the graph the architecture test walks -- a
#: subprocess invocation is not an import edge, and a launcher reachable only
#: by string would read as an unwired module.
LAUNCHER_MODULE = play_match_cli.__name__

_FIELD_COUNT = "RW-SWEEP-001"
_NOT_A_NUMBER = "RW-SWEEP-002"
_NO_WORKERS = "RW-SWEEP-003"
_BAD_INDEX = "RW-SWEEP-004"
_NO_PORT = "RW-SWEEP-005"

#: Fields of a job line, in order, as they appear in a sweep file.
#:
#: The gameplay style is one field now, naming a doctrine file, where it used
#: to be four inline (goals, worker ceiling, mass, reserve) and grew one column
#: per question. The arm's identity moves with it: two jobs sharing a label
#: differ only in seed, and what the label *means* is recorded in a file that
#: outlives the sweep -- so the arm that ran last week can be re-run without
#: reconstructing its knobs from a job line ([[policy-loop]]).
JOB_FIELDS = ("label", "seed", "doctrine", "samples")

#: Width of the label column every report line is written to.
#:
#: :func:`~rw_bot.policy.match_report.format_report` writes each figure as a
#: lowercase label padded to this width followed by the value, and nothing else
#: the planner prints has that shape.
LABEL_WIDTH = 15


class SweepError(RwBotError):
    """A sweep file could not be read as a list of matches.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending line.
    """


class SweepJob(TypedDict):
    """One match to play.

    Attributes:
        label: Which arm this match belongs to. Two jobs sharing a label and
            differing in seed are repeats of one arm.
        seed: What the engine's generator is pinned to. Repeats of an arm differ
            here and nowhere else.
        doctrine: Path to the doctrine file naming the whole gameplay style --
            goals, worker ceiling, wave mass, reserve and the policy switches
            (:mod:`rw_bot.policy.doctrine`). What the knobs were is a file that
            outlives the sweep rather than a job line reconstructed afterwards.
        samples: Observations to play before stopping.
    """

    label: str
    seed: int
    doctrine: str
    samples: int


def decode_sweep_job(payload: Mapping[str, str | int | float | bool]) -> SweepJob:
    """Read one job from a flat payload.

    Args:
        payload: Field values by name.

    Returns:
        The job.

    Raises:
        DecodeError: ``RW-DECODE-001`` when a field is absent, ``RW-DECODE-002``
            when one carries the wrong type, ``RW-DECODE-003`` when a name is
            blank, ``RW-DECODE-004`` when a count is not positive.
    """
    return SweepJob(
        label=require_non_empty_str(payload, "label"),
        seed=require_int(payload, "seed"),
        doctrine=require_non_empty_str(payload, "doctrine"),
        samples=require_positive_int(payload, "samples"),
    )


def encode_sweep_job(job: SweepJob) -> dict[str, str | int]:
    """Write one job back to a flat payload.

    Args:
        job: The job.

    Returns:
        Field values by name, as :func:`decode_sweep_job` reads them.
    """
    return {
        "label": job["label"],
        "seed": job["seed"],
        "doctrine": job["doctrine"],
        "samples": job["samples"],
    }


def parse_job_line(line: str) -> SweepJob:
    """Read one pipe-separated job line.

    The format is positional and narrow on purpose. A sweep file is written by
    hand often enough that a missing field should be an error naming the line,
    not a default quietly changing what the arm means.

    Args:
        line: One line of a sweep file, without its newline.

    Returns:
        The job it describes.

    Raises:
        SweepError: ``RW-SWEEP-001`` when the line does not carry exactly the
            expected fields, ``RW-SWEEP-002`` when a numeric field is not a
            number.
        DecodeError: When a field is present but out of range.
    """
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != len(JOB_FIELDS):
        raise SweepError(
            _FIELD_COUNT,
            f"a job line carries {len(parts)} fields, expected {len(JOB_FIELDS)} "
            f"({'|'.join(JOB_FIELDS)}): {line!r}",
        )
    payload: dict[str, str | int | float | bool] = {"label": parts[0], "doctrine": parts[2]}
    numeric = (
        ("seed", parts[1]),
        ("samples", parts[3]),
    )
    for name, raw in numeric:
        try:
            payload[name] = int(raw)
        except ValueError as error:
            raise SweepError(
                _NOT_A_NUMBER, f"field {name!r} must be a whole number, got {raw!r}: {line!r}"
            ) from error
    return decode_sweep_job(payload)


def parse_jobs(lines: Sequence[str]) -> tuple[SweepJob, ...]:
    """Read a whole sweep file.

    Blank lines and ``#`` comments are skipped, so an arm can be commented out
    of a sweep without deleting the line that documents it.

    Args:
        lines: The file's lines, without newlines.

    Returns:
        Every job the file describes, in file order.

    Raises:
        SweepError: When any line is malformed.
        DecodeError: When any field is out of range.
    """
    return tuple(
        parse_job_line(line) for line in lines if line.strip() and not line.lstrip().startswith("#")
    )


def fresh_seeds(used: AbstractSet[int], count: int, start: int, stop: int) -> tuple[int, ...]:
    """Pick seeds no prior experiment has consumed, spread across a range.

    Untouched seeds are what laws six and nine trade in: selection on one
    dataset, confirmation on another, and a seed reused anywhere breaks
    that independence silently. The picker was written inline four times
    on 2026-09-02/03 before this lift; each copy globbed the sweep files,
    parsed the job lines, and spread the survivors by even stride --
    exactly what this does, once, tested.

    Args:
        used: Every seed any sweep file has ever named, from
            :func:`parse_jobs` over their lines.
        count: How many fresh seeds to pick.
        start: The range's inclusive lower bound; rounded up to odd,
            because every seed this project has ever fielded is odd and a
            mixed convention would make collisions harder to eyeball.
        stop: The range's exclusive upper bound.

    Returns:
        ``count`` odd seeds in ``[start, stop)``, none in ``used``, in
        ascending order, spread by even stride across the available pool.

    Raises:
        SweepError: ``RW-SWEEP-006`` when the range does not hold
            ``count`` unused odd seeds -- the range is wrong, not the
            experiment, and silently reusing a seed would corrupt every
            panel that trusts disjointness.
    """
    pool = [seed for seed in range(start | 1, stop, 2) if seed not in used]
    if len(pool) < count:
        raise SweepError(
            "RW-SWEEP-006",
            f"the range [{start}, {stop}) holds {len(pool)} unused odd seed(s), "
            f"{count} were asked for; widen the range rather than reuse a seed",
        )
    stride = len(pool) // count
    return tuple(pool[index * stride] for index in range(count))


def job_name(job: SweepJob) -> str:
    """Return the name a job's result is filed under.

    Arm and seed and nothing else, because those are exactly what distinguishes
    one match in a sweep from another. Two jobs that would collide here are the
    same match described twice.

    Args:
        job: The job.

    Returns:
        A filename stem.
    """
    return f"{job['label']}-s{job['seed']}"


def assigned(jobs: Sequence[SweepJob], index: int, workers: int) -> tuple[SweepJob, ...]:
    """Return the jobs one worker is responsible for.

    Round-robin rather than a shared queue. Every match takes about the same
    wall time, so this balances as well as a queue would, and it needs no lock
    between workers -- which is a whole class of failure the harness then does
    not own.

    Args:
        jobs: Every job to be played, in order.
        index: Which worker this is, from zero.
        workers: How many workers there are.

    Returns:
        The subset this worker plays, in order.

    Raises:
        SweepError: ``RW-SWEEP-003`` when there are no workers,
            ``RW-SWEEP-004`` when the index does not name one of them.
    """
    if workers <= 0:
        raise SweepError(_NO_WORKERS, f"a sweep needs at least one worker, got {workers}")
    if not 0 <= index < workers:
        raise SweepError(_BAD_INDEX, f"worker index {index} is not in range for {workers} workers")
    return tuple(job for position, job in enumerate(jobs) if position % workers == index)


def play_args(job: SweepJob, trace: str, tree: str = "") -> str:
    """Return the planner's positional argument list for one job.

    **Every match records a trace now, where sweeps used to pass ``-``.** The
    scorecard keeps about two dozen endpoint figures, and endpoints turned out
    to be actively misleading: a match reporting ``extractors 0 -> 0`` had in
    fact held a peak of **14** and led the strongest rival on total worth at
    the midpoint before collapsing. None of that is recoverable afterwards, and
    re-running to get it is a match that no longer reproduces
    ([[policy-trace]]).

    It is also what makes the results tabular rather than prose. One row per
    sample per match, against one text scorecard per match parsed by shape.

    **The doctrine is read from the frozen tree, not the working one.** The
    first snapshot batch proved why within the hour: matches imported frozen
    code but read the job line's doctrine path from the repository root, a
    doctrine field was added to the working tree mid-batch, and the frozen
    parser refused the new field on sixteen straight matches -- the freeze
    exists precisely so a mid-batch edit cannot reach a running experiment,
    and a doctrine file is as much the experiment as the code is
    (log: 2026-07-29).

    **The trace path is given, not composed here.** It used to be built from
    the batch name against a repository-relative root, which is a path that
    only means what it says when the process starts in the repository. A
    compute node's does not, so the trace went somewhere nothing would look
    -- and for a replication panel the trace IS the measurement. Composition
    belongs to :mod:`rw_bot.harness.results_layout`, which knows which root
    this run is filing under.

    Args:
        job: The job.
        trace: Where this match writes its per-sample trace.
        tree: The batch's frozen snapshot, or empty to read the job's doctrine
            path as written.

    Returns:
        The value of ``PLAY_ARGS``: samples, doctrine path, trace path.
    """
    doctrine = f"{tree}/{job['doctrine']}" if tree else job["doctrine"]
    return f"{job['samples']} {doctrine} {trace}"


def make_argv(
    interpreter: str,
    job: SweepJob,
    game_dir: str,
    lockstep: int,
    play_log: str,
    trace: str,
    port: int,
    display: int,
    match: MatchConfig | None = None,
    tree: str = "",
    pin_delta: int = 0,
    fast_forward: int = 0,
) -> tuple[str, ...]:
    """Return the command that plays one match.

    **The command is this package's own launcher, not ``make``.** It used to
    be a ``make play`` line, which put the whole launch behind a PowerShell
    recipe and a PowerShell script -- neither of which can start a match on a
    Linux compute node. The composition now lives in
    :mod:`rw_bot.harness.launch` and this names the module that runs it, so
    one description of a launch serves both platforms
    ([[harness-parallel-matches]]).

    **The interpreter is the harness's own.** The recipe ran the planner under
    ``poetry run python``; a batch inside a container image has no poetry in
    it, and the environment the planner must run in is the one the sweep is
    already running in. Passing it removes a second answer to "which Python".

    Lockstep is passed on every job rather than defaulted by the recipe.
    Free-running, the sample exchange is paced by a wall clock, so parallel
    matches under CPU contention sample at different game-times -- the act of
    running a sweep in parallel would change its results
    ([[policy-determinism]]).

    **The match is a property of the batch, not of the job line.** Which map is
    played decides how many opponents there are -- the engine caps teams by the
    map's own count -- so an arm that varies the opponent is an arm that varies
    the map, and both belong to every job in the batch alike. Leaving it out
    plays the engine's own hardcoded ten-player free-for-all, which is what
    every measurement before this was taken in.

    **So is the tree.** A match imports the source tree at launch, so an edit
    landed mid-batch used to mean later matches ran different code from
    earlier ones -- the whole working tree was frozen for the length of every
    sweep. The batch freezes its own copy instead and every job points at it
    ([[policy-loop]]).

    Args:
        interpreter: The Python the launcher and the planner run under.
        job: The job.
        game_dir: The cloned game directory this worker owns.
        lockstep: Engine frames between samples.
        play_log: Where the engine's and the agent's own logs go.
        trace: Where this match writes its per-sample trace.

            Both are GIVEN rather than composed from the batch name, and that
            is not tidiness. They used to be built here against
            repository-relative roots while the runner created the directories
            against an absolute one, so on a compute node -- where the two
            differ -- the directory that existed and the directory written to
            were not the same. On this workstation they always agreed, because
            the process starts in the repository, which is exactly why nothing
            caught it.
        port: The channel port this match's clone leases. Required rather than
            drawn: two concurrent random draws collided the first time eight
            matches launched in one instant and both died on the bind
            (imp-creep12, 2026-08-08; :func:`~rw_bot.harness.clone.leased_port`).
        display: The X display this match's clone leases, or zero to use the
            machine's own. Leased for the same reason the port is: under
            ``-nodisplay`` the engine still opens a display, so on a headless
            node every concurrent match needs one to itself
            (:func:`~rw_bot.harness.clone.leased_display`).
        match: Which match to play, or None for the engine's own default.
        tree: The batch's frozen code snapshot, or empty to import the working
            tree -- the single-match entry points' behaviour.
        pin_delta: Constant frame delta in milliseconds, or zero to leave the
            engine on the wall clock. Zero is also what a tree frozen before
            the option existed requires -- its agent rejects the unknown key
            -- so the default stays off until those trees retire
            ([[policy-determinism]]).
        fast_forward: Wall-clock multiple to run the simulation at, or zero
            for realtime. N identical pinned steps per loop pass -- the same
            simulation N times as fast, certified bit-exact against realtime
            at 10 (log 2026-08-06). Zero for the same frozen-tree reason as
            the pin: an older agent rejects the unknown key.

    Returns:
        The argument vector, the interpreter first.

    Raises:
        SweepError: ``RW-SWEEP-005`` when the port is not a real lease.
    """
    if port <= 0:
        raise SweepError(
            _NO_PORT,
            f"a sweep job needs the channel port its clone leases, got {port}: two "
            "concurrent random draws collided the first time eight matches launched "
            "in one instant, and both died on the bind (imp-creep12, 2026-08-08)",
        )
    argv = [
        interpreter,
        "-m",
        LAUNCHER_MODULE,
        "--port",
        str(port),
        "--display",
        str(display),
        "--game-dir",
        game_dir,
        "--seed",
        str(job["seed"]),
        "--lockstep",
        str(lockstep),
        "--play-log",
        play_log,
        "--play-args",
        play_args(job, trace, tree),
    ]
    if tree:
        argv.extend(("--tree", tree))
        # The planner reads both of these by path at startup, and the
        # launcher's defaults for them are repository-relative. That is only
        # ever true where a repository is: on a compute node the process
        # starts in a home directory and the first member to reach the
        # planner died on FileNotFoundError for the catalogue, having already
        # patched the engine, seeded it and held the world at frame one. The
        # frozen tree carries both now, so a batch reads them from the same
        # snapshot it imports its code and its doctrine from.
        argv.extend(("--catalogue", f"{tree}/{FROZEN_CATALOGUE}"))
        argv.extend(("--type-dump", f"{tree}/{FROZEN_TYPE_DUMP}"))
    if pin_delta:
        argv.extend(("--pin-delta", str(pin_delta)))
    if fast_forward:
        argv.extend(("--fast-forward", str(fast_forward)))
    if match is not None:
        argv.extend(
            (
                "--map",
                match["map_path"],
                "--opponents",
                str(match["opponents"]),
                "--difficulty",
                str(match["difficulty"]),
            )
        )
    return tuple(argv)


def is_report_line(line: str) -> bool:
    """Report whether one printed line is a figure from the match report.

    **Recognised by shape rather than by a list of labels, and that is the
    whole point.** This used to hold a tuple of every label the report emits,
    which is knowledge duplicated from
    :func:`~rw_bot.policy.match_report.format_report` -- so adding a figure to
    the report silently dropped it from every sweep result. Two were added and
    two went missing, and the batch looked like it had simply measured nothing.

    A report line is a lowercase label padded to :data:`LABEL_WIDTH` followed by
    a value. Nothing else the planner prints matches: its plan commentary
    carries a colon inside the label column, its per-entry lines are indented,
    and the harness's own progress lines start with a bracket or an arrow.

    Args:
        line: One printed line.

    Returns:
        True when the line is a report figure.
    """
    if len(line) <= LABEL_WIDTH or line[LABEL_WIDTH] == " ":
        return False
    label = line[:LABEL_WIDTH]
    if not label[0].islower():
        return False
    return all(character.isalpha() or character == " " for character in label)


def scorecard(lines: Sequence[str]) -> tuple[str, ...]:
    """Return the report lines from a match's output.

    Args:
        lines: Everything the match printed.

    Returns:
        The scorecard lines, in order.
    """
    return tuple(line for line in lines if is_report_line(line))


def is_complete(card: Sequence[str]) -> bool:
    """Report whether a scorecard came from a match that actually finished.

    A match that crashed on boot prints its plan and nothing else, and filing
    that as a result would record a blank as though it were a measurement. The
    verdict line is the planner's last word, so its presence is the test.

    Args:
        card: The scorecard lines.

    Returns:
        True when the match reported a verdict.
    """
    return any(line.startswith("verdict") for line in card)


__all__ = [
    "JOB_FIELDS",
    "LABEL_WIDTH",
    "LAUNCHER_MODULE",
    "SweepError",
    "SweepJob",
    "assigned",
    "decode_sweep_job",
    "encode_sweep_job",
    "fresh_seeds",
    "is_complete",
    "is_report_line",
    "job_name",
    "make_argv",
    "parse_job_line",
    "parse_jobs",
    "play_args",
    "scorecard",
]
