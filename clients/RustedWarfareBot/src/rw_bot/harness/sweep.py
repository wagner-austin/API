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
from typing import TypedDict

from rw_bot import RwBotError
from rw_bot.harness.match import MatchConfig
from rw_bot.validation import require_int, require_non_empty_str, require_positive_int

_FIELD_COUNT = "RW-SWEEP-001"
_NOT_A_NUMBER = "RW-SWEEP-002"
_NO_WORKERS = "RW-SWEEP-003"
_BAD_INDEX = "RW-SWEEP-004"

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


#: Where a match's per-sample record is written, relative to the repository.
TRACE_ROOT = "runs/traces"


def trace_path(job: SweepJob, batch: str) -> str:
    """Return where one job's per-sample trace is written.

    Namespaced by batch, because a job's own name is only unique within one.
    Two sweeps that shared an arm label used to overwrite each other's traces
    -- re-running an arm for a new A/B silently destroyed the record the old
    batch's findings were read from, and only run-to-run determinism kept
    that from mattering ([[policy-trace]]).

    Args:
        job: The job.
        batch: The sweep this job belongs to, as the results directory names
            it.

    Returns:
        A path under :data:`TRACE_ROOT`, named after the batch and the job.
    """
    return f"{TRACE_ROOT}/{batch}/{job_name(job)}.ndjson"


def play_args(job: SweepJob, batch: str, tree: str = "") -> str:
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

    Args:
        job: The job.
        batch: The sweep this job belongs to, for the trace namespace.
        tree: The batch's frozen snapshot, or empty to read the job's doctrine
            path as written.

    Returns:
        The value of ``PLAY_ARGS``: samples, doctrine path, trace path.
    """
    doctrine = f"{tree}/{job['doctrine']}" if tree else job["doctrine"]
    return f"{job['samples']} {doctrine} {trace_path(job, batch)}"


def make_argv(
    job: SweepJob,
    game_dir: str,
    lockstep: int,
    batch: str,
    match: MatchConfig | None = None,
    tree: str = "",
    pin_delta: int = 0,
    fast_forward: int = 0,
) -> tuple[str, ...]:
    """Return the command that plays one match.

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
        job: The job.
        game_dir: The cloned game directory this worker owns.
        lockstep: Engine frames between samples.
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
        The argument vector, program first.
    """
    argv = [
        "make",
        "play",
        f"GAME_DIR={game_dir}",
        f"PLAY_SEED={job['seed']}",
        f"PLAY_SAMPLES={job['samples']}",
        f"PLAY_LOCKSTEP={lockstep}",
        # Into the batch directory, not the shared runs/ floor: engine and
        # agent logs are the deep-debug layer -- one of them named the
        # placeholder bug outright -- and the shared floor overwrote them
        # across batches and replays (log: 2026-07-31).
        f"PLAY_LOG=runs/sweeps/{batch}/logs/{job_name(job)}.log",
        f"PLAY_ARGS={play_args(job, batch, tree)}",
    ]
    if tree:
        argv.append(f"PLAY_TREE={tree}")
    if pin_delta:
        argv.append(f"PLAY_PINDELTA={pin_delta}")
    if fast_forward:
        argv.append(f"PLAY_FASTFORWARD={fast_forward}")
    if match is not None:
        argv.extend(
            [
                f"PLAY_MAP={match['map_path']}",
                f"PLAY_OPPONENTS={match['opponents']}",
                f"PLAY_DIFFICULTY={match['difficulty']}",
            ]
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
    "SweepError",
    "SweepJob",
    "assigned",
    "decode_sweep_job",
    "encode_sweep_job",
    "is_complete",
    "is_report_line",
    "job_name",
    "make_argv",
    "parse_job_line",
    "parse_jobs",
    "play_args",
    "scorecard",
]
