"""Play exactly one match of a batch and file its scorecard.

The cluster's unit of work. On this workstation a batch is one process playing
many matches across many workers; on a cluster each match is its own scheduled
job, because a member of a campaign is done when ITS artifact exists and a job
that declared it completed -- and an artifact only one member writes is a
per-match scorecard, not a shared results directory
(:mod:`rw_bot.harness.campaign`).

It is also the right size. A match is about twenty minutes, comfortably under
hpc3's sixty-minute preemption threshold, so it needs no checkpointing: the
scorecard IS the checkpoint and a preempted match costs one match.

IN THE INSTALLED PACKAGE, NOT IN ``scripts/``. This ran as ``scripts.match``
until 2026-08-29 and could never have worked: ``pyproject`` packages
``rw_bot`` from ``src`` and nothing else, so ``scripts/`` is absent from the
wheel and absent from the image built out of it. The failure would have been
``No module named scripts`` on a compute node, for every member, after the
tree had already been staged. The rest of the monorepo already ran its cluster
payloads this way -- ``mi`` submits ``python -m
model_trainer.cli.gemm_benchmark`` -- so this is that pattern rather than a
new one.

EVERY PATH IT IS GIVEN IS ABSOLUTE, and the two entry points differ in exactly
that. ``sbatch`` sets no working directory, so a relative game directory or
result path resolves against whatever the submitting shell happened to have.
The caller composes them (:mod:`rw_bot.harness.results_layout`); this checks
that the one it was told to write is its own.

Run as ``python -m rw_bot.harness.campaign_match --jobs <file> --batch <name>
--label <arm> --seed <n> --lockstep <frames> --game <dir> --tree <dir>
--traces <dir> --map <path> --difficulty <n> --clones <dir> --result <path>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from platform_core.cli_args import parse_single_flags, require_flag

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks
from rw_bot.harness.clone import CLONE_PREFIX
from rw_bot.harness.match import decode_match_config
from rw_bot.harness.results_layout import declares_result_for, result_path
from rw_bot.harness.runner import check_frozen_tree, decode_sweep_config, play_job, prepare_clone
from rw_bot.harness.sweep import SweepJob, parse_jobs

_NO_SUCH_JOB = "RW-MATCH-001"
_RESULT_DISAGREES = "RW-MATCH-002"

#: Flags naming the one match to play, the tree to play it from, and where its
#: result belongs.
REQUIRED_FLAGS = (
    "--jobs",
    "--batch",
    "--label",
    "--seed",
    "--lockstep",
    "--fast-forward",
    "--game",
    "--tree",
    "--traces",
    "--map",
    "--difficulty",
    "--clones",
    "--result",
)

#: Opponents asked for. One, because the goal is to beat one -- and because
#: the engine caps the count by the map's own team count anyway, so a
#: two-player map gives one whatever is asked.
DUEL_OPPONENTS = 1

#: This entry point plays one match, so one worker is not a tuning choice.
SINGLE_WORKER = 1


#: Milliseconds of simulation every tick carries, whatever the container
#: measured.
#:
#: Three because that is what the measurement says: the container's average
#: is 3.33ms, so the pace shift is under ten percent, and the agent refuses
#: anything above :data:`AgentOptionsParser.MAX_PIN_DELTA_MS` (17) as a
#: value no real frame takes (wiki log, the kill-switch entry).
PINNED_DELTA_MS = 3

#: The match played, its scorecard filed.
EXIT_OK = 0

#: The match ran but printed no verdict, so no result was filed. The campaign
#: reads that as still-missing and submits it again, which is correct: a blank
#: filed as though it were a measurement is the outcome to avoid.
EXIT_INCOMPLETE = 1


class MatchCommandError(RwBotError):
    """One match could not be played as the command line described it.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what disagreed.
    """


def select_member(jobs: Sequence[SweepJob], label: str, seed: int) -> tuple[int, SweepJob]:
    """Return which member of the batch this is, and the job it plays.

    THE POSITION IS THE LEASE. A sweep on this workstation is one process
    whose concurrent matches are its workers, so the worker number leases the
    clone directory, the channel port and the X display. A campaign's
    concurrent matches are its MEMBERS, on nodes that share a filesystem and
    can share a node -- so the member's position in the job file is the same
    kind of exclusive number, and is used as one.

    Every member computing it from the same job file is what makes it a
    lease rather than a guess: no two members of a batch can read the same
    position out of one file, and nothing has to be passed between them.

    Without it every member leased ordinal 1. All twenty-four aimed at one
    ``.game-w1``, resolved against the directory they were SUBMITTED from --
    ``sbatch`` sets no working directory and ``hpc3`` submits from one script
    directory per project. The first began copying 307 MB into it and the
    second, nine seconds later on another node, saw the directory already
    there, skipped the copy and died listing a maps directory that did not
    exist yet (jobs 55663569/55663571, 2026-08-30). Two members that had
    landed on ONE node would also have bound one port and started one X
    display twice.

    Args:
        jobs: Every match the batch's job file describes.
        label: The arm to play.
        seed: The seed to play it at.

    Returns:
        The job's position in the batch, from zero, and the job.

    Raises:
        MatchCommandError: ``RW-MATCH-001`` when the batch has no such job, or
            more than one. A member that plays nothing would report success
            having run no match, and the campaign would then wait forever for
            an artifact nothing was ever going to write.
    """
    found = [
        (position, job)
        for position, job in enumerate(jobs)
        if job["label"] == label and job["seed"] == seed
    ]
    if len(found) != 1:
        raise MatchCommandError(
            _NO_SUCH_JOB,
            f"the batch describes {len(found)} job(s) for arm {label!r} at seed {seed}, "
            "expected exactly one",
        )
    return found[0]


def check_result_agrees(declared: str, batch: str, job: SweepJob) -> None:
    """Raise unless the declared result path is the one this match writes.

    The path is on the command line because a campaign member's artifact must
    be a path its OWN command mentions -- that is what makes the ledger's
    index checkable. Having it there twice would be two places to edit, so
    this refuses a disagreement rather than letting the run write one path
    while the ledger publishes another.

    Compared as a suffix rather than as an equality: the path arrives absolute
    and this process does not know the cluster root or the project that
    prefixed it. Everything that identifies the match -- the batch, the arm,
    the seed -- is below that prefix and is compared.

    Args:
        declared: The path the command line named.
        batch: The sweep this job belongs to.
        job: The job being played.

    Raises:
        MatchCommandError: ``RW-MATCH-002`` when the two disagree.
    """
    if not declares_result_for(declared, batch, job):
        raise MatchCommandError(
            _RESULT_DISAGREES,
            f"--result says {declared!r} but this match writes {result_path(batch, job)!r}: "
            "the ledger would publish a path the run never wrote to",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Play one match of a batch.

    Args:
        argv: Argument list excluding the program name. ``None`` reads the
            process arguments.

    Returns:
        :data:`EXIT_OK` when the scorecard was filed, :data:`EXIT_INCOMPLETE`
        when the match printed no verdict.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or
            required and absent.
        MatchCommandError: ``RW-MATCH-001`` when no single job matches,
            ``RW-MATCH-002`` when the declared result path disagrees.
        SweepError: When the job file is malformed, or ``RW-SWEEP-006``
            when the staged tree does not carry what a match reads out of it.
        CloneError: When this node's copy of the game is unusable.
        OSError: When the job file cannot be read or the result written.
    """
    tokens = list(argv) if argv is not None else _test_hooks.read_argv()
    parsed = parse_single_flags(tokens, REQUIRED_FLAGS)
    for flag in REQUIRED_FLAGS:
        require_flag(parsed, flag)

    batch = parsed["--batch"]
    jobs = parse_jobs(_test_hooks.read_text_lines(Path(parsed["--jobs"])))
    lease, job = select_member(jobs, parsed["--label"], int(parsed["--seed"]))
    check_result_agrees(parsed["--result"], batch, job)

    # Derived from the artifact rather than composed again: the file this
    # match must write is the one the ledger publishes, so its directory is
    # whatever holds that file and cannot be somewhere else.
    #
    # Split as a POSIX path rather than with `Path`, which resolves to the
    # flavour of the interpreter that is RUNNING. That is Linux here in
    # production and Windows in the suite, and the difference is not cosmetic:
    # a directory spelled with backslashes goes into the batch's own config
    # and out again into every path composed from it.
    out_dir = str(PurePosixPath(parsed["--result"]).parent)
    _test_hooks.make_dirs(Path(out_dir))
    # Required, not optional. Which map is played decides how many opponents
    # there are, so the map IS the experiment -- and a member that fell back
    # to the engine's own ten-player free-for-all would run a different
    # simulation from every batch this project has ever measured, with
    # nothing in the document to say which one it was.
    match = decode_match_config(
        {
            "map_path": parsed["--map"],
            "opponents": DUEL_OPPONENTS,
            "difficulty": int(parsed["--difficulty"]),
        }
    )
    config = decode_sweep_config(
        {
            "out_dir": out_dir,
            # Given rather than derived from the result path. A trace is the
            # match's per-sample record and the only thing a replication panel
            # measures; left to resolve against the process's own working
            # directory it would land under a home directory on the node, with
            # nothing to say so.
            "traces": parsed["--traces"],
            "workers": SINGLE_WORKER,
            "lockstep": int(parsed["--lockstep"]),
            # Rooted where the caller says, not where the process happens to
            # have started. A clone name is relative and `sbatch` sets no
            # working directory, so an unrooted prefix aimed every member of
            # the batch at one directory in the project's script directory.
            "clone_prefix": f"{parsed['--clones']}/{CLONE_PREFIX}",
            "source_game_dir": parsed["--game"],
            # The STAGED tree, frozen before submission. A member does not
            # freeze its own: `prepare_tree` copies from repository-relative
            # paths and a node has no repository, so it would report success
            # having copied nothing -- and the agent jar cannot be rebuilt
            # there at all, the Linux depot shipping a JRE with no compiler.
            "tree": parsed["--tree"],
            # **Pinned, unlike the workstation default.** The engine gates
            # decisions on `accumulator -= delta; if (drained) { act }` all
            # through the tick -- the AI's cadence, a unit's effect spawner,
            # a sway target's convergence -- so an unpinned delta makes each
            # of them fire on a schedule the wall clock sets. `pinDeltaMs`
            # writes the engine's own `bu` override, which makes every tick
            # the same fixed quantum whatever the container measured.
            #
            # It is opt-in for one stated reason -- trees frozen before the
            # option existed reject the unknown key, and watch and host runs
            # want the sim glued to wall time (wiki log, the kill-switch
            # entry). Neither applies to a campaign member: the tree is
            # frozen at submission and nothing watches it. Six separate
            # invocations of seed 31337 are bit-identical under this regime,
            # world and all three draw-count streams alike; the workstation
            # default of 0 is what the cluster panel forked under.
            "pin_delta": PINNED_DELTA_MS,
            # Required and read from the command line, because pace is part
            # of the batch's declared regime and a member left to a default
            # would run one the campaign document never stated. Re-checked
            # under the pin with a controlled comparison (2026-08-31): on
            # the two seeds measured, 10x against realtime moved NOTHING in
            # the world -- a divergence first blamed on pace turned out to
            # be the INVOCATION CONTEXT, seed 8128 forking at sample 97
            # between a solo boot and an 8-way-contended one at identical
            # pace while 31337 held, which is the pre-liveness boot-pace
            # seam and not this flag. The flag stays required anyway: two
            # seeds is evidence of equivalence, not a certification, and
            # the document stating the pace is what lets anyone check.
            "fast_forward": int(parsed["--fast-forward"]),
        },
        match,
    )
    # Checked, not built. The marker alone would not do: it certifies that
    # every copy before it finished and says nothing about a source that was
    # absent when the freeze ran.
    check_frozen_tree(Path(config["tree"]))
    # Cloned at this member's own lease rather than at worker zero's. The
    # partitioning `run_worker` does is what a many-worker sweep needs and
    # this is not one: a member plays the single job it was named, and the
    # only thing it needs from an index is an ordinal nothing else holds.
    played = play_job(job, prepare_clone(lease, config), config)
    return EXIT_OK if played else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "EXIT_INCOMPLETE",
    "EXIT_OK",
    "PINNED_DELTA_MS",
    "REQUIRED_FLAGS",
    "SINGLE_WORKER",
    "MatchCommandError",
    "check_result_agrees",
    "main",
    "select_member",
]
