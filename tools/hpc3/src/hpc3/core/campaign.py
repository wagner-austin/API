"""Converging a set of runs on a declared end state, however many tries it takes.

WHAT WAS MISSING, and what it cost on 2026-08-28. The package had two shapes
for many jobs and neither is a session. A sweep is one template run several
ways CONCURRENTLY, submitted once. A chain is several jobs run one after
another. Neither models "this experiment is seven checkpoints and is finished
when seven checkpoints exist" -- so when ``free-gpu`` preempted five of seven
members inside an hour, nothing could say which five, and the answer lived in
an operator's head.

What followed is the evidence: a hand-written four-member resume sweep, then a
separate one-member document for ``uz``, then another when that was preempted
too, then another after an image rebuild. Four documents describing one
experiment, each a transcription of a queue state that had already changed,
and at one point two of them were live at once writing the same checkpoint.

A CAMPAIGN IS THE SAME DOCUMENT, RUN AGAIN. It takes the sweep document that
already exists -- no new shape to learn, and every committed sweep is already
one -- and instead of submitting all members, it computes what is missing:

* **done** -- the member's artifact exists on the cluster AND a job that
  declared it reached ``COMPLETED``. Both halves are load-bearing, see below.
* **in flight** -- a live job is already writing it. Nothing to do, and
  submitting would be the ``uz_best.pt`` race (:mod:`hpc3.core.inflight`).
* **missing** -- neither. Submit it.

So re-running after a preemption wave IS the resume, and it is idempotent: run
it twice in a row and the second run submits nothing. There is no state to
keep between runs because the cluster holds it -- the artifacts that exist and
the jobs that are live are both facts you can ask for, and neither can go
stale the way a transcription does.

EXISTENCE IS NOT COMPLETION, and assuming it was is the first thing this
command got wrong. ``turkic-lstm.bases-kk`` was preempted at 1273 seconds,
having written ``kk_best.pt`` -- because a training loop writes its best
checkpoint whenever validation improves, not at the end. The file existed and
the run had not finished, and the campaign called it done: it would have
stopped resubmitting a member that was silently under-trained, and reported
the experiment complete. A checkpoint is a progress marker that happens to
live at the artifact's path.

So a member is done when the artifact exists AND some job that declared it
reached ``COMPLETED``. The states are asked of the cluster rather than read
from the closure file alone, because closures are written by ``hpc3-triage``
and a campaign that silently depended on someone having run another command
would resubmit finished work whenever they had not.

EVERY MEMBER MUST DECLARE AN ARTIFACT, and this is the one thing a campaign
refuses. Done is defined in terms of the artifact; a member that writes no
file of its own has no done, so every run would resubmit it forever. That is
not a hypothetical -- every ``cleargbm`` sweep member runs ``--no-save-model``
and declares ``null``, correctly. Those sweeps are perfectly good sweeps and
cannot be campaigns, and being told so once beats an infinite loop that looks
like enthusiasm.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence

from platform_core.errors import AppError, Hpc3ErrorCode
from typing_extensions import TypedDict

from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import qualified_name
from hpc3.contracts.ledger import LedgerEntry

FINISHED_STATE = "COMPLETED"
"""The only terminal state that means the artifact is the finished thing."""

_PRESENT = "PRESENT|"
_ABSENT = "ABSENT|"


class CampaignPlan(TypedDict):
    """What a campaign run would do, before it does any of it.

    Attributes:
        done: Qualified names whose artifact already exists, in declaration
            order.
        in_flight: Qualified name to the live job already writing that
            member's artifact.
        missing: Specs to submit -- the gap, and nothing else.
    """

    done: list[str]
    in_flight: dict[str, str]
    missing: list[JobSpec]


def require_every_member_declares_an_artifact(specs: Sequence[JobSpec]) -> list[str]:
    """Refuse a campaign whose progress cannot be measured.

    Args:
        specs: The expanded members.

    Returns:
        Each member's artifact, in declaration order.

    Raises:
        AppError: With ``CAMPAIGN_MEMBER_HAS_NO_ARTIFACT`` if any member
            declares none. Refused rather than skipped: a member with no
            artifact is never done, so a campaign carrying one would resubmit
            it on every run forever, and an infinite loop that reports
            progress every time is worse than a refusal that happens once.
    """
    artifacts: list[str] = []
    for spec in specs:
        artifact = spec["artifact"]
        if artifact is None:
            label = qualified_name(spec["project"], spec["name"])
            raise AppError(
                Hpc3ErrorCode.CAMPAIGN_MEMBER_HAS_NO_ARTIFACT,
                f"{label} declares no artifact, so nothing can say whether it has "
                "finished and every run of this campaign would submit it again. A "
                "sweep whose members write no file of their own is a sweep, not a "
                "campaign; submit it with hpc3-sweep.",
            )
        artifacts.append(artifact)
    return artifacts


#: Longest artifact list one existence command may carry. The probe used to
#: pack every member into one shell line, and a 136-member search round
#: built a ~10 KB command -- past cmd.exe's 8191-character argument limit on
#: the Windows submitter, so the command reached bash TRUNCATED mid-loop and
#: died on "unexpected end of file" (vhsearch2-r0, 2026-09-02). Sixty paths
#: at this project's path lengths stay comfortably under half the limit; the
#: round-trip count this was avoiding stays bounded at members/60 rather
#: than members.
EXISTENCE_CHUNK = 60


def existence_commands(artifacts: Sequence[str]) -> tuple[str, ...]:
    """Build the commands that report which artifacts exist.

    Few commands rather than one per member: a campaign of thirty members
    over SSH is thirty round trips, each with its own chance to fail halfway
    and leave the plan built from a mixture of two moments. But not ONE
    command either -- see :data:`EXISTENCE_CHUNK` for the truncation a
    single 136-member line met.

    Args:
        artifacts: Absolute cluster paths. Never empty -- an empty campaign
            is refused before this is reached.

    Returns:
        Shell commands, each printing one ``PRESENT|<path>`` or
        ``ABSENT|<path>`` line per artifact, chunk by chunk in the order
        given. Each path is shell-quoted, so a filename holding a space or a
        quote is tested rather than interpreted.

    Raises:
        ValueError: If no artifact is given.
    """
    if len(artifacts) == 0:
        raise ValueError("existence_commands requires at least one artifact")
    commands: list[str] = []
    for start in range(0, len(artifacts), EXISTENCE_CHUNK):
        chunk = artifacts[start : start + EXISTENCE_CHUNK]
        quoted = " ".join(shlex.quote(artifact) for artifact in chunk)
        commands.append(
            f"for p in {quoted}; do "
            f'if [ -e "$p" ]; then echo "{_PRESENT}$p"; else echo "{_ABSENT}$p"; fi; done'
        )
    return tuple(commands)


def parse_existence(output: str) -> set[str]:
    """Read which artifacts the cluster reported as present.

    Args:
        output: The command's standard output.

    Returns:
        The paths that exist.

    Raises:
        AppError: With ``SACCT_FIELD_UNPARSABLE`` if a line carries neither
            marker. A line this cannot read is a member whose state is
            unknown, and treating unknown as absent would resubmit a job that
            is already finished -- straight into the artifact it wrote.
    """
    present: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        if stripped.startswith(_PRESENT):
            present.add(stripped[len(_PRESENT) :])
            continue
        if stripped.startswith(_ABSENT):
            continue
        raise AppError(
            Hpc3ErrorCode.SACCT_FIELD_UNPARSABLE,
            f"existence probe returned a line carrying neither {_PRESENT!r} nor "
            f"{_ABSENT!r}: {line!r}",
        )
    return present


def finished_artifacts(entries: Sequence[LedgerEntry], states: Mapping[str, str]) -> set[str]:
    """Find artifacts some job actually ran to completion.

    Args:
        entries: The whole ledger, which maps a job to the artifact it
            declared.
        states: Terminal state by job id, however obtained -- accounting,
            the closure file, or both merged. A job absent from it has no
            claim to having finished.

    Returns:
        The artifacts with at least one ``COMPLETED`` job behind them.
        PREEMPTED, FAILED and CANCELLED are all absent deliberately: each of
        them can leave a plausible file at the artifact's path, and
        ``kk_best.pt`` after a preemption at 1273 seconds is exactly such a
        file.
    """
    finished: set[str] = set()
    for entry in entries:
        artifact = entry["artifact"]
        if artifact is None:
            continue
        if states.get(entry["job_id"]) == FINISHED_STATE:
            finished.add(artifact)
    return finished


def plan_campaign(
    specs: Sequence[JobSpec],
    *,
    present: set[str],
    finished: set[str],
    claimed: dict[str, str],
) -> CampaignPlan:
    """Split the members into finished, running, and still to do.

    Args:
        specs: The expanded members, every one declaring an artifact.
        present: Artifacts that exist on the cluster.
        finished: Artifacts a job ran to completion, from
            :func:`finished_artifacts`.
        claimed: Artifact to the live job writing it, from
            :func:`~hpc3.core.inflight.claimed_artifacts`.

    Returns:
        The three groups.

        A member is done only when its artifact is in BOTH sets. Present
        alone is a checkpoint from a run that was killed; finished alone is a
        run whose output has since been moved or deleted, and resubmitting
        that is right.

        A member that is both present and in flight counts as in flight,
        because that is the state that forbids submitting: a file being
        written by a live job is a partially-written file whatever else is
        true of it.
    """
    done: list[str] = []
    in_flight: dict[str, str] = {}
    missing: list[JobSpec] = []
    for spec in specs:
        artifact = spec["artifact"]
        label = qualified_name(spec["project"], spec["name"])
        holder = claimed.get(artifact) if artifact is not None else None
        if holder is not None:
            in_flight[label] = holder
        elif artifact in present and artifact in finished:
            done.append(label)
        else:
            missing.append(spec)
    return CampaignPlan(done=done, in_flight=in_flight, missing=missing)


__all__ = [
    "EXISTENCE_CHUNK",
    "FINISHED_STATE",
    "CampaignPlan",
    "existence_commands",
    "finished_artifacts",
    "parse_existence",
    "plan_campaign",
    "require_every_member_declares_an_artifact",
]
