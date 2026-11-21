"""Retiring a settled dispatch's directory on its node (MCPs board task bfca20e6).

A dispatch's export lives in its run's enclosure ``<stage_root>/<run_id>``
(:func:`fleet.core.names.run_enclosure`) on the node it ran on, beside its
transcript. Until this module nothing removed it: measured on
diphtheria 2026-09-27, the stage root held 130 exports in 293 GB and the disk
was at 96 percent, with 133 ``mkdir-`` scripts at the root outliving the trees
they made. Nothing reads an export again once its runner has read the
transcript's tail, so the runner retires it at that moment: the transcript
moves to ``<stage_root>/logs/<run_id>.log`` (:func:`fleet.core.names.retained_log_path`),
which is the path the verdict line names, and the enclosure and the run's
root scripts are removed.

ONE RETIRE, FOUR CALLERS, like :mod:`fleet.core.stop`: the node runner's
settle (a finished run, or one stopped past its lease), every stop that
closes a row (a cancel, by ``fleet-cancel`` or under a runner), the hub's
``fleet-collect``, and every collect pass's owed retires. For a stop and the
hub's collect, which retire before the row closes, a failure propagates
with the row still live, so the next tick settles the run again; every
step of the script tolerates what an earlier attempt already did. The node
runner's settle retires after the queue close and the row's, so a node
that does not answer it leaves the retire owed and a later collect pass
sends it (:mod:`fleet.core.retire_owed`, MCPs board task 8776b828); so does
a run that never settles, refused while staging or left by a runner that
ended, since the record owes every run's retire from its lease (MCPs board
task 3d9eec0d).

THE RUN'S VIRTUALENV GOES WITH IT. Every build keeps poetry's virtualenv in
its project (:data:`fleet.core.names.POETRY_IN_PROJECT`, MCPs board task
346abc45), so it is inside the export this removes, and on Linux the copy a
docker project's build made in execdocker's home goes too.

THE CACHE GENERATIONS ARE JUDGED AFTER THE CLOSE, NOT HERE (MCPs board tasks
8993c306 and 346abc45). Removing the run removes its generation marker, so
each caller then sends the node :mod:`fleet.core.cache_generations`, once the
row is closed. Housekeeping was part of this retire until 2026-10-05, and it
stood between a finished check and its queue close: one more script round
trip, which on loki and sedona made a settle take 11 to 24 s. A removal that
fails after the close leaves nothing a later one does not remove, since it
judges every generation, not the run's alone.
"""

from __future__ import annotations

from typing import Final

from platform_core.errors import FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.node import NodeConfig
from fleet.core import dialect, names, remote
from fleet.core.retire_script import (
    RetireAnswer,
    answer_missing,
    parse_retire_answer,
    retire_script,
)


def runs_isolated(node: NodeConfig) -> bool:
    """Whether a node runs docker projects' builds as execdocker.

    Args:
        node: The node.

    Returns:
        True when it declares that user's rootless daemon (``docker``),
        the capability a docker project's dispatch requires.
    """
    return node["docker"] is not None


class RetireAttempt(TypedDict):
    """What one retire sent to a node came to.

    Attributes:
        answer: The node's answer: the process that kept the run whole, or
            the bytes its removal freed. :data:`UNANSWERED` when ``failure``
            is set.
        failure: ``None`` when the node answered; otherwise why it did not.
    """

    answer: RetireAnswer
    failure: remote.RemoteFailure | None


#: The answer an attempt carries beside a failure: no holder, nothing freed.
UNANSWERED: Final = RetireAnswer(holder=None, freed_bytes=0)


def attempt_retire_on_node(node: NodeConfig, *, run_id: str) -> RetireAttempt:
    """Keep one dispatch's transcript and remove its export and root scripts,
    unless a process uses them, reporting a failure as a value.

    Sent only by :mod:`fleet.core.retire_owed`, which records each retire
    done in the record that has owed it since the run's lease: a settle,
    which runs after the row is closed and so must tell a node that did not
    answer, whose retire a later pass does, from one that answered and
    failed (MCPs board task 8776b828), and a stop or the hub's collect,
    which raise either (:func:`fleet.core.retire_owed.retire_or_raise`).

    Args:
        node: The node it was dispatched to.
        run_id: The dispatch.

    Returns:
        The node's answer (:func:`fleet.core.retire_script.parse_retire_answer`),
        or the failure: ``NODE_UNREACHABLE`` when ssh could not reach the
        node or timed out, ``DISPATCH_FAILED`` carrying the node's own error
        when a file could not be moved or removed, or naming the line it
        ended on when the script exited 0 without an answer.
    """
    stage_root = node["stage_root"]
    outcome = remote.attempt_script(
        node["host"],
        dialect.script_path(stage_root, names.retire_stem(run_id)),
        retire_script(
            stage_root=stage_root,
            run_id=run_id,
            retained=names.retained_log_path(stage_root, run_id),
            scripts=tuple(
                dialect.script_path(stage_root, stem) for stem in names.root_script_stems(run_id)
            ),
            isolated=runs_isolated(node),
        ),
    )
    failure = outcome["failure"]
    if failure is not None:
        return RetireAttempt(answer=UNANSWERED, failure=failure)
    answer = parse_retire_answer(outcome["output"])
    if answer is None:
        missing = answer_missing(node["host"], outcome["output"])
        return RetireAttempt(
            answer=UNANSWERED,
            failure=remote.RemoteFailure(code=FleetErrorCode.DISPATCH_FAILED, message=missing),
        )
    return RetireAttempt(answer=answer, failure=None)


__all__ = ["UNANSWERED", "RetireAttempt", "attempt_retire_on_node", "runs_isolated"]
