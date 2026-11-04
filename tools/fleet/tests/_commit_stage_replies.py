"""The commands a queued commit's stage runs, for the node-lane sequences that fake them.

One list for every sequence that stages a commit from the node's mirror
(:func:`fleet.core.commit_stage.stage_commit`, MCPs board task 1e0ee821), as
:func:`tests.conftest.stage_replies` is for a working tree's archive.
"""

from __future__ import annotations

from fleet.core import _test_hooks
from tests.conftest import ok


def commit_stage_replies(bundle_digest: str) -> list[_test_hooks.CommandResult]:
    """Every command staging one commit on a node with no mirror of its remote runs.

    Args:
        bundle_digest: What the node should report having received.

    Returns:
        One result per call, in order.
    """
    return [
        ok(""),  # stage: send the prepare script
        ok(""),  # stage: run it; the node's mirror does not exist yet
        ok(""),  # bundle: git update-ref of the run's bundle ref
        ok(""),  # bundle: git bundle create, the commit's whole history
        ok(""),  # bundle: git update-ref -d of the run's bundle ref
        ok(""),  # stage: scp the bundle into the staging directory
        ok(""),  # stage: send the digest script
        ok(bundle_digest),  # stage: run it
        ok(""),  # stage: send the unpack script
        ok(""),  # stage: run it
    ]


__all__ = ["commit_stage_replies"]
