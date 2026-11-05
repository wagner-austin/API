"""Two jobs carrying one companion tip at once each name their own bundle file
(MCPs board task f9ee56aa).

Every runner on the hub bundled a companion into one file named by its
mirror key and tip, and ``git bundle create`` does not write one ref's bytes
twice alike, so a second runner's bundle replaced the first's between that
runner's digest and its copy: wiki-check job a520c74c on diphtheria was
refused ``STAGE_DIGEST_MISMATCH`` at 06:57Z on 2026-10-15, its received digest
the hub file's after lavender-wsl's runner rewrote it at 06:56:11Z. Since MCPs
board task 5280a25d a companion's bundle is written only once the node has
said what its mirror lacks (:mod:`fleet.core.companion_stage`), so the
prepare writes none and names the file its stage writes. The case prepares
two claimed jobs of one project carrying one companion in two threads,
through the real prepare and the real export, with a git that answers each
command and fails any bundle.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_prepare
from fleet.contracts.dispatch import DispatchJob, decode_job
from fleet.contracts.source import ProjectCompanion
from fleet.contracts.workspace import require_project
from fleet.core import _test_hooks, export
from tests._node_agent_fixtures import (
    COMPANION_DIRECTORY,
    COMPANION_REF,
    COMPANION_REMOTE,
    COMPANION_SHA,
    NPM_CI,
    _credentials_in_env,
    sourced_document,
)
from tests._queue_fakes import DEFAULT_SHA, queue_job
from tests._registry import write_registry
from tests.conftest import DEMO_PROJECT, ok

__all__ = ["_credentials_in_env"]

#: The two claimed jobs, one per runner.
FIRST_JOB = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
SECOND_JOB = "cccccccc-3333-4333-8333-cccccccccccc"


def hub_git(
    argv: Sequence[str],
    *,
    timeout_seconds: int,
    stdin_bytes: bytes | None = None,
    unset_env: Sequence[str] = (),
    set_env: Sequence[tuple[str, str]] = (),
) -> _test_hooks.CommandResult:
    """The hub's git for two concurrent prepares of one companion.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`. Answered by
    command rather than in order, because the two prepares interleave.

    Args:
        argv: The command.
        timeout_seconds: The deadline the caller chose.
        stdin_bytes: Its standard input, or None.
        unset_env: The variables the caller withheld from the child.
        set_env: The variables the caller set in the child.

    Returns:
        The tip for ``rev-parse``, and success for every other command.

    Raises:
        AssertionError: For a ``git bundle create``, which no prepare runs.
    """
    assert tuple(argv[3:5]) != ("bundle", "create"), f"a prepare bundled: {list(argv)}"
    if "rev-parse" in argv:
        return ok(f"{COMPANION_SHA}\n")
    return ok("")


def _job(job_id: str) -> DispatchJob:
    """One claimed check of the demo project.

    Args:
        job_id: The queue's id for it.

    Returns:
        The decoded job.
    """
    row = queue_job(id=job_id, status="claimed")
    return decode_job(row, answer=dump_json_str({"claimed": row}))


class TestTwoJobsCarryingOneCompanionTipAtOnce:
    def test_each_names_a_bundle_in_its_own_attempt_and_writes_none(
        self, config_path: pathlib.Path
    ) -> None:
        sourced = write_registry(
            config_path,
            sourced_document(
                (NPM_CI,),
                (
                    ProjectCompanion(
                        remote=COMPANION_REMOTE, ref=COMPANION_REF, directory=COMPANION_DIRECTORY
                    ),
                ),
            ),
        )
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced)})
        admitted = node_prepare.Admitted(
            plan=require_project(loaded.workspace, DEMO_PROJECT), workers=1
        )
        _test_hooks.run = hub_git

        def prepare(job_id: str) -> export.CompanionExport:
            prepared = node_prepare.prepare(
                loaded, _job(job_id), admitted=admitted, sha=DEFAULT_SHA
            )
            (companion,) = prepared["companions"]
            return companion

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.map(prepare, (FIRST_JOB, SECOND_JOB))

        # Each in its launch attempt's own directory, named for its job
        # (board task 2e3e69cf), so neither job nor a later attempt of one
        # can write where the other did; nothing is there yet.
        name = f"companion-wagner-austin-MCPs-{COMPANION_SHA}.bundle"
        for companion, job_id in ((first, FIRST_JOB), (second, SECOND_JOB)):
            assert (companion["bundle"].name, companion["bundle"].parent.parent) == (
                name,
                loaded.archives,
            )
            assert companion["bundle"].parent.name.startswith(f"{job_id}-")
            assert not companion["bundle"].exists()
            assert (companion["sha"], companion["key"]) == (
                COMPANION_SHA,
                "companion-wagner-austin-MCPs",
            )
        assert first["bundle"].parent != second["bundle"].parent
