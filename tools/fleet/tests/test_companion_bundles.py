"""Two jobs bundling one companion tip at once each stage their own bytes
(MCPs board task f9ee56aa).

Every runner on the hub bundled a companion into one file named by its
mirror key and tip, and ``git bundle create`` does not write one ref's bytes
twice alike, so a second runner's bundle replaced the first's between that
runner's digest and its copy: wiki-check job a520c74c on diphtheria was
refused ``STAGE_DIGEST_MISMATCH`` at 06:57Z on 2026-10-15, its received digest
the hub file's after lavender-wsl's runner rewrote it at 06:56:11Z. The case
prepares two claimed jobs of one project carrying one companion in two
threads, through the real prepare and the real export, with a git that writes
different bytes on every bundle and holds each bundle until both are
written, which is the interleaving that lost the run.
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_prepare
from fleet.contracts.dispatch import DispatchJob, decode_job
from fleet.contracts.source import ProjectCompanion
from fleet.contracts.workspace import require_project
from fleet.core import _test_hooks, export, staging
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


class BundlingGit:
    """The hub's git for two concurrent prepares of one companion.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`. Answered by
    command rather than in order, because the two prepares interleave. Each
    ``git bundle create`` writes bytes no other call writes, as git does, and
    returns only once both have written, so each export reads its bundle back
    after the other's write, which is when a shared file was replaced.

    Attributes:
        written: Each bundle path and the bytes written there, in write order.
    """

    written: list[tuple[pathlib.Path, bytes]]

    def __init__(self) -> None:
        """Start with nothing written and both bundles awaited."""
        self.written = []
        self._lock = threading.Lock()
        self._both_written = threading.Barrier(2, timeout=10)

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Answer one git command.

        Args:
            argv: The command.
            timeout_seconds: The deadline the caller chose.
            stdin_bytes: Its standard input, or None.
            unset_env: The variables the caller withheld from the child.
            set_env: The variables the caller set in the child.

        Returns:
            The tip for ``rev-parse``, and success for every other command.
        """
        if "rev-parse" in argv:
            return ok(f"{COMPANION_SHA}\n")
        if tuple(argv[3:5]) == ("bundle", "create"):
            path = pathlib.Path(argv[-2])
            with self._lock:
                payload = f"# v2 git bundle\nwrite {len(self.written)}".encode()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                self.written.append((path, payload))
            self._both_written.wait()
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


class TestTwoJobsBundlingOneCompanionTipAtOnce:
    def test_each_stages_the_bytes_its_own_bundle_wrote(self, config_path: pathlib.Path) -> None:
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
        git = BundlingGit()
        _test_hooks.run = git

        def prepare(job_id: str) -> export.CompanionExport:
            prepared = node_prepare.prepare(
                loaded, _job(job_id), admitted=admitted, sha=DEFAULT_SHA
            )
            (companion,) = prepared["companions"]
            return companion

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.map(prepare, (FIRST_JOB, SECOND_JOB))

        # What scp copies is what the node's digest is held to, for both runs.
        for companion in (first, second):
            assert staging.digest(companion["path"].read_bytes()) == staging.digest(
                companion["data"]
            )
        assert first["data"] != second["data"]
        assert sorted(git.written) == sorted(
            [(first["path"], first["data"]), (second["path"], second["data"])]
        )
        name = f"companion-wagner-austin-MCPs-{COMPANION_SHA}.bundle"
        assert first["path"] == loaded.archives / FIRST_JOB / name
        assert second["path"] == loaded.archives / SECOND_JOB / name
