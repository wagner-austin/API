"""Building an archive, sending it, and the scripts that run on the node.

Only the ssh boundary is faked. The archive is a real tar built by the real
tar binary over a real temporary tree and the digest is a real SHA-256, so
what is exercised is the transport as it will actually behave -- including
the exclusion that keeps one machine's ``.venv`` off another.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import cancel
from fleet.contracts.project import ProjectConfig
from fleet.core import _test_hooks, dispatch, staging
from tests.conftest import FakeClock, FakeRun, ok


def _dispatch_replies(archive_digest: str) -> list[_test_hooks.CommandResult]:
    """Every command a successful dispatch runs, in order.

    Written out rather than indexed into, because the sequence is the thing
    under test: the archive step runs `tar` through the SAME hook as ssh, so
    a list built by patching one position silently misaligns the moment a
    step is added. Naming each call makes that visible.

    Args:
        archive_digest: What the node should report having reassembled. The
            real digest for a success; anything else exercises the refusal.

    Returns:
        One result per call.
    """
    return [
        ok(""),  # probe: send script
        ok(_PROBE_OK),  # probe: run it
        ok(""),  # tar, locally
        ok(""),  # stage: send mkdir script
        ok(""),  # stage: run mkdir
        ok(""),  # stage: send the base64 payload
        ok(""),  # stage: send reassemble script
        ok(archive_digest),  # stage: run reassemble
        ok(""),  # stage: send extract script
        ok(""),  # stage: run extract
        ok(""),  # launch: send script
        ok("launched"),  # launch: run it
    ]


_NOW = 1_757_000_000
_PROJECT = "libs/demo"
_RUN_ID = f"libs-demo-{_NOW}"

_PROBE_OK = "free_ram_gb=27.0\nfree_disk_gb=860.0\n"


def _workspace_document() -> JSONObject:
    """Build a one-node, one-project workspace as JSON.

    Returns:
        The document, ready to serialise.
    """
    return {
        "nodes": {
            "lavender": {
                "host": "lavender",
                "stage_root": "C:/fleet/stage",
                "logical_cores": 16,
                "ram_gb": 32.0,
                "gpu": None,
                "budget": {
                    "reserved_cores": 2,
                    "reserved_ram_gb": 4.0,
                    "worker_ram_gb": 1.1,
                    "max_concurrent_runs": 2,
                    "max_disk_gb": 20.0,
                },
            }
        },
        "projects": {
            _PROJECT: {
                "worker_ram_gb": 1.1,
                "minimum_workers": 2,
                "expected_minutes": 5,
            }
        },
        "ledger": "ledger.jsonl",
        "feed": "feed.jsonl",
        "leases": "leases.json",
    }


@pytest.fixture(name="repo")
def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a tiny monorepo with one project in it.

    A real tree rather than a fixture archive, because the archive step runs
    the real ``tar`` and a fabricated one would test nothing about it.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The repo root.
    """
    root = tmp_path / "repo"
    (root / _PROJECT).mkdir(parents=True)
    (root / _PROJECT / "Makefile").write_text("check:\n\techo ok\n", encoding="utf-8")
    (root / _PROJECT / ".venv").mkdir()
    (root / _PROJECT / ".venv" / "huge.bin").write_text("x" * 4096, encoding="utf-8")
    return root


@pytest.fixture(name="config_path")
def _config_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a workspace document and pin the clock.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path to the written document.
    """
    _test_hooks.now = FakeClock(_NOW)
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(_workspace_document()), encoding="utf-8")
    return path


def _plan() -> ProjectConfig:
    """The demo project's declaration.

    Returns:
        The project.
    """
    return ProjectConfig(worker_ram_gb=1.1, minimum_workers=2, expected_minutes=5)


class TestArchive:
    def test_it_builds_a_real_archive_and_excludes_the_venv(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """.venv leads the exclusion list for the reason the package exists.

        One machine's has absolute paths baked into it, and it is the exact
        thing two dispatches must not share.
        """
        destination = tmp_path / "tree.tgz"

        payload = staging.archive(repo, _PROJECT, destination)

        assert destination.is_file()
        # A real gzip member, not merely some bytes: 1f 8b is the magic, and
        # the payload must be exactly what landed on disk or the digest the
        # node is asked to match would be of something else.
        assert payload[:2] == b"\x1f\x8b"
        assert payload == destination.read_bytes()
        listing = _test_hooks.run(["tar", "-tzf", str(destination)])["stdout"]
        assert "Makefile" in listing
        assert ".venv" not in listing

    def test_a_project_that_does_not_exist_is_refused(
        self, repo: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(AppError) as excinfo:
            staging.archive(repo, "libs/absent", tmp_path / "tree.tgz")

        assert excinfo.value.code is FleetErrorCode.STAGE_ARCHIVE_UNREADABLE

    def test_the_digest_is_a_full_length_sha256(self) -> None:
        assert len(staging.digest(b"payload")) == 64

    def test_encoding_round_trips_through_base64(self) -> None:
        """Base64 because raw bytes do not survive ssh into PowerShell."""
        import base64

        payload = bytes(range(256))

        assert base64.b64decode(staging.encode(payload)) == payload


class TestStage:
    def test_a_verified_archive_is_unpacked(self) -> None:
        payload = b"archive-bytes"
        runner = FakeRun(
            [
                ok(""),  # send mkdir script
                ok(""),  # run mkdir
                ok(""),  # send the base64
                ok(""),  # send reassemble script
                ok(staging.digest(payload)),  # run reassemble -> digest
                ok(""),  # send extract script
                ok(""),  # run extract
            ]
        )
        _test_hooks.run = runner

        target = staging.stage(
            "lavender", run_id=_RUN_ID, stage_root="C:/fleet/stage", payload=payload
        )

        assert target == f"C:/fleet/stage/{_RUN_ID}"
        assert any("tar -xzmf" in (call[-1] if call else "") for call in runner.calls) or any(
            b"tar -xzmf" in (sent or b"") for sent in runner.stdin
        )

    def test_a_mismatched_digest_refuses_before_unpacking(self) -> None:
        """Nothing is extracted, so no unverified tree lands where make looks."""
        runner = FakeRun([ok(""), ok(""), ok(""), ok(""), ok("0" * 64)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as excinfo:
            staging.stage("lavender", run_id=_RUN_ID, stage_root="C:/fleet/stage", payload=b"bytes")

        assert excinfo.value.code is FleetErrorCode.STAGE_DIGEST_MISMATCH
        assert "nothing has been unpacked" in excinfo.value.message
        assert not any(b"tar -xzmf" in (sent or b"") for sent in runner.stdin)


class TestScripts:
    def test_the_launch_script_registers_a_scheduled_task(self) -> None:
        """Not an ssh child. Windows OpenSSH puts that in a job object that
        dies with the connection, and this command returns immediately."""
        body = dispatch.launch_script(target="C:/s/run-1", project=_PROJECT, workers=6)

        assert "Register-ScheduledTask" in body
        assert "Start-ScheduledTask" in body

    def test_the_launch_script_sets_priority_four(self) -> None:
        """Priority 7 is the Register-ScheduledTask default and sets LOW I/O.

        A run that inherits it crawls, and the symptom reads as a slow node
        rather than a misconfigured launch.
        """
        body = dispatch.launch_script(target="C:/s/run-1", project=_PROJECT, workers=6)

        assert "-Priority 4" in body
        assert "[TimeSpan]::Zero" in body
        assert "-LogonType S4U" in body

    def test_the_launch_script_pins_the_worker_count(self) -> None:
        body = dispatch.launch_script(target="C:/s/run-1", project=_PROJECT, workers=6)

        assert "PYTEST_XDIST_AUTO_NUM_WORKERS = '6'" in body

    def test_the_extract_script_keeps_the_node_s_clock(self) -> None:
        """Without -m, a tree from a fast clock makes targets look fresh.

        The build then does nothing, which reads as a suite that passed
        instantly.
        """
        assert "-xzmf" in staging.extract_script("C:/s/run-1")

    def test_the_result_script_prints_nothing_while_running(self) -> None:
        """Absence is the signal, so an unfinished run is not read as exit 0."""
        body = dispatch.result_script("C:/s/run-1")

        assert "Test-Path" in body

    def test_the_stop_script_never_prompts(self) -> None:
        """There is nobody at the node to answer, and a prompt would hang."""
        assert "-Confirm:$false" in cancel.stop_script(_RUN_ID)
