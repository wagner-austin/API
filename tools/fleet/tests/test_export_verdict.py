"""The export of a commit (MCPs board task fd5cabfa, A2) and the verdict a
finished check becomes (A3).

The export half runs git through the one command hook: what is asserted is
the exact argv of every git call, its deadline, and which of git's refusals
becomes which fleet code. The mirror directory and the archive file are
real, in a temporary directory, because the file hooks are left on their
real implementations by the autouse reset. The verdict half is pure: real
transcript tails from vitest and pytest-cov in, the one thread line out.
"""

from __future__ import annotations

import pathlib
import socket

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.source import ProjectSource
from fleet.core import _test_hooks, export, verdict
from tests.conftest import FakeRun, failed, ok

SHA = "4e3c6bc1d9f0a7b2c3e4f5061728394a5b6c7d8e"
REMOTE = "https://github.com/wagner-austin/MCPs.git"
PROJECT = "MCPs/packages/wiki-search"

VITEST_TAIL = (
    " % Coverage report from v8\n"
    "--------------|---------|----------|---------|---------|\n"
    "File          | % Stmts | % Branch | % Funcs | % Lines |\n"
    "All files     |   99.71 |    98.42 |     100 |   99.71 |\n"
    " Test Files  60 passed (60)\n"
    "      Tests  2 failed | 885 passed (887)\n"
    "ERROR: Coverage for branches (98.42%) does not meet global threshold (100%)\n"
)

PYTEST_TAIL = (
    "src\\fleet\\core\\verdict.py     54      0     14      0 100.00%\n"
    "----------------------------------------------------------------\n"
    "TOTAL                        3131      0    862      0 100.00%\n"
    "\n"
    "771 passed, 4 skipped in 20.06s\n"
    "=== ALL CHECKS PASSED ===\n"
)


class TestMirrorPath:
    def test_the_key_folds_its_slashes_into_one_directory(self, tmp_path: pathlib.Path) -> None:
        assert export.mirror_path(tmp_path, PROJECT) == tmp_path / "MCPs-packages-wiki-search.git"
        assert export.mirror_path(tmp_path, "slime") == tmp_path / "slime.git"


class TestEnsureMirror:
    def test_a_new_mirror_is_initialised_bare_under_a_made_parent(
        self, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([ok("")])
        _test_hooks.run = runner
        mirror = tmp_path / "runs" / "mirrors" / "slime.git"

        export.ensure_mirror(mirror)

        assert runner.calls == [("git", "init", "--bare", "--quiet", str(mirror))]
        assert runner.timeouts == [export.INIT_TIMEOUT_SECONDS]
        assert mirror.parent.is_dir()

    def test_an_existing_mirror_is_left_alone(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner
        mirror = tmp_path / "slime.git"
        mirror.mkdir()

        export.ensure_mirror(mirror)

        assert runner.calls == []

    def test_a_failing_init_is_an_export_failure_with_gits_words(
        self, tmp_path: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun([failed(128, "fatal: cannot mkdir: Permission denied")])

        with pytest.raises(AppError) as raised:
            export.ensure_mirror(tmp_path / "slime.git")

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message.startswith("git init --bare ")
        assert raised.value.message.endswith("Permission denied")

    def test_a_failure_with_nothing_on_stderr_still_says_so(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([failed(1, "   ")])

        with pytest.raises(AppError, match=r"\(git wrote nothing to stderr\)"):
            export.ensure_mirror(tmp_path / "slime.git")


class TestFetchCommit:
    def test_a_commit_already_held_costs_no_fetch(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([ok("")])
        _test_hooks.run = runner

        export.fetch_commit(tmp_path, REMOTE, SHA)

        assert runner.calls == [("git", "-C", str(tmp_path), "cat-file", "-e", f"{SHA}^{{commit}}")]

    def test_a_missing_commit_is_fetched_by_sha_without_tags(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([failed(128, "missing"), ok("")])
        _test_hooks.run = runner

        export.fetch_commit(tmp_path, REMOTE, SHA)

        assert runner.calls[1] == (
            "git",
            "-C",
            str(tmp_path),
            "fetch",
            "--quiet",
            "--no-tags",
            REMOTE,
            SHA,
        )
        assert runner.timeouts == [export.INIT_TIMEOUT_SECONDS, export.FETCH_TIMEOUT_SECONDS]

    @pytest.mark.parametrize("marker", export.NOT_ON_REMOTE_MARKERS)
    def test_every_spelling_of_an_unserved_object_is_sha_not_on_remote(
        self, tmp_path: pathlib.Path, marker: str
    ) -> None:
        _test_hooks.run = FakeRun([failed(128, "missing"), failed(128, f"fatal: {marker}: {SHA}")])

        with pytest.raises(AppError) as raised:
            export.fetch_commit(tmp_path, REMOTE, SHA)

        assert raised.value.code is FleetErrorCode.SHA_NOT_ON_REMOTE
        assert raised.value.message.startswith(f"{REMOTE} does not serve {SHA}")
        assert f"git said: fatal: {marker}: {SHA}" in raised.value.message

    def test_any_other_fetch_failure_is_an_export_failure(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun(
            [failed(128, "missing"), failed(128, "fatal: unable to access: Could not resolve host")]
        )

        with pytest.raises(AppError) as raised:
            export.fetch_commit(tmp_path, REMOTE, SHA)

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message == (
            f"git fetch {REMOTE} {SHA} into {tmp_path}: fatal: unable to access: Could not "
            "resolve host"
        )


class TestArchiveCommit:
    def test_the_archive_is_written_by_git_and_read_back_as_bytes(
        self, tmp_path: pathlib.Path
    ) -> None:
        destination = tmp_path / "archives" / "run-1-lavender.tgz"
        payload = b"\x1f\x8b\x08\x00archive"
        # git is faked, so the bytes it would have written are placed first;
        # the read after the call is the real file hook.
        destination.parent.mkdir()
        destination.write_bytes(payload)
        runner = FakeRun([ok("")])
        _test_hooks.run = runner

        data = export.archive_commit(tmp_path / "slime.git", SHA, destination)

        assert data == payload
        assert runner.calls == [
            (
                "git",
                "-C",
                str(tmp_path / "slime.git"),
                "archive",
                "--format=tar.gz",
                "-o",
                str(destination),
                SHA,
            )
        ]
        assert runner.timeouts == [export.ARCHIVE_TIMEOUT_SECONDS]
        assert destination.parent.is_dir()

    def test_a_failing_archive_is_an_export_failure(self, tmp_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([failed(128, f"fatal: not a valid object name: {SHA}")])

        with pytest.raises(AppError) as raised:
            export.archive_commit(tmp_path / "slime.git", SHA, tmp_path / "out.tgz")

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message.startswith(f"git archive {SHA} from ")


class TestRequireSourceAndPrepare:
    def test_a_declared_source_is_handed_back(self) -> None:
        source = ProjectSource(remote=REMOTE, path="packages/wiki-search", install=())

        assert export.require_source(PROJECT, source) is source

    def test_a_project_without_one_is_refused_by_name_with_the_two_ways_out(self) -> None:
        with pytest.raises(AppError) as raised:
            export.require_source("libs/demo", None)

        assert raised.value.code is FleetErrorCode.PROJECT_REMOTE_MISSING
        assert raised.value.message.startswith("project 'libs/demo' declares no source")
        assert "give it a source (remote, path, install)" in raised.value.message
        assert "fleet-run" in raised.value.message

    def test_prepare_initialises_probes_and_fetches_in_that_order(
        self, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([ok(""), failed(128, "missing"), ok("")])
        _test_hooks.run = runner

        mirror = export.prepare_mirror(tmp_path, project=PROJECT, remote=REMOTE, sha=SHA)

        assert mirror == tmp_path / "MCPs-packages-wiki-search.git"
        assert runner.calls[0][:2] == ("git", "init")
        assert runner.calls[1][3] == "cat-file"
        assert runner.calls[2][3] == "fetch"
        assert len(runner.calls) == 3


class TestReadingTheTail:
    def test_a_vitest_tail_reads_counts_and_the_tree_row(self) -> None:
        assert verdict.read_tests(VITEST_TAIL) == "885p/2f"
        assert verdict.read_coverage(VITEST_TAIL) == "statements=99.71% branches=98.42%"

    def test_a_pytest_cov_tail_reads_counts_and_the_total_row(self) -> None:
        assert verdict.read_tests(PYTEST_TAIL) == "771p/0f"
        assert verdict.read_coverage(PYTEST_TAIL) == "total=100.00%"

    def test_the_last_passed_count_wins_over_per_file_lines(self) -> None:
        tail = "test_a.py 12 passed\ntest_b.py 30 passed\n42 passed in 1.0s\n"

        assert verdict.read_tests(tail) == "42p/0f"

    def test_a_threshold_refusal_without_the_tree_row_names_the_figure(self) -> None:
        tail = "ERROR: Coverage for statements (99.5%) does not meet global threshold (100%)\n"

        assert verdict.read_coverage(tail) == "statements=99.5% below threshold"

    def test_a_tail_with_neither_is_unread_not_zero(self) -> None:
        tail = "make: *** [Makefile:12: check] Error 2\n"

        assert verdict.read_tests(tail) == "unread"
        assert verdict.read_coverage(tail) == "unread"


class TestJudgeAndRender:
    def test_a_green_run_renders_the_one_line_a_closure_cites(self) -> None:
        judged = verdict.judge(
            job_id="3f2a9c1e-0000-4000-8000-000000000000",
            project=PROJECT,
            sha=SHA,
            node="lavender",
            exit_code=0,
            tail=PYTEST_TAIL,
            log_path="C:/fleet/stage/run-1/result.txt.log",
            run_id="run-1",
        )

        assert judged["banner"] is True
        assert verdict.render_verdict(judged) == (
            f"FLEET-CHECK 3f2a9c1e {PROJECT} sha={SHA} node=lavender exit=0 banner=yes "
            "tests=771p/0f coverage=total=100.00% log=lavender:C:/fleet/stage/run-1/result.txt.log "
            "run=run-1"
        )

    def test_a_red_run_says_no_banner_and_carries_the_failed_count(self) -> None:
        judged = verdict.judge(
            job_id="3f2a9c1e-0000-4000-8000-000000000000",
            project=PROJECT,
            sha=SHA,
            node="sedona",
            exit_code=1,
            tail=VITEST_TAIL,
            log_path="/srv/fleet/stage/run-2/result.txt.log",
            run_id="run-2",
        )

        assert judged["banner"] is False
        line = verdict.render_verdict(judged)
        assert line.startswith("FLEET-CHECK 3f2a9c1e ")
        assert " exit=1 banner=no tests=885p/2f coverage=statements=99.71% branches=98.42% " in line
        assert line.endswith("log=sedona:/srv/fleet/stage/run-2/result.txt.log run=run-2")
        assert "\n" not in line


class TestHostnameHook:
    def test_the_real_hostname_reader_lowercases_the_machines_name(self) -> None:
        assert _test_hooks._default_hostname() == socket.gethostname().lower()
        assert _test_hooks._default_hostname() != ""
