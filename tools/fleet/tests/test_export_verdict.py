"""The export of a commit (MCPs board task fd5cabfa, A2) and the verdict a
finished check becomes (A3).

The export half runs git through the one command hook: what is asserted is
the exact argv of every git call, its deadline, and which of git's refusals
becomes which fleet code; the fetches are ``tests/test_mirror_fetch.py``'s
and a commit's bundle and unpack ``tests/test_commit_stage.py``'s. The
mirror directory and the bundle file are real, in a temporary directory,
because the file hooks are left on their real implementations by the
autouse reset. The verdict half is pure: real transcript tails from vitest
and pytest-cov in, the one thread line out.
"""

from __future__ import annotations

import pathlib
import socket
import threading
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.source import ProjectCompanion, ProjectSource
from fleet.core import _test_hooks, export, mirror_fetch, verdict
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
    def test_the_key_names_one_bare_directory(self, tmp_path: pathlib.Path) -> None:
        mcps = export.mirror_path(tmp_path, "wagner-austin-MCPs")
        slime = export.mirror_path(tmp_path, "wagner-austin-slime")

        assert mcps == tmp_path / "wagner-austin-MCPs.git"
        assert slime == tmp_path / "wagner-austin-slime.git"


class TestEnsureMirror:
    def test_a_new_mirror_is_initialised_bare_under_a_made_parent(
        self, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([ok("")])
        _test_hooks.run = runner
        mirror = tmp_path / "runs" / "mirrors" / "slime.git"

        export.ensure_mirror(mirror)

        assert runner.calls == [("git", "init", "--bare", "--quiet", str(mirror))]
        assert runner.timeouts == [mirror_fetch.INIT_TIMEOUT_SECONDS]
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


class TestRequireSourceAndPrepare:
    def test_a_declared_source_is_handed_back(self) -> None:
        source = ProjectSource(
            remote=REMOTE, path="packages/wiki-search", install=(), companions=()
        )

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
        runner = FakeRun([ok(""), failed(1, ""), failed(128, "missing"), ok("")])
        _test_hooks.run = runner

        mirror = export.prepare_mirror(tmp_path, remote=REMOTE, sha=SHA)

        assert mirror == tmp_path / "wagner-austin-MCPs.git"
        assert runner.calls[0][:2] == ("git", "init")
        assert runner.calls[1][3] == "merge-base"
        assert runner.calls[2][3] == "cat-file"
        assert runner.calls[3][3] == "fetch"
        assert len(runner.calls) == 4

    def test_a_second_package_of_the_remote_reuses_the_mirror_and_its_commit(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Every package of one repository shares one mirror: the second
        prepare finds the directory, finds the commit the first fetched, and
        neither initialises nor fetches."""
        (tmp_path / "wagner-austin-MCPs.git").mkdir()
        runner = FakeRun([ok("")])
        _test_hooks.run = runner

        mirror = export.prepare_mirror(tmp_path, remote=REMOTE, sha=SHA)

        assert mirror == tmp_path / "wagner-austin-MCPs.git"
        assert runner.calls == [
            (
                "git",
                "-C",
                str(mirror),
                "merge-base",
                "--is-ancestor",
                SHA,
                mirror_fetch.FETCHED_REF,
            ),
        ]


class TestRemoteMirrorKey:
    @pytest.mark.parametrize(
        "remote",
        [
            "https://github.com/wagner-austin/MCPs.git",
            "git@github.com:wagner-austin/MCPs.git",
            "ssh://git@github.com/wagner-austin/MCPs.git",
        ],
    )
    def test_every_spelling_of_one_repository_names_one_mirror(self, remote: str) -> None:
        assert export.remote_mirror_key(remote) == "wagner-austin-MCPs"

    def test_two_repositories_of_one_name_do_not_share_a_mirror(self) -> None:
        ours = export.remote_mirror_key("https://github.com/wagner-austin/MCPs.git")
        theirs = export.remote_mirror_key("https://github.com/someone-else/MCPs.git")

        assert ours != theirs


class TestCompanionMirrorKey:
    @pytest.mark.parametrize(
        "remote",
        [
            "https://github.com/wagner-austin/MCPs.git",
            "git@github.com:wagner-austin/MCPs.git",
            "ssh://git@github.com/wagner-austin/MCPs.git",
        ],
    )
    def test_every_spelling_of_one_repository_names_one_mirror(self, remote: str) -> None:
        """The three remote forms of a repository are that repository, so they
        share the mirror rather than fetching it three times."""
        assert export.companion_mirror_key(remote) == "companion-wagner-austin-MCPs"

    def test_two_repositories_of_one_name_do_not_share_a_mirror(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The key is the remote and not the directory a companion lands in:
        one mirror serving two repositories would flip between them under one
        ref, and the tree staged would be whichever fetched last."""
        ours = export.companion_mirror_key("https://github.com/wagner-austin/MCPs.git")
        theirs = export.companion_mirror_key("https://github.com/someone-else/MCPs.git")

        assert ours != theirs
        assert export.mirror_path(tmp_path, ours) != export.mirror_path(tmp_path, theirs)


class TestExportingCompanions:
    def test_a_companion_is_mirrored_and_fetched_at_its_tip_and_bundled_nowhere(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The whole path one declaration takes on the hub before the lease,
        in order: the mirror made and the ref's tip fetched and resolved. Its
        bundle is named in the job's directory and written only once the
        node has said what its mirror lacks (MCPs board task 5280a25d)."""
        archives = tmp_path / "archives"
        runner = FakeRun([ok(""), ok(""), ok(f"{SHA}\n")])
        _test_hooks.run = runner

        exported = export.export_companions(
            tmp_path / "mirrors",
            archives,
            (ProjectCompanion(remote=REMOTE, ref="main", directory="MCPs"),),
        )

        mirror = tmp_path / "mirrors" / "companion-wagner-austin-MCPs.git"
        assert exported == (
            export.CompanionExport(
                directory="MCPs",
                ref="main",
                sha=SHA,
                mirror=mirror,
                key="companion-wagner-austin-MCPs",
                bundle=archives / f"companion-wagner-austin-MCPs-{SHA}.bundle",
            ),
        )
        assert runner.calls[0][:2] == ("git", "init")
        assert runner.calls[1][3] == "fetch"
        assert runner.calls[2][3] == "rev-parse"
        assert len(runner.calls) == 3
        assert not archives.exists()

    def test_companions_are_fetched_at_once_and_answered_in_declared_order(
        self, tmp_path: pathlib.Path
    ) -> None:
        """packages/wiki-check declares 17 companions, and one after another
        their fetches kept lavender idle 51 s between job 30708103's claim
        and its first phase (MCPs board task 5280a25d). Each fetch here waits
        until all three are under way, so fetched one at a time the first
        never returns and the barrier breaks."""
        companions = tuple(
            ProjectCompanion(
                remote=f"https://github.com/wagner-austin/{name}.git", ref="main", directory=name
            )
            for name in ("wiki", "tech-wiki", "me-wiki")
        )
        under_way = threading.Barrier(len(companions), timeout=10)
        tips = {"wiki": "1" * 40, "tech-wiki": "2" * 40, "me-wiki": "3" * 40}

        def hub_git(
            argv: Sequence[str],
            *,
            timeout_seconds: int,
            stdin_bytes: bytes | None = None,
            unset_env: Sequence[str] = (),
            set_env: Sequence[tuple[str, str]] = (),
        ) -> _test_hooks.CommandResult:
            name = pathlib.PurePath(argv[-1] if argv[1] == "init" else argv[2]).name
            remote_name = name.removeprefix("companion-wagner-austin-").removesuffix(".git")
            if "fetch" in argv:
                under_way.wait()
            return ok(f"{tips[remote_name]}\n" if "rev-parse" in argv else "")

        _test_hooks.run = hub_git

        exported = export.export_companions(tmp_path / "mirrors", tmp_path / "archives", companions)

        assert [(row["directory"], row["sha"]) for row in exported] == [
            ("wiki", "1" * 40),
            ("tech-wiki", "2" * 40),
            ("me-wiki", "3" * 40),
        ]
        assert len(companions) <= export.COMPANION_FETCHES

    def test_a_project_declaring_none_asks_git_nothing(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner

        assert export.export_companions(tmp_path, tmp_path, ()) == ()
        assert runner.calls == []


class TestReadingTheTail:
    def test_a_vitest_tail_reads_counts_and_the_tree_row(self) -> None:
        assert verdict.read_tests(VITEST_TAIL) == "885p/2f"
        assert verdict.read_coverage(VITEST_TAIL) == "statements=99.71% branches=98.42%"

    def test_a_pytest_cov_tail_reads_counts_and_the_total_row(self) -> None:
        assert verdict.read_tests(PYTEST_TAIL) == "771p/0f"
        assert verdict.read_coverage(PYTEST_TAIL) == "total=100.00%"

    def test_a_gradle_summary_tail_reads_its_total_rows(self) -> None:
        """The last lines of osrs-lab desktop-input's make check as execdocker
        on colossus, 2026-11-02 (board task fd8e5f9b), byte for byte: each
        module's row, then the totals the verdict reads."""
        tail = (
            "> Task :check\n"
            "TESTS :input-policy:test: 5 passed, 0 failed, 0 skipped of 5\n"
            "TESTS :test: 63 passed, 0 failed, 0 skipped of 63\n"
            "COVERAGE :input-policy:jacocoTestReport: lines 37/37 branches 26/26\n"
            "COVERAGE :jacocoTestReport: lines 555/555 branches 352/352\n"
            "TESTS TOTAL: 68 passed, 0 failed, 0 skipped of 68\n"
            "COVERAGE TOTAL: lines 592/592 branches 378/378\n"
            "\n"
            "BUILD SUCCESSFUL in 41s\n"
            "12 actionable tasks: 12 executed\n"
            "=== ALL CHECKS PASSED ===\n"
        )

        assert verdict.read_tests(tail) == "68p/0f"
        assert verdict.read_coverage(tail) == "lines=592/592 branches=378/378"

    def test_a_gradle_module_row_alone_is_not_the_total(self) -> None:
        tail = "COVERAGE :jacocoTestReport: lines 725/951 branches 461/646\n"

        assert verdict.read_coverage(tail) == "unread"

    def test_the_last_passed_count_wins_over_per_file_lines(self) -> None:
        tail = "test_a.py 12 passed\ntest_b.py 30 passed\n42 passed in 1.0s\n"

        assert verdict.read_tests(tail) == "42p/0f"

    def test_a_service_log_line_saying_failed_is_not_the_failed_count(self) -> None:
        """Lines 3562, 4156 and 4157 of colossus's
        MCPs-watcher-colossus-1791760604.log (FLEET-CHECK 0bce943c, MCPs/watcher
        at 4ae0ec95f, 2026-10-11), byte for byte: the http client's retry
        warning, then vitest's two total lines. The run failed nothing, and
        the verdict read ``2312p/3f`` off the warning's ``1/3 failed``."""
        tail = (
            '{"ts":"2026-10-11T23:18:22.497Z","svc":"watcher","tid":"-","rid":"-",'
            '"msg":"warn [http-retry] GET http://127.0.0.1:45133/AgendaViewer.php: '
            'attempt 1/3 failed, retrying: fetch failed: other side closed"}\n'
            "\x1b[2m Test Files \x1b[22m \x1b[1m\x1b[32m220 passed"
            "\x1b[39m\x1b[22m\x1b[90m (220)\x1b[39m\n"
            "\x1b[2m      Tests \x1b[22m \x1b[1m\x1b[32m2312 passed"
            "\x1b[39m\x1b[22m\x1b[90m (2312)\x1b[39m\n"
        )

        assert verdict.read_tests(tail) == "2312p/0f"

    def test_a_pytest_failed_count_is_read_off_the_totals_line(self) -> None:
        tail = "test_a.py:3: AssertionError\n=== 3 failed, 1166 passed in 12.3s ==="

        assert verdict.read_tests(tail) == "1166p/3f"

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
            ended_unix=1791146008,
            log_path="C:/fleet/stage/run-1/result.txt.log",
            run_id="run-1",
            workers=4,
        )

        assert judged["banner"] is True
        assert verdict.render_verdict(judged) == (
            f"FLEET-CHECK 3f2a9c1e {PROJECT} sha={SHA} node=lavender exit=0 banner=yes "
            "tests=771p/0f coverage=total=100.00% budget=unread ended=2026-10-04T20:33:28Z "
            "log=lavender:C:/fleet/stage/run-1/result.txt.log "
            "run=run-1 workers=4 slowest=unread ci=unread dispatch=unread vmtests=unread "
            "generation=unread began=unread phases=unread steps=unread registry=unread"
        )

    def test_a_red_run_says_no_banner_and_carries_the_failed_count(self) -> None:
        judged = verdict.judge(
            job_id="3f2a9c1e-0000-4000-8000-000000000000",
            project=PROJECT,
            sha=SHA,
            node="sedona",
            exit_code=1,
            tail=VITEST_TAIL,
            ended_unix=1791146008,
            log_path="/srv/fleet/stage/run-2/result.txt.log",
            run_id="run-2",
            workers=7,
        )

        assert judged["banner"] is False
        line = verdict.render_verdict(judged)
        assert line.startswith("FLEET-CHECK 3f2a9c1e ")
        assert " exit=1 banner=no tests=885p/2f coverage=statements=99.71% branches=98.42% " in line
        assert line.endswith(
            "log=sedona:/srv/fleet/stage/run-2/result.txt.log run=run-2 workers=7 "
            "slowest=unread ci=unread dispatch=unread vmtests=unread generation=unread "
            "began=unread phases=unread steps=unread registry=unread"
        )
        assert "\n" not in line


class TestHostnameHook:
    def test_the_real_hostname_reader_lowercases_the_machines_name(self) -> None:
        assert _test_hooks._default_hostname() == socket.gethostname().lower()
        assert _test_hooks._default_hostname() != ""
