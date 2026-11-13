"""The launch records the registry commit beside the run, and the tail read prints it back.

MCPs board task 1fd01f09: job 0fa14fab's verdict said nothing of which
registry gave its install steps. The launch script is executed here by the
real ``sh`` with only the account's tools answered from ``PATH``
(``tests/test_linux_unit_end.py`` runs the rest of it), and the tail read is
executed against a real run directory, so the line the verdict reads is the
line ``sh`` prints.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.check_unit import CHECK_UNIT
from fleet.core import dialect, names, run_registry, verdict
from fleet.core.check_results import sh_failures_line, sh_results_digest_line
from fleet.core.dialect import PROLOGUE, SH_INVOCATION
from fleet.core.linux_unit_end import unit_end_path
from fleet.core.run_summary import UNREAD
from tests._roll_reads import NEXT_COMMIT, ROLLED_COMMIT
from tests.conftest import DEMO_RUN_ID

TARGET = "/home/corvis/fleet/stage/run-1"


def _sh(body: str, directory: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run a script's text with the real ``sh``.

    Args:
        body: The script.
        directory: Where to write it.

    Returns:
        The finished process.
    """
    script = directory / "script.sh"
    script.write_text(body, encoding="utf-8")
    return subprocess.run(
        [*SH_INVOCATION, str(script)], capture_output=True, text=True, check=False, timeout=60
    )


class TestTheLaunchRecordsTheCommit:
    def test_it_writes_the_commit_beside_the_run_before_the_unit_starts(self) -> None:
        body = run_registry.launch_script(
            target=TARGET, run_id=DEMO_RUN_ID, registry_commit=ROLLED_COMMIT
        )
        recorded = f"printf '%s\\n' '{ROLLED_COMMIT}' > '{TARGET}/registry-commit.txt'\n"

        assert recorded in body
        assert body.index(recorded) < body.index(f"{CHECK_UNIT} start ")
        assert run_registry.registry_commit_path(TARGET) == f"{TARGET}/registry-commit.txt"

    def test_a_fleet_run_from_a_checkout_records_the_working_tree(self) -> None:
        body = run_registry.launch_script(
            target=TARGET, run_id=DEMO_RUN_ID, registry_commit=run_registry.WORKING_TREE
        )

        assert f"printf '%s\\n' 'working-tree' > '{TARGET}/registry-commit.txt'\n" in body

    @pytest.mark.parametrize("given", ["84bc5ff0b6a9", ROLLED_COMMIT.upper(), "main", ""])
    def test_anything_else_is_refused_before_a_script_is_written(self, given: str) -> None:
        with pytest.raises(ValueError, match="a run's registry commit is a full lowercase commit"):
            run_registry.launch_script(target=TARGET, run_id=DEMO_RUN_ID, registry_commit=given)


class TestTheLaunchStartsTheUnit:
    def test_it_refuses_before_starting_when_lingering_is_off(self) -> None:
        """The unit is given the account's runtime directory and bus, which
        end with the session that started it unless the user lingers, so the
        script checks first and names the fix."""
        body = run_registry.launch_script(
            target=TARGET, run_id=DEMO_RUN_ID, registry_commit=ROLLED_COMMIT
        )
        lines = body.splitlines()

        check = next(index for index, line in enumerate(lines) if "loginctl show-user" in line)
        start = next(index for index, line in enumerate(lines) if line.startswith(CHECK_UNIT))
        assert check < start
        assert "sudo loginctl enable-linger $user" in body
        assert "exit 1" in body

    def test_it_starts_the_build_through_the_helper_in_dispatchs_share_and_proves_it(
        self,
    ) -> None:
        """MCPs board task bc8c445c: a --user unit sat under user.slice, which
        neither the owner's nor dispatch's slice reached; the helper starts a
        system unit in the checks' slice inside system-dispatch.slice."""
        body = run_registry.launch_script(
            target=TARGET, run_id=DEMO_RUN_ID, registry_commit=ROLLED_COMMIT
        )
        unit = names.task_name(DEMO_RUN_ID)
        build = dialect.script_path(TARGET, names.BUILD_STEM)
        start = f"\n{CHECK_UNIT} start '{unit}' '{TARGET}' '{build}' '{unit_end_path(TARGET)}'\n"

        assert body.startswith(PROLOGUE)
        assert start in body
        assert CHECK_UNIT == "sudo -n /usr/local/sbin/fleet-check-unit"
        assert "--user" not in body
        assert body.rstrip().endswith("printf 'launched\\n'")
        assert "make check" not in body

    def test_the_unit_name_is_the_one_the_stop_script_stops(self) -> None:
        unit = names.task_name(DEMO_RUN_ID)

        assert unit in run_registry.launch_script(
            target=TARGET, run_id=DEMO_RUN_ID, registry_commit=ROLLED_COMMIT
        )
        assert f"{CHECK_UNIT} stop '{unit}'\n" in dialect.stop_script(DEMO_RUN_ID)


class TestTheTailReadPrintsTheCommit:
    def test_its_text_reads_the_transcript_then_the_record(self) -> None:
        assert run_registry.log_tail_script(TARGET, 200) == (
            f"{PROLOGUE}if [ -f '{TARGET}/{names.RESULT_NAME}.log' ]; then "
            f"tail -n 200 '{TARGET}/{names.RESULT_NAME}.log'; fi\n"
            f"if [ -f '{TARGET}/registry-commit.txt' ]; then "
            f"commit=\"$(cat '{TARGET}/registry-commit.txt')\"; "
            f"printf 'FLEET_REGISTRY: %s\\n' \"$commit\"; fi\n"
            f"{sh_results_digest_line(TARGET)}\n"
            f"{sh_failures_line(TARGET)}\n"
        )

    def test_sh_prints_the_last_lines_and_the_commit(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "run-1"
        target.mkdir()
        (target / f"{names.RESULT_NAME}.log").write_text("one\ntwo\nthree\n", encoding="utf-8")
        (target / run_registry.REGISTRY_COMMIT_NAME).write_text(
            f"{ROLLED_COMMIT}\n", encoding="utf-8"
        )

        completed = _sh(run_registry.log_tail_script(target.as_posix(), 2), tmp_path)

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == f"two\nthree\nFLEET_REGISTRY: {ROLLED_COMMIT}\n"
        assert run_registry.read_registry(completed.stdout) == ROLLED_COMMIT

    def test_a_run_launched_before_the_record_prints_no_commit(
        self, tmp_path: pathlib.Path
    ) -> None:
        target = tmp_path / "run-1"
        target.mkdir()
        (target / f"{names.RESULT_NAME}.log").write_text("only\n", encoding="utf-8")

        completed = _sh(run_registry.log_tail_script(target.as_posix(), 5), tmp_path)

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == "only\n"
        assert run_registry.read_registry(completed.stdout) == UNREAD


class TestTheVerdictNamesTheCommit:
    def test_the_last_registry_line_is_the_one_read(self) -> None:
        tail = f"FLEET_REGISTRY: {NEXT_COMMIT}\nx\nFLEET_REGISTRY: {ROLLED_COMMIT}\n"

        assert run_registry.read_registry(tail) == ROLLED_COMMIT

    def test_a_working_tree_record_is_read_as_such(self) -> None:
        assert run_registry.read_registry("FLEET_REGISTRY: working-tree\n") == "working-tree"

    def test_a_line_naming_no_commit_is_not_read(self) -> None:
        assert run_registry.read_registry("FLEET_REGISTRY: 84bc5ff\n") == UNREAD

    def test_the_line_ends_with_the_commit_the_launch_recorded(self) -> None:
        judged = verdict.judge(
            job_id="0fa14fab-0000-4000-8000-000000000000",
            project="MCPs/execution-twin",
            sha="6dd4a3af9" + "0" * 31,
            node="pendragon",
            exit_code=0,
            tail=f"=== ALL CHECKS PASSED ===\nFLEET_REGISTRY: {ROLLED_COMMIT}\n",
            ended_unix=1793403600,
            log_path="/home/corvis/fleet/stage/run-1/result.txt.log",
            results_path="/home/corvis/fleet/stage/logs/run-1.results",
            run_id="run-1",
            workers=4,
        )

        assert judged["registry"] == ROLLED_COMMIT
        assert verdict.render_verdict(judged).endswith(
            f" steps=unread registry={ROLLED_COMMIT} artifact=unread mutation=unread "
            "results=pendragon:/home/corvis/fleet/stage/logs/run-1.results"
        )
