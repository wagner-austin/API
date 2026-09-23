"""The ``sh`` scripts a Linux node is handed.

TEXT ASSERTIONS, WITH ONE REAL EXECUTION WHERE THE MACHINE ALLOWS. The build,
launch, result and stop scripts need a node with a user systemd manager and
are exercised by a real dispatch to diphtheria (board task 33bb86ce); here
their text is pinned to the properties that made them correct -- fail-fast
prologue, the PATH that finds pipx's poetry, the lingering refusal, the
guarded stop -- so a rewording that loses one fails a test. The two probes
and the reassembly are plain POSIX tools and are RUN when this suite runs
under ``sh``, which is every Linux node and never the Windows hub.
"""

from __future__ import annotations

import base64
import hashlib
import pathlib
import socket
import subprocess
import sys

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, load_json_str

from fleet.core import names
from fleet.core.dialect_linux import (
    OBSERVE_SESSIONS_PYTHON,
    OBSERVE_SESSIONS_SCRIPT,
    PROLOGUE,
    SH_INVOCATION,
    LinuxDialect,
)
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID

DIALECT = LinuxDialect()

TARGET = "/home/corvis/fleet/stage/run-1"

#: One registration document in the shape the harness writes, as a Linux
#: node would hold it: a POSIX cwd and a socket under the user's runtime dir.
SESSION_RECORD: JSONObject = {
    "sessionId": "d31e5228-3269-4a22-a27f-e535cbef1894",
    "pid": 4242,
    "startedAt": 1_789_000_000_000,
    "updatedAt": 1_789_000_030_500,
    "name": "api-7e",
    "nameSource": "derived",
    "cwd": "/home/corvis/PROJECTS/API",
    "messagingSocketPath": "/run/user/1000/claude-cc-msg-abc",
    "version": "2.1.278",
    "status": "idle",
    "pidDomain": "linux:diphtheria",
}


def home_with_sessions(tmp_path: pathlib.Path, *records: JSONObject) -> dict[str, str]:
    """Lay out a home directory holding these session records.

    Args:
        tmp_path: The directory that becomes the home.
        *records: The registration documents, written as ``<pid>.json``.

    Returns:
        The environment that makes ``~`` resolve there under both
        ``os.path.expanduser`` spellings (``HOME`` on POSIX, ``USERPROFILE``
        on Windows), over the parent environment read through the
        monorepo's one permitted reader so the interpreter still starts.
    """
    if records:
        sessions = tmp_path / ".claude" / "sessions"
        sessions.mkdir(parents=True)
        for record in records:
            (sessions / f"{record['pid']}.json").write_text(dump_json_str(record), encoding="utf-8")
    parent = config_test_hooks.get_environment()
    return {**parent, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}


def fields_of(output: str) -> dict[str, str]:
    """Split ``key=value`` probe lines into a mapping.

    Args:
        output: A probe's standard output.

    Returns:
        The pairs, later lines winning.
    """
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, _separator, value = line.partition("=")
        fields[key] = value
    return fields


def test_every_script_begins_with_the_fail_fast_prologue_and_the_user_path() -> None:
    """``set -eu`` so a failed command ends the script with its status, and
    ``~/.local/bin`` first because that is where pipx puts poetry and a
    non-interactive ssh command does not read the profile that adds it."""
    scripts = [
        DIALECT.make_directory_script(TARGET),
        DIALECT.reassemble_script(TARGET),
        _build(workers=4),
        DIALECT.log_tail_script(TARGET, 200),
        DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID),
        DIALECT.result_script(TARGET),
        DIALECT.stop_script(target=TARGET, run_id=DEMO_RUN_ID),
        DIALECT.capacity_probe_script(),
        DIALECT.toolchain_probe_script(),
        DIALECT.observe_sessions_script(),
    ]
    assert PROLOGUE.startswith("set -eu\n")
    assert 'PATH="$HOME/.local/bin:$PATH"' in PROLOGUE
    for script in scripts:
        assert script.startswith(PROLOGUE)


def _build(
    *, path: str = DEMO_PROJECT, install: tuple[tuple[str, ...], ...] = (), workers: int = 6
) -> str:
    """Render the Linux build script for one run under the fixture's roots.

    Args:
        path: The recipe's directory inside the export.
        install: The install steps.
        workers: The worker count.

    Returns:
        The script's text.
    """
    return DIALECT.build_script(
        target=TARGET, path=path, workers=workers, install=install, cache_root="/s/cache"
    )


class TestBuildScript:
    def test_it_runs_the_recipe_in_the_project_with_the_worker_count(self) -> None:
        body = _build()

        assert f"cd '{TARGET}/{DEMO_PROJECT}'\n" in body
        assert "PYTEST_XDIST_AUTO_NUM_WORKERS='6'\n" in body
        assert (
            "export npm_config_cache POETRY_CACHE_DIR PLAYWRIGHT_BROWSERS_PATH "
            "PYTEST_XDIST_AUTO_NUM_WORKERS\n"
        ) in body
        assert f"make check >> '{TARGET}/{names.RESULT_NAME}.log' 2>&1\n" in body

    def test_a_root_project_runs_its_recipe_at_the_export_root(self) -> None:
        body = _build(path="")

        assert body.count(f"cd '{TARGET}'\n") == 2

    def test_it_points_the_three_package_managers_at_the_node_cache(self) -> None:
        body = _build()

        assert "npm_config_cache='/s/cache/npm'\n" in body
        assert "POETRY_CACHE_DIR='/s/cache/pypoetry'\n" in body
        assert "PLAYWRIGHT_BROWSERS_PATH='/s/cache/ms-playwright'\n" in body

    def test_install_steps_run_at_the_root_before_the_recipe_and_end_it_when_they_fail(
        self,
    ) -> None:
        body = _build(install=(("npm", "ci"), ("npm", "rebuild")))
        lines = body.splitlines()

        log = f"{TARGET}/{names.RESULT_NAME}.log"
        root = lines.index(f"cd '{TARGET}'")
        first = lines.index(f"npm ci >> '{log}' 2>&1")
        second = lines.index(f"npm rebuild >> '{log}' 2>&1")
        recipe = lines.index(f"cd '{TARGET}/{DEMO_PROJECT}'")
        assert root < first < second < recipe
        assert lines[first - 2] == f"printf '$ %s\\n' 'npm ci' >> '{log}'"
        assert lines[first - 1] == "set +e"
        assert lines[first + 1] == "status=$?"
        assert lines[first + 2] == "set -e"
        assert lines[first + 3] == (
            f'if [ "$status" -ne 0 ]; then printf \'%s\\n\' "$status" > '
            f"'{TARGET}/{names.RESULT_NAME}'; exit 0; fi"
        )

    def test_a_failing_suite_is_recorded_not_fatal(self) -> None:
        """The recipe runs with -e off and its status captured, so a red suite
        still writes the result file; everything around it stays fail-fast."""
        body = _build()
        lines = body.splitlines()

        recipe = f"make check >> '{TARGET}/{names.RESULT_NAME}.log' 2>&1"
        assert lines.index("set +e") < lines.index(recipe)
        assert lines.index(recipe) + 1 == lines.index("status=$?")
        assert lines.index("status=$?") + 1 == lines.index("set -e")
        assert lines[-1] == f"printf '%s\\n' \"$status\" > '{TARGET}/{names.RESULT_NAME}'"


class TestLogTailScript:
    def test_it_reads_the_last_lines_of_the_transcript_or_nothing(self) -> None:
        body = DIALECT.log_tail_script(TARGET, 200)

        assert body == (
            f"{PROLOGUE}if [ -f '{TARGET}/{names.RESULT_NAME}.log' ]; then "
            f"tail -n 200 '{TARGET}/{names.RESULT_NAME}.log'; fi\n"
        )


class TestLaunchScript:
    def test_it_refuses_before_starting_when_lingering_is_off(self) -> None:
        """A --user unit dies with the session that started it unless the
        user lingers, so the script checks first and names the fix."""
        body = DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID)
        lines = body.splitlines()

        check = next(index for index, line in enumerate(lines) if "loginctl show-user" in line)
        start = next(index for index, line in enumerate(lines) if "systemd-run" in line)
        assert check < start
        assert "sudo loginctl enable-linger $user" in body
        assert "exit 1" in body

    def test_it_starts_a_transient_user_unit_named_for_the_run_and_proves_it(self) -> None:
        body = DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID)
        unit = names.task_name(DEMO_RUN_ID)

        assert f"systemd-run --user --unit='{unit}' --collect --quiet" in body
        assert f"--property=WorkingDirectory='{TARGET}'" in body
        assert f"/bin/sh '{DIALECT.script_path(TARGET, names.BUILD_STEM)}'" in body
        assert body.rstrip().endswith("printf 'launched\\n'")
        assert "make check" not in body

    def test_the_unit_name_is_the_one_the_stop_script_stops(self) -> None:
        unit = names.task_name(DEMO_RUN_ID)

        assert unit in DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID)
        stopped = DIALECT.stop_script(target=TARGET, run_id=DEMO_RUN_ID)

        assert f"systemctl --user is-active --quiet '{unit}'" in stopped
        assert f"systemctl --user stop '{unit}'" in stopped


class TestResultAndStopScripts:
    def test_the_result_script_prints_status_and_mtime_or_nothing(self) -> None:
        body = DIALECT.result_script(TARGET)
        result = f"{TARGET}/{names.RESULT_NAME}"

        assert f"if [ -f '{result}' ]; then" in body
        assert f"stat -c %Y '{result}'" in body
        assert 'printf \'%s %s\\n\' "$code" "$written"' in body

    def test_the_stop_script_is_guarded_and_always_reports(self) -> None:
        """systemctl stop of a collected unit exits 5; the cancel must still
        close the row, so the stop is guarded by is-active."""
        body = DIALECT.stop_script(target=TARGET, run_id=DEMO_RUN_ID)
        unit = names.task_name(DEMO_RUN_ID)

        assert body.count("systemctl --user stop") == 1
        assert body.rstrip().endswith(f"printf 'stopped {unit}\\n'")

    def test_the_stop_reads_no_process_id(self) -> None:
        """Stopping a unit stops its control group, the whole tree the build
        started, so the id the Windows build records is neither written nor
        read here, and the build script writes no such file."""
        assert names.PID_NAME not in DIALECT.stop_script(target=TARGET, run_id=DEMO_RUN_ID)
        assert names.PID_NAME not in _build(workers=4)


class TestTransportShape:
    def test_scripts_are_sh_and_run_through_bin_sh_by_path(self) -> None:
        assert DIALECT.script_path("/s", "build") == "/s/build.sh"
        assert DIALECT.invocation() == SH_INVOCATION == ("/bin/sh",)

    def test_the_write_command_creates_the_parent_and_streams_stdin(self) -> None:
        command = DIALECT.write_command("/s/run-1/x.sh")

        assert command == "mkdir -p \"$(dirname '/s/run-1/x.sh')\" && cat > '/s/run-1/x.sh'"

    def test_the_directory_script_is_mkdir_p(self) -> None:
        assert DIALECT.make_directory_script("/s/run-1").endswith("mkdir -p '/s/run-1'\n")

    def test_the_reassembly_decodes_digests_and_does_not_extract(self) -> None:
        body = DIALECT.reassemble_script(TARGET)

        encoded = f"{TARGET}/{names.ENCODED_NAME}"
        assert f"base64 -d '{encoded}' > '{TARGET}/{names.ARCHIVE_NAME}'" in body
        assert f"sha256sum '{TARGET}/{names.ARCHIVE_NAME}' | cut -d ' ' -f 1" in body
        assert "tar" not in body

    def test_echo_is_printf_of_a_quoted_literal(self) -> None:
        assert DIALECT.echo_command("installing make") == "printf '%s\\n' 'installing make'"

    def test_the_toolchain_probe_asks_python3_and_reports_it_as_python(self) -> None:
        """The contract names the interpreter ``python``; the Makefile prologue
        calls ``python3`` on Linux, so that is what is asked."""
        body = DIALECT.toolchain_probe_script()

        assert "report python python3\n" in body
        for tool in ("poetry", "git", "make", "tar", "apt-get", "pipx"):
            assert f"report {tool} {tool}\n" in body
        assert "winget" not in body
        assert "choco" not in body

    def test_the_fleet_directory_is_the_literal_home_path(self) -> None:
        """The write command single-quotes the path, so ``~`` would be a
        directory called ``~``; the account's home is spelled out."""
        assert DIALECT.fleet_directory("corvis") == "/home/corvis/.fleet"
        assert DIALECT.script_path(DIALECT.fleet_directory("corvis"), "observe-sessions") == (
            "/home/corvis/.fleet/observe-sessions.sh"
        )

    def test_the_observe_script_feeds_python3_through_a_quoted_heredoc(self) -> None:
        """sh cannot join files into one JSON array that survives an empty
        directory or a quote in a field; python3 can, and it is on every
        Linux node by the toolchain contract (board task cd5010c4)."""
        body = DIALECT.observe_sessions_script()

        assert body == OBSERVE_SESSIONS_SCRIPT
        assert body == f"{PROLOGUE}python3 - <<'PY'\n{OBSERVE_SESSIONS_PYTHON}PY\n"
        assert "'.claude', 'sessions'" in OBSERVE_SESSIONS_PYTHON
        assert "if os.path.isdir(directory):" in OBSERVE_SESSIONS_PYTHON
        assert "sorted(os.listdir(directory))" in OBSERVE_SESSIONS_PYTHON
        assert "'platform': sys.platform" in OBSERVE_SESSIONS_PYTHON
        assert "'hostname': socket.gethostname().lower()" in OBSERVE_SESSIONS_PYTHON
        assert "separators=(',', ':')" in OBSERVE_SESSIONS_PYTHON


class TestObserveBodyForReal:
    """The python3 body of the observe script, run by this interpreter.

    The sh around it is one heredoc and runs only where sh exists (the class
    below); the body is what reads the directory and builds the document, and
    it runs the same under every interpreter the fleet has, so it is executed
    HERE, on the hub too, against a home laid out on disk.
    """

    def run_body(self, environment: dict[str, str]) -> JSONObject:
        """Run the body exactly as the heredoc feeds it, and decode its line.

        Args:
            environment: The environment, with the home to read.

        Returns:
            The decoded document.
        """
        completed = subprocess.run(
            [sys.executable, "-"],
            input=OBSERVE_SESSIONS_PYTHON,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
        document: JSONValue = load_json_str(completed.stdout)
        if not isinstance(document, dict):
            raise AssertionError(f"the script printed {type(document).__name__}, not an object")
        return document

    def test_it_reports_every_record_verbatim_under_the_platform_and_lowercased_host(
        self, tmp_path: pathlib.Path
    ) -> None:
        second: JSONObject = {
            **SESSION_RECORD,
            "pid": 31868,
            "sessionId": "06e0b3cd-6de1-45ec-8aa2-c3b48a7f60fc",
        }

        document = self.run_body(home_with_sessions(tmp_path, second, SESSION_RECORD))

        assert document == {
            "platform": sys.platform,
            "hostname": socket.gethostname().lower(),
            "records": [second, SESSION_RECORD],
        }

    def test_it_reports_zero_records_for_a_home_that_never_ran_claude_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        """diphtheria on 2026-09-21: no ``~/.claude`` at all. That is a node
        with no sessions, not a fault."""
        document = self.run_body(home_with_sessions(tmp_path))

        assert document["records"] == []
        assert document["platform"] == sys.platform


@pytest.mark.skipif(sys.platform == "win32", reason="the scripts are sh; run them where it exists")
class TestForRealUnderSh:
    """The probes and the reassembly, executed by /bin/sh on this machine."""

    def run_script(self, tmp_path: pathlib.Path, body: str) -> str:
        """Write a script under tmp_path and run it by path.

        Args:
            tmp_path: Where to write it.
            body: The script's text.

        Returns:
            Its standard output.
        """
        script = tmp_path / "script.sh"
        script.write_text(body, encoding="utf-8")
        completed = subprocess.run(
            [*SH_INVOCATION, str(script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    def test_the_capacity_probe_reports_the_three_numbers(self, tmp_path: pathlib.Path) -> None:
        fields = fields_of(self.run_script(tmp_path, DIALECT.capacity_probe_script()))

        assert set(fields) == {"free_ram_gb", "free_disk_gb", "logical_cores"}
        assert float(fields["free_ram_gb"]) > 0.0
        assert float(fields["free_disk_gb"]) > 0.0
        assert int(fields["logical_cores"]) >= 1

    def test_the_toolchain_probe_reports_every_tool(self, tmp_path: pathlib.Path) -> None:
        fields = fields_of(self.run_script(tmp_path, DIALECT.toolchain_probe_script()))

        assert set(fields) == {"python", "poetry", "git", "make", "tar", "apt-get", "pipx"}
        assert fields["python"].startswith("yes=Python 3.")
        for value in fields.values():
            assert value.startswith(("yes=", "no="))

    def test_the_reassembly_reproduces_the_bytes_and_their_digest(
        self, tmp_path: pathlib.Path
    ) -> None:
        payload = b"\x1f\x8b" + bytes(range(256)) * 3
        target = tmp_path / "run-1"
        target.mkdir()
        (target / names.ENCODED_NAME).write_text(base64.b64encode(payload).decode("ascii"))

        output = self.run_script(tmp_path, DIALECT.reassemble_script(str(target)))

        assert output.strip() == hashlib.sha256(payload).hexdigest()
        assert (target / names.ARCHIVE_NAME).read_bytes() == payload

    def test_the_observe_script_reports_the_records_under_the_home(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The whole script under sh: the prologue, the heredoc, the body."""
        home = tmp_path / "home"
        home.mkdir()
        script = tmp_path / "observe-sessions.sh"
        script.write_text(DIALECT.observe_sessions_script(), encoding="utf-8")

        completed = subprocess.run(
            [*SH_INVOCATION, str(script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=home_with_sessions(home, SESSION_RECORD),
        )

        assert completed.returncode == 0, completed.stderr
        assert load_json_str(completed.stdout) == {
            "platform": sys.platform,
            "hostname": socket.gethostname().lower(),
            "records": [SESSION_RECORD],
        }
