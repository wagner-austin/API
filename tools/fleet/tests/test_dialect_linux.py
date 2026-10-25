"""The ``sh`` scripts a Linux node is handed.

TEXT ASSERTIONS, WITH ONE REAL EXECUTION WHERE THE MACHINE ALLOWS. The build,
launch, result and stop scripts need a node with MCPs' fleet-check-unit and
are exercised by a real dispatch to diphtheria (board tasks 33bb86ce and MCPs
bc8c445c); here their text is pinned to the properties that made them correct
-- fail-fast prologue, the PATH that finds pipx's poetry, the lingering
refusal, the helper's start and stop -- so a rewording that loses one fails a
test. The two probes
and the reassembly are plain POSIX tools and are RUN when this suite runs
under ``sh``, which is every Linux node and never the Windows hub.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess

import pytest

from fleet.contracts.check_unit import CHECK_UNIT
from fleet.contracts.source import InstallStep
from fleet.core import names, tool_caches
from fleet.core.dialect_linux import (
    PROLOGUE,
    SH_INVOCATION,
    LinuxDialect,
)
from fleet.core.linux_unit_end import unit_end_path
from fleet.core.names import CACHE_VARIABLE
from fleet.core.phase_markers import SH_UTC_STAMP
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID

DIALECT = LinuxDialect()

TARGET = "/home/corvis/fleet/stage/run-1"


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
        DIALECT.digest_script(TARGET),
        _build(workers=4),
        DIALECT.log_tail_script(TARGET, 200),
        DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID, elevated=False),
        DIALECT.result_script(target=TARGET, run_id=DEMO_RUN_ID),
        DIALECT.stop_script(target=TARGET, run_id=DEMO_RUN_ID),
        DIALECT.capacity_probe_script(),
        DIALECT.toolchain_probe_script(),
        DIALECT.observe_sessions_script(),
    ]
    assert PROLOGUE.startswith("set -eu\n")
    assert 'PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' in PROLOGUE
    for script in scripts:
        assert script.startswith(PROLOGUE)


def _step(*argv: str) -> InstallStep:
    """An install step of the install phase."""
    return InstallStep(phase="install", argv=argv)


def _build(
    *,
    path: str = DEMO_PROJECT,
    install: tuple[InstallStep, ...] = (),
    workers: int = 6,
    loopback_unit: str | None = None,
) -> str:
    """Render the Linux build script for one run under the fixture's roots.

    Args:
        path: The recipe's directory inside the export.
        install: The install steps.
        workers: The worker count.
        loopback_unit: The check's unit for a loopback-only project.

    Returns:
        The script's text.
    """
    return DIALECT.build_script(
        target=TARGET,
        path=path,
        workers=workers,
        install=install,
        cache_root="/s/cache",
        isolated_docker=False,
        elevated=False,
        agent="opus-demo-0929",
        loopback_unit=loopback_unit,
    )


class TestBuildScript:
    def test_it_runs_the_recipe_in_the_project_with_the_worker_count(self) -> None:
        body = _build()

        assert f"cd '{TARGET}/{DEMO_PROJECT}'\n" in body
        assert "PYTEST_XDIST_AUTO_NUM_WORKERS='6'\n" in body
        assert "CORVIS_TEST_MAX_WORKERS='6'\n" in body
        assert f"CORVIS_FLEET_WORKSPACE={DEMO_PROJECT}\n" in body
        assert "CORVIS_FLEET_WORKSPACE=.\n" in _build(path="")
        assert f"CORVIS_FLEET_REGISTRY='{names.registry_directory(TARGET)}'\n" in body
        assert (
            "export npm_config_cache POETRY_CACHE_DIR POETRY_VIRTUALENVS_IN_PROJECT "
            "POETRY_KEYRING_ENABLED PLAYWRIGHT_BROWSERS_PATH "
            "PYTEST_XDIST_AUTO_NUM_WORKERS CORVIS_TEST_MAX_WORKERS "
            "BOARD_AGENT_LABEL "
            "CORVIS_FLEET_CACHE CORVIS_FLEET_WORKSPACE CORVIS_FLEET_REGISTRY\n"
        ) in body
        assert f"make check >> '{TARGET}/{names.RESULT_NAME}.log' 2>&1\n" in body

    def test_it_exports_the_submitter_s_label_and_refuses_one_outside_the_grammar(
        self,
    ) -> None:
        """MCPs board task 6c4516af A4: a fleet hold the suite takes is
        attributed to whoever asked for the run."""
        body = _build()

        assert "BOARD_AGENT_LABEL='opus-demo-0929'\n" in body
        with pytest.raises(ValueError, match="'Opus Demo' is outside the board's kebab-case"):
            DIALECT.build_script(
                target=TARGET,
                path=DEMO_PROJECT,
                workers=1,
                install=(),
                cache_root="/s/cache",
                isolated_docker=False,
                elevated=False,
                agent="Opus Demo",
                loopback_unit=None,
            )

    def test_a_root_project_runs_its_recipe_at_the_export_root(self) -> None:
        body = _build(path="")

        assert body.count(f"cd '{TARGET}'\n") == 2

    def test_it_points_npm_and_poetry_at_the_current_generation_and_playwright_at_the_cache(
        self,
    ) -> None:
        """MCPs board task 346abc45: the generation lines run first, and
        the virtualenv is the project's."""
        body = _build()

        generation = "".join(
            f"{line}\n"
            for line in tool_caches.sh_build_lines(
                tools="/s/cache/tools", marker=f"{TARGET}/{names.GENERATION_MARKER_NAME}", run_as=""
            )
        )
        assert body.startswith(PROLOGUE + generation)
        assert 'npm_config_cache="$tool_cache/npm"\n' in body
        assert 'POETRY_CACHE_DIR="$tool_cache/pypoetry"\n' in body
        assert "POETRY_VIRTUALENVS_IN_PROJECT='true'\n" in body
        assert 'PLAYWRIGHT_BROWSERS_PATH="$tool_cache/ms-playwright"\n' in body

    def test_it_turns_poetry_s_keyring_off_before_any_install(self) -> None:
        """MCPs board task c837fd8d: left on, poetry blocks on the D-Bus
        SecretService on a headless node, and the build times out inside
        ``poetry install``."""
        lines = _build(install=(_step("npm", "ci"),)).splitlines()

        keyring = lines.index("POETRY_KEYRING_ENABLED='false'")
        assert keyring < lines.index(f"printf '$ %s\\n' 'npm ci' >> '{names.log_path(TARGET)}'")

    def test_install_steps_run_at_the_root_before_the_recipe_and_end_it_when_they_fail(
        self,
    ) -> None:
        rebuild = InstallStep(phase="workspace-build", argv=("npm", "rebuild"))
        body = _build(install=(_step("npm", "ci"), rebuild))
        lines = body.splitlines()

        log = f"{TARGET}/{names.RESULT_NAME}.log"
        root = lines.index(f"cd '{TARGET}'")
        first = lines.index(f"npm ci >> '{log}' 2>&1")
        second = lines.index(f"npm rebuild >> '{log}' 2>&1")
        recipe = lines.index(f"cd '{TARGET}/{DEMO_PROJECT}'")
        assert root < first < second < recipe
        assert lines[first - 4 : first] == [
            f"printf '$ %s\\n' 'npm ci' >> '{log}'",
            'phase_started="$(date +%s)"',
            f"printf 'fleet-phase %s started %s\\n' 'install' {SH_UTC_STAMP} >> '{log}'",
            "set +e",
        ]
        assert lines[first + 1 : first + 4] == [
            "status=$?",
            "set -e",
            f"printf 'fleet-phase %s ended %s after %s s, exit %s\\n' 'install' {SH_UTC_STAMP} "
            f'"$(( $(date +%s) - phase_started ))" "$status" >> \'{log}\'',
        ]
        assert lines[first + 4] == (
            f'if [ "$status" -ne 0 ]; then printf \'%s\\n\' "$status" > '
            f"'{TARGET}/{names.RESULT_NAME}'; exit 0; fi"
        )
        assert f"'workspace-build' {SH_UTC_STAMP}" in lines[second - 2]

    def test_the_build_names_the_node_s_cache_for_install_steps_that_keep_state(self) -> None:
        """MCPs' fleet-prepare keeps a prepared tree under it (MCPs board
        task 74b13c20)."""
        lines = _build().splitlines()

        assert f"{CACHE_VARIABLE}='/s/cache'" in lines
        assert any(line.startswith("export ") and CACHE_VARIABLE in line for line in lines)

    def test_a_failing_suite_is_recorded_not_fatal(self) -> None:
        """The recipe runs with -e off and its status captured as the phase
        ``check``, so a red suite still writes the result file; everything
        around it stays fail-fast."""
        body = _build()
        lines = body.splitlines()

        recipe = f"make check >> '{TARGET}/{names.RESULT_NAME}.log' 2>&1"
        assert lines[lines.index(recipe) - 2].startswith("printf 'fleet-phase %s started %s")
        assert "'check'" in lines[lines.index(recipe) - 2]
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
        """The unit is given the account's runtime directory and bus, which
        end with the session that started it unless the user lingers, so the
        script checks first and names the fix."""
        body = DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID, elevated=False)
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
        body = DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID, elevated=False)
        unit = names.task_name(DEMO_RUN_ID)
        build = DIALECT.script_path(TARGET, names.BUILD_STEM)
        start = f"\n{CHECK_UNIT} start '{unit}' '{TARGET}' '{build}' '{unit_end_path(TARGET)}'\n"

        assert start in body
        assert CHECK_UNIT == "sudo -n /usr/local/sbin/fleet-check-unit"
        assert "--user" not in body
        assert body.rstrip().endswith("printf 'launched\\n'")
        assert "make check" not in body

    def test_an_elevated_launch_or_build_is_refused_rather_than_rendered(self) -> None:
        """No linux node declares an elevated runner (MCPs board task
        a98d7083), and a build rendered anyway would lack the rights its
        suite was written for."""
        with pytest.raises(ValueError, match=r"requires the elevated tag, and a linux node has no"):
            DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID, elevated=True)
        with pytest.raises(ValueError, match=r"requires the elevated tag, and a linux node has no"):
            DIALECT.build_script(
                target=TARGET,
                path="execution",
                workers=1,
                install=(),
                cache_root="/s/cache",
                isolated_docker=False,
                elevated=True,
                agent="opus-demo-0929",
                loopback_unit=None,
            )

    def test_the_unit_name_is_the_one_the_stop_script_stops(self) -> None:
        unit = names.task_name(DEMO_RUN_ID)

        assert unit in DIALECT.launch_script(target=TARGET, run_id=DEMO_RUN_ID, elevated=False)
        stopped = DIALECT.stop_script(target=TARGET, run_id=DEMO_RUN_ID)

        assert f"{CHECK_UNIT} stop '{unit}'\n" in stopped


class TestResultAndStopScripts:
    def test_the_result_script_prints_status_and_mtime_or_nothing(self) -> None:
        body = DIALECT.result_script(target=TARGET, run_id=DEMO_RUN_ID)
        result = f"{TARGET}/{names.RESULT_NAME}"

        assert f"if [ -f '{result}' ]; then" in body
        assert f"stat -c %Y '{result}'" in body
        assert 'printf \'%s %s\\n\' "$code" "$written"' in body

    def test_the_stop_script_stops_through_the_helper_and_always_reports(self) -> None:
        """systemctl stop of a collected unit exits 5; the cancel must still
        close the row, so the helper's stop, which exits 0 for a unit no
        longer active, is the script's one stop."""
        body = DIALECT.stop_script(target=TARGET, run_id=DEMO_RUN_ID)
        unit = names.task_name(DEMO_RUN_ID)

        assert body == f"{PROLOGUE}{CHECK_UNIT} stop '{unit}'\nprintf 'stopped {unit}\\n'\n"

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
        assert DIALECT.make_directory_script("/s/run-1").endswith("mkdir -p /s/run-1\n")
        assert DIALECT.make_directory_script("/s/run 1").endswith("mkdir -p '/s/run 1'\n")

    def test_the_digest_script_digests_the_landed_archive_and_does_not_extract(self) -> None:
        body = DIALECT.digest_script(TARGET)

        assert body == (f"{PROLOGUE}sha256sum '{TARGET}/{names.ARCHIVE_NAME}' | cut -d ' ' -f 1\n")

    def test_echo_is_printf_of_a_quoted_literal(self) -> None:
        assert DIALECT.echo_command("installing make") == "printf '%s\\n' 'installing make'"

    def test_the_toolchain_probe_asks_python3_and_reports_it_as_python(self) -> None:
        """The contract names the interpreter ``python``; the Makefile prologue
        calls ``python3`` on Linux, so that is what is asked."""
        body = DIALECT.toolchain_probe_script()

        assert "report python python3\n" in body
        assert "report go go version\n" in body
        # cargo is asked with the Rust check's toolchain (test_dialect_linux_rust).
        for tool in ("poetry", "git", "make", "node", "ffmpeg", "tar", "apt-get", "pipx"):
            assert f"report {tool} {tool}\n" in body
        assert "printf 'cxx=yes=%s\\n' \"$(g++ -dumpfullversion)\"\n" in body
        # The docker line asks execdocker's own socket, never the PATH's
        # docker, which is the stack's daemon on diphtheria (6c4516af).
        assert "sudo -n -u execdocker docker -H" in body
        assert "*name=rootless*)\n" in body
        assert "sudo -n -u execdocker docker compose version" in body
        assert "sudo -n -u execdocker docker buildx version" in body
        assert "winget" not in body
        assert "choco" not in body

    def test_the_fleet_directory_is_the_literal_home_path(self) -> None:
        """The write command single-quotes the path, so ``~`` would be a
        directory called ``~``; the account's home is spelled out."""
        assert DIALECT.fleet_directory("corvis") == "/home/corvis/.fleet"
        assert DIALECT.script_path(DIALECT.fleet_directory("corvis"), "observe-sessions") == (
            "/home/corvis/.fleet/observe-sessions.sh"
        )


@pytest.mark.host_linux
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

    def test_the_toolchain_probe_reports_every_tool(self, tmp_path: pathlib.Path) -> None:
        fields = fields_of(self.run_script(tmp_path, DIALECT.toolchain_probe_script()))

        assert set(fields) == {
            "python",
            "poetry",
            "git",
            "make",
            "node",
            "ffmpeg",
            "go",
            "tar",
            "cargo",
            "cxx",
            "docker",
            "stack",
            "gpu",
            "testdb",
            "hooks",
            "chrome",
            "pdftotext",
            "apt-get",
            "pipx",
        }
        assert fields["python"].startswith("yes=Python 3.")
        for value in fields.values():
            assert value.startswith(("yes=", "no="))

    def test_the_cxx_line_carries_g_plus_plus_s_bare_version(self, tmp_path: pathlib.Path) -> None:
        """A g++ ahead on PATH answers -dumpfullversion as diphtheria's did
        (13.3.0, 2026-09-27), and the probe line carries exactly that, which
        is the shape :func:`fleet.contracts.capability.measured` reads."""
        tools = tmp_path / "tools"
        tools.mkdir()
        fake = tools / "g++"
        fake.write_bytes(b'#!/bin/sh\n[ "$1" = "-dumpfullversion" ] && echo 13.3.0\n')
        fake.chmod(0o755)
        body = DIALECT.toolchain_probe_script().replace(
            PROLOGUE, PROLOGUE + f"PATH='{tools.as_posix()}':$PATH\n", 1
        )
        fields = fields_of(self.run_script(tmp_path, body))

        assert fields["cxx"] == "yes=13.3.0"

    def test_the_digest_script_prints_the_landed_bytes_digest_and_leaves_them(
        self, tmp_path: pathlib.Path
    ) -> None:
        payload = b"\x1f\x8b" + bytes(range(256)) * 3
        target = tmp_path / "run-1"
        target.mkdir()
        (target / names.ARCHIVE_NAME).write_bytes(payload)

        output = self.run_script(tmp_path, DIALECT.digest_script(str(target)))

        assert output == f"{hashlib.sha256(payload).hexdigest()}\n"
        assert (target / names.ARCHIVE_NAME).read_bytes() == payload
