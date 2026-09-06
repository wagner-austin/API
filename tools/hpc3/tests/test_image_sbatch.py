"""Tests for the rendered build job.

This renderer exists because the job was hand written once per image version
and two defects entered through the copying. Both have an assertion here, so
neither can return: a smoke command reaching for a host path, and a failure
message naming a cause the job never established.
"""

from __future__ import annotations

import pathlib
import subprocess

from hpc3.core.image_exec import APPTAINER_MODULE
from hpc3.core.image_layout import SELFCHECK_NAME, SPEC_DIR
from hpc3.core.image_sbatch import (
    BUILD_PARTITION,
    CACHE_DIR,
    TMP_DIR,
    render_build_sbatch,
)
from tests.bash_discovery import bash_beside_git, is_wsl_launcher, posix_bash

_SMOKE = "/opt/env/bin/python -m pkg.probe --device cpu --out /tmp/probe.json"

#: A smoke command carrying quotes, which is what any assertion about a string
#: looks like. Rendered raw into a double-quoted echo this still PARSES --
#: bash concatenates the adjacent quoted runs -- and prints the command with
#: its quotes stripped, so the line naming a failing command is not the
#: command.
_QUOTED_SMOKE = "/opt/env/bin/python -c \"assert PROBE_LABEL == 'gpt2-tiny', PROBE_LABEL\""

#: A smoke command whose own double quotes wrap a shell metacharacter. Valid
#: on the `if` line, where the quotes are the shell's to parse; a hard syntax
#: error in a raw echo, where those quotes close against the echo's own and
#: leave `((1,2))` bare. Verified in both directions before being written
#: down: `bash -n` accepts the if-line rendering and rejects the raw echo.
_METACHAR_SMOKE = '/opt/env/bin/python -c "print((1,2))"'


def _render(smoke_commands: list[str] | None = None) -> str:
    """Render a job with the given smoke commands."""
    return render_build_sbatch(
        image_name="abl.sif",
        job_name="img.abl-sif-v9",
        image_dir="/pub/wagnera3/images/v9",
        env_prefix="/opt/env",
        smoke_commands=[] if smoke_commands is None else smoke_commands,
    )


class TestThePreemptionGuard:
    """The free partition is preemptible and this build is long."""

    def test_it_requeues(self) -> None:
        # Without this a build dies to preemption and stays dead; one died at
        # 26 seconds.
        assert "#SBATCH --requeue" in _render()

    def test_it_asks_for_the_free_partition(self) -> None:
        assert f"#SBATCH -p {BUILD_PARTITION}" in _render()

    def test_it_asks_for_no_gpu(self) -> None:
        # `apptainer build` has no use for one, and asking queues the build
        # behind jobs that do.
        assert "--gres" not in _render()


class TestTheEnvironment:
    """What the build needs before it can run at all."""

    def test_it_loads_the_apptainer_module_the_rest_of_the_package_uses(self) -> None:
        # A fourth spelling of the module name is how preflight and the job
        # end up verifying different worlds.
        assert f"module load {APPTAINER_MODULE}" in _render()

    def test_it_sources_modules_before_being_strict(self) -> None:
        rendered = _render()
        source_at = rendered.index("source /etc/profile.d/rcic-modules.sh")
        module_at = rendered.index(f"module load {APPTAINER_MODULE}")

        assert source_at < module_at

    def test_it_redirects_the_apptainer_caches_off_the_home_volume(self) -> None:
        # $HOME is 50 GB on HPC3 and a multi-gigabyte build fills it, failing
        # as though the build were broken rather than the disk.
        rendered = _render()

        assert f"export APPTAINER_CACHEDIR={CACHE_DIR}" in rendered
        assert f"export APPTAINER_TMPDIR={TMP_DIR}" in rendered

    def test_it_probes_outbound_access_before_building(self) -> None:
        rendered = _render()
        probe_at = rendered.index("pypi")
        build_at = rendered.index("bash build.sh")

        assert probe_at < build_at


class TestTheSelfCheckIsReRun:
    """Root ran it during %post; that says nothing about the real user."""

    def test_it_re_runs_the_selfcheck_as_the_unprivileged_user(self) -> None:
        rendered = _render()

        assert f"/opt/env/bin/python {SPEC_DIR}/{SELFCHECK_NAME}" in rendered
        assert "$(id -un)" in rendered

    def test_it_re_runs_only_when_the_build_succeeded(self) -> None:
        rendered = _render()
        build_at = rendered.index("rc=$?")
        guard_at = rendered.index("if [ $rc -eq 0 ]; then")

        assert build_at < guard_at


class TestSmokeCommands:
    """Importing a symbol is not computing with it."""

    def test_a_declared_command_is_run_inside_the_image(self) -> None:
        rendered = _render([_SMOKE])

        assert f"apptainer exec abl.sif {_SMOKE}" in rendered

    def test_it_runs_every_declared_command(self) -> None:
        rendered = _render([_SMOKE, "/opt/env/bin/python -c pass"])

        assert "smoke 1/2" in rendered
        assert "smoke 2/2" in rendered

    def test_a_failing_command_sets_a_distinct_exit_code(self) -> None:
        assert "rc=5" in _render([_SMOKE])

    def test_smoke_commands_get_no_binds(self) -> None:
        # At build time the image is the only thing that exists: there is no
        # job and nothing declaring what to mount. The first hand-written
        # version wrote its output to a host path and a CORRECT image failed
        # its own build with a read-only filesystem error.
        rendered = _render([_SMOKE])
        smoke_line = next(
            line for line in rendered.splitlines() if _SMOKE in line and "exec" in line
        )

        assert "--bind" not in smoke_line

    def test_the_failure_message_does_not_diagnose_a_cause(self) -> None:
        # The hand-written version printed "the guard is not in this image",
        # which the traceback directly contradicted -- the guard ran and the
        # OUTPUT PATH was unwritable. A check that names the wrong cause is
        # worse than one that says only that it failed.
        rendered = _render([_SMOKE])

        assert "SMOKE COMMAND 1 FAILED" in rendered
        assert "did not succeed inside it" in rendered
        assert "is not in this image" not in rendered

    def test_a_command_carrying_quotes_reaches_the_shell_unaltered(self) -> None:
        # The `if` line is the one place the command is a COMMAND, so its own
        # quoting has to survive: shell-quoting it here would hand python the
        # literal quotes and the assertion would never run.
        rendered = _render([_QUOTED_SMOKE])

        assert f"    if apptainer exec abl.sif {_QUOTED_SMOKE}; then" in rendered

    def test_the_announcement_actually_prints_the_command_it_ran(self) -> None:
        # Asserted by RUNNING the echo, not by reading it. Rendered raw the
        # line is valid shell and prints `python -c assert PROBE_LABEL ==
        # gpt2-tiny, PROBE_LABEL` -- quotes stripped -- so every substring
        # assertion on the rendered text passed while the log lied about
        # which command had failed.
        rendered = _render([_QUOTED_SMOKE])
        announce = next(line for line in rendered.splitlines() if line.startswith("    echo '"))

        printed = subprocess.run(
            [posix_bash(), "-c", announce.strip()],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")

        assert printed == f"--- smoke 1/1: {_QUOTED_SMOKE} ---\n"

    def test_the_failure_message_actually_prints_the_command_that_failed(self) -> None:
        rendered = _render([_QUOTED_SMOKE])
        failure = next(
            line for line in rendered.splitlines() if line.strip().startswith("echo 'SMOKE")
        )

        printed = subprocess.run(
            [posix_bash(), "-c", failure.strip().removesuffix(" >&2")],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")

        assert printed == f"SMOKE COMMAND 1 FAILED: {_QUOTED_SMOKE}\n"

    def test_declaring_none_says_so_rather_than_rendering_nothing(self) -> None:
        # Silence would read as a renderer that forgot, not as a spec that
        # asserts no behaviour.
        rendered = _render([])

        assert "declares no smoke commands" in rendered
        assert "apptainer exec abl.sif /opt/env/bin/python -m" not in rendered


class TestTheInterpreterIsChosenNotInherited:
    """The six failures in this file that were being called flaky.

    They resolved ``bash`` through an ambient PATH, so which program answered
    depended on the shell that launched pytest: MSYS bash from Git Bash, and
    ``C:\\Windows\\System32\\bash.exe`` -- the WSL launcher, not a shell --
    from the PowerShell the Makefile uses. When the WSL service was down the
    launcher exited 1 and six tests failed in a ``make check`` about something
    else entirely.
    """

    def test_the_launcher_is_recognised(self) -> None:
        assert is_wsl_launcher(r"C:\Windows\System32\bash.exe") is True

    def test_a_difference_in_case_does_not_hide_it(self) -> None:
        """Windows spells this directory both ways and means the same one."""
        assert is_wsl_launcher(r"C:\WINDOWS\system32\bash.exe") is True

    def test_the_32_bit_spelling_is_recognised_too(self) -> None:
        """`make` recipes run in 32-bit PowerShell, which sees SysWOW64."""
        assert is_wsl_launcher(r"C:\Windows\SysWOW64\bash.exe") is True

    def test_a_real_shell_is_not_mistaken_for_it(self) -> None:
        assert is_wsl_launcher(r"C:\Program Files\Git\usr\bin\bash.exe") is False
        assert is_wsl_launcher("/usr/bin/bash") is False

    def test_it_finds_the_bash_git_ships_two_directories_up(self, tmp_path: pathlib.Path) -> None:
        """The step that matters from PowerShell, where excluding the launcher
        leaves no bash on PATH at all."""
        (tmp_path / "usr" / "bin").mkdir(parents=True)
        (tmp_path / "usr" / "bin" / "bash.exe").write_bytes(b"")
        found = bash_beside_git(str(tmp_path / "cmd" / "git.exe"))
        assert found == str(tmp_path / "usr" / "bin" / "bash.exe")

    def test_it_reports_none_when_no_bash_ships_beside_git(self, tmp_path: pathlib.Path) -> None:
        """Which is the normal case off Windows, where the PATH search already
        succeeded and this is never reached."""
        assert bash_beside_git(str(tmp_path / "cmd" / "git.exe")) is None

    def test_the_resolved_interpreter_is_not_the_launcher(self) -> None:
        assert "system32" not in posix_bash().lower()

    def test_the_resolved_interpreter_runs_a_script(self) -> None:
        """The property every test below depends on, asserted once directly."""
        result = subprocess.run([posix_bash(), "-c", "echo ok"], capture_output=True, check=True)
        assert result.stdout.decode("utf-8").strip() == "ok"


class TestItIsRunnableShell:
    """Asserting on rendered text cannot see a syntax error in it.

    That distinction is not academic here: this renderer replaced four
    hand-copied scripts, and the defects that motivated it were both
    invisible to every substring assertion anyone had written. So the
    rendered script is handed to a real parser.
    """

    def _parse(self, rendered: str) -> subprocess.CompletedProcess[bytes]:
        """Run the rendered script through bash's syntax checker.

        Three details that each produced a wrong answer before being fixed:

        The interpreter is RESOLVED, not looked up on PATH -- see
        :func:`posix_bash`. This paragraph used to assert that "on Windows the
        bash on PATH is MSYS", which is true of a Git Bash session and false
        of the PowerShell one the Makefile runs pytest in, where the first
        match is the WSL launcher.

        The script is fed on stdin rather than written to a file and named on
        the command line. MSYS bash rewrites a native path argument into
        something it cannot open, so a path-based check returns 127 for every
        input -- failing everything while looking like a working test.

        The input is BYTES. With ``text=True`` Python translates ``\\n`` to
        ``\\r\\n`` on the way in, bash reads ``then\\r`` instead of ``then``,
        and every script is reported as a syntax error at its last line. The
        script really is LF-only; the harness was corrupting it.
        """
        return subprocess.run(
            [posix_bash(), "-n"],
            input=rendered.encode("utf-8"),
            capture_output=True,
            check=False,
        )

    def test_a_job_with_no_smoke_commands_parses(self) -> None:
        result = self._parse(_render())

        assert result.returncode == 0, result.stderr.decode("utf-8")

    def test_a_job_with_smoke_commands_parses(self) -> None:
        result = self._parse(_render([_SMOKE, "/opt/env/bin/python -c pass"]))

        assert result.returncode == 0, result.stderr.decode("utf-8")

    def test_a_job_whose_smoke_command_carries_quotes_parses(self) -> None:
        result = self._parse(_render([_QUOTED_SMOKE]))

        assert result.returncode == 0, result.stderr.decode("utf-8")

    def test_a_job_whose_smoke_command_carries_a_metacharacter_parses(self) -> None:
        # Rendered raw this one is a hard syntax error, not merely a wrong
        # message: the command's own quotes close against the echo's and
        # `((1,2))` lands bare, so bash rejects the whole script. A build job
        # that will not parse fails at its first line having built nothing.
        result = self._parse(_render([_METACHAR_SMOKE]))

        assert result.returncode == 0, result.stderr.decode("utf-8")

    def test_the_checker_actually_rejects_broken_shell(self) -> None:
        # Without this, a checker that silently passed everything -- which is
        # exactly what the path-based version did -- would look identical to
        # a working one.
        result = self._parse("if [ 1 -eq 1 ]; then\n  echo unterminated\n")

        assert result.returncode != 0


class TestTheJobIdentity:
    """Where the job runs and where its output lands."""

    def test_the_job_name_and_directories_come_from_the_caller(self) -> None:
        rendered = _render()

        assert "#SBATCH -J img.abl-sif-v9" in rendered
        assert "#SBATCH -o /pub/wagnera3/images/v9/build-%j.out" in rendered
        assert "#SBATCH -e /pub/wagnera3/images/v9/build-%j.err" in rendered
        assert "cd /pub/wagnera3/images/v9" in rendered

    def test_it_propagates_the_exit_code(self) -> None:
        assert _render().rstrip().endswith("exit $rc")

    def test_it_ends_with_a_newline(self) -> None:
        assert _render().endswith("\n")
