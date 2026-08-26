"""Tests for the rendered build job.

This renderer exists because the job was hand written once per image version
and two defects entered through the copying. Both have an assertion here, so
neither can return: a smoke command reaching for a host path, and a failure
message naming a cause the job never established.
"""

from __future__ import annotations

import subprocess

from hpc3.core.image_exec import APPTAINER_MODULE
from hpc3.core.image_layout import SELFCHECK_NAME, SPEC_DIR
from hpc3.core.image_sbatch import (
    BUILD_PARTITION,
    CACHE_DIR,
    TMP_DIR,
    render_build_sbatch,
)

_SMOKE = "/opt/env/bin/python -m pkg.probe --device cpu --out /tmp/probe.json"


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

    def test_declaring_none_says_so_rather_than_rendering_nothing(self) -> None:
        # Silence would read as a renderer that forgot, not as a spec that
        # asserts no behaviour.
        rendered = _render([])

        assert "declares no smoke commands" in rendered
        assert "apptainer exec abl.sif /opt/env/bin/python -m" not in rendered


class TestItIsRunnableShell:
    """Asserting on rendered text cannot see a syntax error in it.

    That distinction is not academic here: this renderer replaced four
    hand-copied scripts, and the defects that motivated it were both
    invisible to every substring assertion anyone had written. So the
    rendered script is handed to a real parser.
    """

    def _parse(self, rendered: str) -> subprocess.CompletedProcess[bytes]:
        """Run the rendered script through bash's syntax checker.

        Two details that both produced a wrong answer before being fixed:

        The script is fed on stdin rather than written to a file and named on
        the command line. On Windows the bash on PATH is MSYS, which rewrites
        a native path argument into something it cannot open, so a path-based
        check returns 127 for every input -- failing everything while looking
        like a working test.

        The input is BYTES. With ``text=True`` Python translates ``\\n`` to
        ``\\r\\n`` on the way in, bash reads ``then\\r`` instead of ``then``,
        and every script is reported as a syntax error at its last line. The
        script really is LF-only; the harness was corrupting it.
        """
        return subprocess.run(
            ["bash", "-n"],
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
