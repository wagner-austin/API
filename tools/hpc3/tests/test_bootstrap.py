"""The first environment gets created, and is proven to be the one asked for.

These tests drive the real command against the scripted runner, so what is
exercised is the command line this package would actually send to the cluster
-- including the ``module load ... && conda create`` join, which is the detail
a separate-calls implementation gets wrong invisibly.

The refusals matter more here than the happy path. Every other refusal in this
package fires at a project that already runs; these three fire at what the
command itself just built, and the one worth having is the borrowed-interpreter
check, because a borrowed environment works perfectly until the environment it
borrowed from is deleted.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.cli import bootstrap as bootstrap_cli
from hpc3.core import bootstrap
from tests.conftest import FakeRun, LoggedEvent, workspace_document, write_workspace

_ENV = "/pub/wagnera3/envs/newcomer"
_PYTHON = "3.11"

# What the identity probe prints: version, then base_prefix.
_SELF_CONTAINED = f"{_PYTHON}\n{_ENV}\n"
_BORROWED = f"{_PYTHON}\n/pub/wagnera3/envs/cleargbm\n"


def _args(tmp_path: pathlib.Path, **overrides: str) -> list[str]:
    """Build a valid argument list.

    Args:
        tmp_path: Directory holding the workspace document.
        **overrides: Flag values to replace.

    Returns:
        The argument list.
    """
    values = {
        "--config": str(tmp_path / "hpc3.json"),
        "--project": "newcomer",
        "--env-path": _ENV,
        "--python": _PYTHON,
    }
    values.update(overrides)
    return [token for flag, value in values.items() for token in (flag, value)]


def _workspace(tmp_path: pathlib.Path) -> None:
    """Write a workspace declaring no projects at all.

    A project being bootstrapped cannot be registered yet, so the document
    this command reads in real use has nothing in its project table for the
    name being passed. Writing it that way is the test that the command reads
    the CONNECTION rather than the registry.

    Args:
        tmp_path: Directory to write into.
    """
    _ = write_workspace(tmp_path / "hpc3.json", workspace_document(projects={}))


def _script_success(fake: FakeRun) -> None:
    """Script a cluster where the path is free and creation works.

    Args:
        fake: The runner to script.
    """
    fake.add("test -e", stdout="absent\n")
    fake.add("conda create", stdout="done\n")
    fake.add("-c ", stdout=_SELF_CONTAINED)


class TestTheCommandItSends:
    """The command line is the product; a wrong one fails only on a cluster."""

    def test_module_load_and_conda_create_are_one_command(self) -> None:
        """Separate SSH calls would lose the PATH the module sets.

        Each run_remote is its own session with its own shell, so a load in
        one call is gone before the next. Joined by && they share a shell.
        """
        command = bootstrap.create_command(_ENV, _PYTHON)

        assert command == (
            f"module load {bootstrap.CONDA_MODULE} && "
            f"conda create -y -p '{_ENV}' 'python={_PYTHON}'"
        )
        assert " && " in command

    def test_the_conda_module_is_version_pinned(self) -> None:
        """A bare name resolves to whatever the cluster currently defaults to."""
        assert bootstrap.CONDA_MODULE == "miniconda3/24.9.2"

    def test_creation_never_prompts(self) -> None:
        """A prompt over BatchMode ssh hangs rather than failing."""
        assert "-y" in bootstrap.create_command(_ENV, _PYTHON)

    def test_the_probe_runs_the_environments_own_interpreter_by_path(self) -> None:
        """Through PATH it would answer for whatever a login shell activates."""
        assert bootstrap.identity_command(_ENV).startswith(f"'{_ENV}/bin/python' -c ")

    def test_the_probe_carries_no_newline(self) -> None:
        """A real newline would be split by the remote shell before Python saw it."""
        assert "\n" not in bootstrap.identity_command(_ENV)


class TestParseIdentity:
    """An unreadable answer must not be read as a wrong version."""

    def test_two_lines_become_version_and_base_prefix(self) -> None:
        identity = bootstrap.parse_identity(_SELF_CONTAINED)

        assert identity == {"version": "3.11", "base_prefix": _ENV}

    def test_surrounding_blank_lines_are_ignored(self) -> None:
        identity = bootstrap.parse_identity(f"\n  {_PYTHON}  \n\n  {_ENV}  \n\n")

        assert identity == {"version": "3.11", "base_prefix": _ENV}

    @pytest.mark.parametrize("output", ["", "3.11\n", "3.11\n/a\n/b\n", "Traceback...\n"])
    def test_anything_that_is_not_two_lines_is_refused(self, output: str) -> None:
        """Read as a version of '' this would blame conda for a failed probe."""
        with pytest.raises(AppError) as excinfo:
            _ = bootstrap.parse_identity(output)

        assert excinfo.value.code is Hpc3ErrorCode.ENV_PROBE_UNREADABLE


class TestCheckIdentity:
    """What was built is held to what was asked for."""

    def test_a_matching_self_contained_environment_passes(self) -> None:
        bootstrap.check_identity(
            {"version": _PYTHON, "base_prefix": _ENV}, env_path=_ENV, python_version=_PYTHON
        )

    def test_a_different_version_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            bootstrap.check_identity(
                {"version": "3.10", "base_prefix": _ENV}, env_path=_ENV, python_version=_PYTHON
            )

        assert excinfo.value.code is Hpc3ErrorCode.BOOTSTRAP_PYTHON_MISMATCH
        assert "3.10" in str(excinfo.value)
        assert "3.11" in str(excinfo.value)

    def test_a_borrowed_interpreter_is_refused(self) -> None:
        """The measured defect: envs/tankpit's python lives in envs/cleargbm.

        Correct version, working environment, and a dependency on another
        project that no run document records. Nothing else in this package
        reads base_prefix, so this is the only check that can see it.
        """
        with pytest.raises(AppError) as excinfo:
            bootstrap.check_identity(
                {"version": _PYTHON, "base_prefix": "/pub/wagnera3/envs/cleargbm"},
                env_path=_ENV,
                python_version=_PYTHON,
            )

        assert excinfo.value.code is Hpc3ErrorCode.BOOTSTRAP_ENV_NOT_SELF_CONTAINED
        assert "/pub/wagnera3/envs/cleargbm" in str(excinfo.value)


class TestRefuseExisting:
    """An existing directory may be the source an image spec already names."""

    def test_an_absent_path_is_allowed(self, fake_run: FakeRun) -> None:
        fake_run.add("test -e", stdout="absent\n")

        bootstrap.refuse_existing("hpc3", _ENV)

    def test_an_occupied_path_is_refused(self, fake_run: FakeRun) -> None:
        fake_run.add("test -e", stdout="present\n")

        with pytest.raises(AppError) as excinfo:
            bootstrap.refuse_existing("hpc3", _ENV)

        assert excinfo.value.code is Hpc3ErrorCode.BOOTSTRAP_ENV_EXISTS
        assert _ENV in str(excinfo.value)


class TestBootstrapEnvironment:
    """The whole sequence, in the order that makes a failure cheap."""

    def test_it_returns_the_verified_identity(self, fake_run: FakeRun) -> None:
        _script_success(fake_run)

        identity = bootstrap.bootstrap_environment("hpc3", _ENV, _PYTHON)

        assert identity == {"version": _PYTHON, "base_prefix": _ENV}

    def test_an_occupied_path_stops_before_conda_runs(self, fake_run: FakeRun) -> None:
        """A mistyped path must not cost an environment build first."""
        fake_run.add("test -e", stdout="present\n")

        with pytest.raises(AppError) as excinfo:
            _ = bootstrap.bootstrap_environment("hpc3", _ENV, _PYTHON)

        assert excinfo.value.code is Hpc3ErrorCode.BOOTSTRAP_ENV_EXISTS
        assert not any("conda create" in call.remote_command for call in fake_run.calls)

    def test_a_borrowed_environment_is_refused_after_creation(self, fake_run: FakeRun) -> None:
        """Created and wrong is the state this exists to refuse leaving behind."""
        fake_run.add("test -e", stdout="absent\n")
        fake_run.add("conda create", stdout="done\n")
        fake_run.add("-c ", stdout=_BORROWED)

        with pytest.raises(AppError) as excinfo:
            _ = bootstrap.bootstrap_environment("hpc3", _ENV, _PYTHON)

        assert excinfo.value.code is Hpc3ErrorCode.BOOTSTRAP_ENV_NOT_SELF_CONTAINED

    def test_a_conda_failure_surfaces_with_the_clusters_own_stderr(self, fake_run: FakeRun) -> None:
        fake_run.add("test -e", stdout="absent\n")
        fake_run.add("conda create", returncode=1, stderr="PackagesNotFoundError: python=3.99")

        with pytest.raises(AppError) as excinfo:
            _ = bootstrap.bootstrap_environment("hpc3", _ENV, "3.99")

        assert excinfo.value.code is Hpc3ErrorCode.REMOTE_COMMAND_FAILED
        assert "PackagesNotFoundError" in str(excinfo.value)


class TestTheCommand:
    """End to end through main, against a workspace with no projects in it."""

    def test_it_succeeds_and_reports_where_to_go_next(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        logged: list[LoggedEvent],
    ) -> None:
        _workspace(tmp_path)
        _script_success(fake_run)

        assert bootstrap_cli.main(_args(tmp_path)) == 0

        assert any("hpc3-image-capture --env-path" in line for line in emitted)

    def test_it_reads_a_workspace_that_declares_no_projects(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The bootstrap paradox: registration needs an image, which starts here.

        Loading the full workspace would refuse this command over a registry
        entry that cannot exist yet.
        """
        _workspace(tmp_path)
        _script_success(fake_run)

        assert bootstrap_cli.main(_args(tmp_path, **{"--project": "not-registered"})) == 0

    def test_the_audit_event_records_what_makes_it_repeatable(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        logged: list[LoggedEvent],
    ) -> None:
        """A directory on shared storage with no record of what made it is the
        problem this event exists to stop repeating."""
        _workspace(tmp_path)
        _script_success(fake_run)

        _ = bootstrap_cli.main(_args(tmp_path))

        events = [event for event in logged if event.event == "hpc3_environment_bootstrapped"]
        assert len(events) == 1
        assert events[0].fields == {
            "host": "hpc3",
            "project": "newcomer",
            "env_path": _ENV,
            "python_version": _PYTHON,
            "conda_module": bootstrap.CONDA_MODULE,
        }

    def test_no_event_is_recorded_when_the_environment_is_wrong(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        logged: list[LoggedEvent],
    ) -> None:
        """An event for an environment that turned out wrong is a false trail."""
        _workspace(tmp_path)
        fake_run.add("test -e", stdout="absent\n")
        fake_run.add("conda create", stdout="done\n")
        fake_run.add("-c ", stdout=_BORROWED)

        with pytest.raises(AppError):
            _ = bootstrap_cli.main(_args(tmp_path))

        assert [e for e in logged if e.event == "hpc3_environment_bootstrapped"] == []

    def test_a_missing_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError):
            _ = bootstrap_cli.main([str(tmp_path / "hpc3.json"), "--project", "newcomer"])


class TestEntrypoint:
    """``entrypoint`` reads sys.argv, so it is exercised for real."""

    def test_it_reads_the_process_arguments(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        logged: list[LoggedEvent],
        argv: list[str],
    ) -> None:
        _workspace(tmp_path)
        _script_success(fake_run)
        argv[:] = ["hpc3-bootstrap", *_args(tmp_path)]

        with pytest.raises(SystemExit) as excinfo:
            bootstrap_cli.entrypoint()

        assert excinfo.value.code == 0
