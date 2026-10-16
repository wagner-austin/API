"""Tests for comparing the code an image runs with the code its submitter has.

Board task 7aab32ef. A module the image HAS is admitted, so reproducing an old
command under its old image stays legal, but when its source has moved on
since the image's commit the run executes older code than HEAD, and that is
measured, printed and recorded. The comparison runs REAL git against a
repository built here with two commits, because what is being checked is
which paths git says changed, and a scripted git would only agree with this
file. The refusals git cannot be made to produce on demand are scripted.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.drift import ImageDrift, ModuleDrift, decode_image_drift, encode_image_drift
from hpc3.core.drift import (
    describe_drift,
    drift_for_command,
    measure_drift,
    merge_drifts,
    source_paths,
)
from hpc3.core.entrypoint import EntrypointReading, ModuleReading
from tests.conftest import FakeRun, write_file

_WALL_SECONDS = 60
"""Bound on each git call; every one of them answers in milliseconds."""

_SITE = "/opt/env/lib/python3.11/site-packages"
_JOBS = "/pub/wagnera3/mi/scripts"


def _git(cwd: pathlib.Path, *arguments: str) -> str:
    """Run git in a directory.

    Args:
        cwd: Where to run it.
        *arguments: Git's arguments.

    Returns:
        Its standard output, stripped.
    """
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "init.defaultBranch=main",
            *arguments,
        ],
        cwd=cwd,
        capture_output=True,
        check=True,
        text=True,
        timeout=_WALL_SECONDS,
    ).stdout.strip()


def _commit(repo: pathlib.Path, files: dict[str, bytes], message: str) -> str:
    """Write files and commit them.

    Args:
        repo: The repository.
        files: Paths relative to it, and their bytes.
        message: The commit message.

    Returns:
        The new commit.
    """
    for path, payload in files.items():
        write_file(repo / path, payload)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class _History:
    """A repository whose image was built at one commit and whose HEAD moved on."""

    def __init__(self, root: pathlib.Path) -> None:
        """Build it: the image commit, then one that edits the trainer.

        Args:
            root: An absent directory to create.
        """
        root.mkdir(parents=True)
        _git(root, "init", "-q")
        self.repo = root.resolve()
        self.image_commit = _commit(
            self.repo,
            {
                "services/trainer/src/model_trainer/__init__.py": b"",
                "services/trainer/src/model_trainer/cli/__init__.py": b"",
                "services/trainer/src/model_trainer/cli/probe.py": b"STEPS = 1\n",
                "libs/steady/__init__.py": b"",
                "libs/core/__init__.py": b"",
                "libs/core/model_trainer.py": b"CONFIG = 1\n",
                "tools/single.py": b"X = 1\n",
                "README.md": b"one\n",
            },
            "the image's commit",
        )
        self.head = _commit(
            self.repo,
            {
                "services/trainer/src/model_trainer/cli/probe.py": b"STEPS = 2\n",
                "libs/core/model_trainer.py": b"CONFIG = 2\n",
                "README.md": b"two\n",
            },
            "after the image",
        )


def _in_image(module: str) -> ModuleReading:
    """Build the reading of a module the image's environment holds.

    Args:
        module: The module.

    Returns:
        The reading.
    """
    return ModuleReading(
        directory=_JOBS,
        module=module,
        location=f"{_SITE}/{module.split('.')[0]}",
        in_environment=True,
    )


def _compared(module: str, compared: list[str], changed: list[str]) -> ModuleDrift:
    """Build the entry of a module that was compared.

    Args:
        module: The module.
        compared: What was compared.
        changed: What changed.

    Returns:
        The entry.
    """
    return ModuleDrift(
        directory=_JOBS, module=module, compared=compared, changed=changed, not_compared=""
    )


class TestMeasuringAgainstRealHistory:
    def test_a_module_whose_source_moved_names_what_changed(self, tmp_path: pathlib.Path) -> None:
        """Only its own package: the README edit in the same commit is not its
        code, and neither is libs/core/model_trainer.py, a submodule of
        another package that shares the name -- the first real run counted
        platform_core's config/model_trainer.py as the trainer's."""
        history = _History(tmp_path / "repo")
        reading = EntrypointReading(
            image_commit=history.image_commit, modules=[_in_image("model_trainer.cli.probe")]
        )
        assert measure_drift(history.repo, reading) == ImageDrift(
            image_commit=history.image_commit,
            head=history.head,
            modules=[
                _compared(
                    "model_trainer.cli.probe",
                    ["services/trainer/src/model_trainer"],
                    ["services/trainer/src/model_trainer/cli/probe.py"],
                )
            ],
        )

    def test_a_package_and_a_single_file_module_that_did_not_move(
        self, tmp_path: pathlib.Path
    ) -> None:
        history = _History(tmp_path / "repo")
        reading = EntrypointReading(
            image_commit=history.image_commit,
            modules=[_in_image("steady"), _in_image("single")],
        )
        assert measure_drift(history.repo, reading)["modules"] == [
            _compared("steady", ["libs/steady"], []),
            _compared("single", ["tools/single.py"], []),
        ]

    def test_what_is_not_compared_says_why(self, tmp_path: pathlib.Path) -> None:
        """A checkout's own copy, and a package with no source here."""
        history = _History(tmp_path / "repo")
        from_checkout = ModuleReading(
            directory="/pub/api",
            module="scripts.optimize",
            location="/pub/api/scripts",
            in_environment=False,
        )
        reading = EntrypointReading(
            image_commit=history.image_commit,
            modules=[from_checkout, _in_image("torch.distributed.run")],
        )
        first, second = measure_drift(history.repo, reading)["modules"]
        assert first["not_compared"] == (
            "resolved from /pub/api/scripts, outside the image's environment, so the image's "
            "commit does not describe its code"
        )
        assert second["not_compared"] == f"{history.repo} tracks no source for 'torch' at HEAD"
        assert first["compared"] == second["compared"] == []

    def test_a_commit_the_repository_does_not_hold_is_refused(self, tmp_path: pathlib.Path) -> None:
        history = _History(tmp_path / "repo")
        unknown = "0" * 40
        reading = EntrypointReading(
            image_commit=unknown, modules=[_in_image("model_trainer.cli.probe")]
        )
        with pytest.raises(AppError) as excinfo:
            measure_drift(history.repo, reading)
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_IMAGE_COMMIT_UNKNOWN
        assert f"built from {unknown}, which {history.repo} does not hold" in excinfo.value.message

    def test_a_repository_that_is_not_one_is_refused_alike(self, tmp_path: pathlib.Path) -> None:
        reading = EntrypointReading(image_commit="fab1d991", modules=[])
        with pytest.raises(AppError) as excinfo:
            measure_drift(tmp_path, reading)
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_IMAGE_COMMIT_UNKNOWN
        assert "not a git repository" in excinfo.value.message

    def test_a_run_with_no_image_compares_nothing_and_says_so(self, tmp_path: pathlib.Path) -> None:
        """No git is run: there is no build commit to compare from."""
        reading = EntrypointReading(image_commit=None, modules=[_in_image("model_trainer")])
        assert measure_drift(tmp_path / "absent", reading) == ImageDrift(
            image_commit="",
            head="",
            modules=[
                ModuleDrift(
                    directory=_JOBS,
                    module="model_trainer",
                    compared=[],
                    changed=[],
                    not_compared="the run has no image, so no build commit identifies its code",
                )
            ],
        )


class TestGitFailingOnceTheCommitIsKnown:
    """Refusals real git cannot be made to produce on demand, scripted."""

    def test_a_head_it_cannot_resolve_is_refused(self, fake_run: FakeRun) -> None:
        fake_run.add("HEAD^{commit}", returncode=128, stderr="fatal: ambiguous argument")
        with pytest.raises(AppError) as excinfo:
            measure_drift(pathlib.Path("/r"), EntrypointReading(image_commit="c", modules=[]))
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_DRIFT_UNREADABLE
        assert "exited 128: fatal: ambiguous argument" in excinfo.value.message

    def test_a_head_that_names_nothing_is_refused_before_it_is_recorded(
        self, fake_run: FakeRun
    ) -> None:
        """The ledger refuses an image commit with no HEAD, so it is never written."""
        fake_run.add("HEAD^{commit}", stdout="\n")
        with pytest.raises(AppError) as excinfo:
            measure_drift(pathlib.Path("/r"), EntrypointReading(image_commit="c", modules=[]))
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_DRIFT_UNREADABLE
        assert "named no HEAD commit" in excinfo.value.message


class TestWhereAPackageLives:
    def test_every_copy_at_a_source_root_is_found_once(self) -> None:
        tracked = [
            "src/pkg/__init__.py",
            "src/pkg/sub/__init__.py",
            "other/src/pkg/__init__.py",
            "tools/pkg.py",
            "pkg/__init__.py",
            "notpkg/__init__.py",
            "",
        ]
        assert source_paths(tracked, "pkg") == ["other/src/pkg", "pkg", "src/pkg", "tools/pkg.py"]

    def test_a_same_named_submodule_of_another_package_is_not_it(self) -> None:
        """``host.pkg`` is imported as ``host.pkg``, never as ``pkg``."""
        tracked = [
            "lib/host/__init__.py",
            "lib/host/pkg.py",
            "lib/host/nested/__init__.py",
            "lib/host/nested/pkg/__init__.py",
        ]
        assert source_paths(tracked, "pkg") == []


_DRIFT = ImageDrift(
    image_commit="fab1d991",
    head="51b39a57",
    modules=[
        _compared("a.run", ["src/a"], ["src/a/run.py"]),
        _compared("b", ["src/b"], []),
    ],
)


class TestOneSubmissionManyMembers:
    def test_a_member_row_names_only_what_it_runs(self) -> None:
        assert drift_for_command(
            _DRIFT, "python -m b --seed 1", working_directory=_JOBS
        ) == ImageDrift(image_commit="fab1d991", head="51b39a57", modules=[_DRIFT["modules"][1]])

    def test_members_merge_back_into_the_submission_once_each(self) -> None:
        first = drift_for_command(_DRIFT, "python -m a.run; python -m b", working_directory=_JOBS)
        second = drift_for_command(_DRIFT, "python -m b", working_directory=_JOBS)
        assert merge_drifts([first, second]) == _DRIFT


class TestWhatACommandPrints:
    def test_a_moved_module_is_a_warning_naming_both_commits(self) -> None:
        assert describe_drift(_DRIFT) == [
            "  code: image built from fab1d991, compared with HEAD 51b39a57",
            "  WARNING a.run: 1 file(s) under src/a changed since the image's commit, so this "
            "run executes the image's older code, not HEAD's: src/a/run.py",
            "  b: unchanged since the image's commit under src/b",
        ]

    def test_a_long_change_list_is_counted_past_five(self) -> None:
        changed = [f"src/a/{index}.py" for index in range(7)]
        lines = describe_drift(
            ImageDrift(image_commit="c", head="h", modules=[_compared("a", ["src/a"], changed)])
        )
        assert lines[1].endswith(
            "src/a/0.py, src/a/1.py, src/a/2.py, src/a/3.py, src/a/4.py and 2 more"
        )

    def test_nothing_compared_is_said_rather_than_left_blank(self) -> None:
        not_compared = ModuleDrift(
            directory=_JOBS, module="m", compared=[], changed=[], not_compared="the reason"
        )
        assert describe_drift(ImageDrift(image_commit="", head="", modules=[not_compared])) == [
            "  code: no image, so no build commit to compare",
            "  m: not compared: the reason",
        ]
        assert describe_drift(ImageDrift(image_commit="c", head="h", modules=[])) == [
            "  code: image built from c, compared with HEAD h",
            "  code: the command starts no python -m module, so no source was compared",
        ]


class TestTheRecordedShape:
    def test_a_record_round_trips(self) -> None:
        assert decode_image_drift(encode_image_drift(_DRIFT)) == _DRIFT

    @pytest.mark.parametrize(
        ("record", "why"),
        [
            ({"image_commit": "c", "head": "", "modules": []}, "both set"),
            ({"image_commit": "", "head": "h", "modules": []}, "both empty"),
            (
                {
                    "image_commit": "c",
                    "head": "h",
                    "modules": [
                        {
                            "directory": "/j",
                            "module": "m",
                            "compared": [],
                            "changed": [],
                            "not_compared": "",
                        }
                    ],
                },
                "either list what was compared or say why",
            ),
            (
                {
                    "image_commit": "c",
                    "head": "h",
                    "modules": [
                        {
                            "directory": "/j",
                            "module": "m",
                            "compared": ["x"],
                            "changed": [],
                            "not_compared": "y",
                        }
                    ],
                },
                "and not both",
            ),
        ],
    )
    def test_a_record_that_contradicts_itself_is_refused(
        self, record: dict[str, JSONValue], why: str
    ) -> None:
        with pytest.raises(JSONTypeError) as excinfo:
            decode_image_drift(record)
        assert why in str(excinfo.value)
