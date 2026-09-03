"""Tests for the image-rendering CLI.

The command's value is that four files come from one document, so the tests
that matter read the files back and check they agree with each other -- a
definition installing one torch beside a self-check expecting another is the
failure this command exists to prevent.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from hpc3.cli import image as image_cli
from hpc3.core.image_layout import (
    DEFINITION_NAME,
    REQUIREMENTS_NAME,
    SBATCH_NAME,
    SELFCHECK_NAME,
)

_COMMIT = "d11efacd231ef92426eaf92483c33a8504bd770f"


def _payload(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid image-spec document.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        The document.
    """
    base: dict[str, JSONValue] = {
        "base_image": "python:3.11.16-slim-bookworm",
        "env_prefix": "/opt/env",
        "git_commit": _COMMIT,
        "system_packages": [],
        "extra_index_urls": ["https://download.pytorch.org/whl/cu124"],
        "requirements": ["torch==2.6.0+cu124", "transformers==4.46.3"],
        "wheels": ["model_trainer_server-0.1.0-py3-none-any.whl"],
        "expected_versions": {"torch": "2.6.0+cu124"},
        "required_symbols": [
            {"module": "model_trainer.cluster.preflight", "attribute": "check_corpus_certified"}
        ],
        "smoke_commands": [],
        "labels": {"org.corvis.captured": "2026-08-25"},
        "project": "abl",
    }
    base.update(overrides)
    return base


def _write_spec(tmp_path: pathlib.Path, payload: JSONValue) -> pathlib.Path:
    """Write a spec document for the CLI to read.

    Args:
        tmp_path: Directory to write into.
        payload: Document to serialise.

    Returns:
        Path to the written file.
    """
    path = tmp_path / "spec.json"
    path.write_text(dump_json_str(payload), encoding="utf-8")
    return path


#: The project the default workspace declares. The job name is DERIVED
#: from it -- this file used to carry `JOB_NAME = "img.abl-sif-test"`,
#: whose project half names nothing, and rendering that was what pushed a
#: real build onto the unrecorded `sbatch` path.
BUILD_NAME = "image-test"
QUALIFIED_JOB_NAME = "abl.image-test"
IMAGE_DIR = "/pub/wagnera3/images/test"


def _argv(tmp_path: pathlib.Path, spec_path: pathlib.Path, out_dir: pathlib.Path) -> list[str]:
    """Build a complete command line for the renderer.

    Args:
        tmp_path: Working directory. The renderer reads no workspace: the
            project it composes the job name from comes from the spec.
        spec_path: The spec document to render from.
        out_dir: Directory the rendered files are written to.

    Returns:
        Every required flag, so a new one is added in a single place rather
        than in each of the call sites that previously spelled the list out.
    """
    return [
        "--name",
        BUILD_NAME,
        "--spec",
        str(spec_path),
        "--out-dir",
        str(out_dir),
        "--image-name",
        "abl.sif",
        "--image-dir",
        IMAGE_DIR,
    ]


def _render(tmp_path: pathlib.Path, payload: JSONValue) -> pathlib.Path:
    """Run the CLI and return the output directory.

    Args:
        tmp_path: Working directory.
        payload: Spec document.

    Returns:
        The directory the CLI wrote into.
    """
    spec_path = _write_spec(tmp_path, payload)
    out_dir = tmp_path / "build"
    assert image_cli.main(_argv(tmp_path, spec_path, out_dir)) == 0
    return out_dir


class TestItWritesEveryFile:
    """Five artifacts, one document."""

    def test_every_file_is_written(self, tmp_path: pathlib.Path, emitted: list[str]) -> None:
        out_dir = _render(tmp_path, _payload())
        written = sorted(p.name for p in out_dir.iterdir())
        assert written == sorted(
            [
                DEFINITION_NAME,
                REQUIREMENTS_NAME,
                SBATCH_NAME,
                image_cli.BUILD_SCRIPT_NAME,
                SELFCHECK_NAME,
            ]
        )
        assert len(emitted) == 7

    def test_it_creates_a_missing_output_directory(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        out_dir = _render(tmp_path, _payload())
        assert out_dir.is_dir()
        assert emitted[-2] == f"commit {_COMMIT}"

    def test_rendering_twice_replaces_rather_than_appends(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        out_dir = _render(tmp_path, _payload())
        first = (out_dir / DEFINITION_NAME).read_text(encoding="utf-8")
        _ = _render(tmp_path, _payload())
        assert (out_dir / DEFINITION_NAME).read_text(encoding="utf-8") == first

    def test_files_are_written_with_lf_endings(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        """CRLF would make the cluster report the interpreter as missing."""
        out_dir = _render(tmp_path, _payload())
        for name in (DEFINITION_NAME, SELFCHECK_NAME, image_cli.BUILD_SCRIPT_NAME):
            assert b"\r" not in (out_dir / name).read_bytes()


class TestTheFilesAgree:
    """The reason all four are rendered together."""

    def test_the_definition_and_selfcheck_name_the_same_versions(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        out_dir = _render(tmp_path, _payload())
        requirements = (out_dir / REQUIREMENTS_NAME).read_text(encoding="utf-8")
        selfcheck = (out_dir / SELFCHECK_NAME).read_text(encoding="utf-8")
        assert "torch==2.6.0+cu124" in requirements
        assert "'2.6.0+cu124'" in selfcheck

    def test_the_build_script_carries_the_spec_commit(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        out_dir = _render(tmp_path, _payload())
        script = (out_dir / image_cli.BUILD_SCRIPT_NAME).read_text(encoding="utf-8")
        assert f"'{_COMMIT}'" in script

    def test_the_definition_references_the_rendered_selfcheck_by_name(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        out_dir = _render(tmp_path, _payload())
        definition = (out_dir / DEFINITION_NAME).read_text(encoding="utf-8")
        assert SELFCHECK_NAME in definition
        assert (out_dir / SELFCHECK_NAME).is_file()


class TestItRefusesRatherThanRenders:
    """A spec that cannot be trusted must not produce a buildable definition."""

    def test_an_unpinned_requirement_writes_nothing(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_spec(tmp_path, _payload(requirements=["torch"]))
        out_dir = tmp_path / "build"
        with pytest.raises(JSONTypeError, match="must pin an exact version"):
            _ = image_cli.main(_argv(tmp_path, spec_path, out_dir))
        assert not out_dir.exists()

    def test_a_bind_mounted_env_prefix_writes_nothing(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_spec(tmp_path, _payload(env_prefix="/pub/wagnera3/envs/abl"))
        out_dir = tmp_path / "build"
        with pytest.raises(JSONTypeError, match="bind-mounts over"):
            _ = image_cli.main(_argv(tmp_path, spec_path, out_dir))
        assert not out_dir.exists()

    def test_the_job_name_is_the_qualified_one_and_reaches_the_script(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Composed from the SPEC's project, not taken from the caller.

        This is what makes the renderer and `hpc3-image-build` agree: the
        submitter reads `#SBATCH -J` back out and requires it to equal
        `<project>.<name>`, so deriving it from the spec means the two cannot
        disagree rather than disagreeing and being refused.

        Args:
            tmp_path: Working directory.
        """
        out_dir = _render(tmp_path, _payload())

        assert f"#SBATCH -J {QUALIFIED_JOB_NAME}" in (out_dir / SBATCH_NAME).read_text(
            encoding="utf-8"
        )

    def test_the_job_name_follows_the_spec_when_the_spec_changes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Proves the DERIVATION, not just that one string happens to match.

        The previous test would pass if the project were hard-coded anywhere
        in the renderer. Changing only the spec's project must change the
        rendered job name, which is what makes `check_name_agrees` meaningful
        downstream.

        Args:
            tmp_path: Working directory.
        """
        payload = _payload()
        payload["project"] = "turkic-lstm"

        out_dir = _render(tmp_path, payload)

        rendered = (out_dir / SBATCH_NAME).read_text(encoding="utf-8")
        assert f"#SBATCH -J turkic-lstm.{BUILD_NAME}" in rendered
        assert QUALIFIED_JOB_NAME not in rendered

    def test_the_project_cannot_be_supplied_alongside_the_spec(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The defect this closes, and it is closed harder than before.

        `img.abl-sif-v22` was rendered because the job name was free text;
        `img` names no project, so the raw sbatch that name invited recorded
        nothing. That used to be caught by looking the project up in the
        workspace, which a project being ONBOARDED is not in.

        There is now nothing to look up, because there is nothing to supply:
        the project comes from the spec capture wrote, and the flag is gone.
        """
        spec_path = _write_spec(tmp_path, _payload())
        out_dir = tmp_path / "build"

        with pytest.raises(ValueError, match="unknown argument"):
            _ = image_cli.main([*_argv(tmp_path, spec_path, out_dir), "--project", "img"])

        assert not out_dir.exists()

    def test_a_spec_naming_an_unusable_project_writes_nothing(self, tmp_path: pathlib.Path) -> None:
        """The name still has to be one squeue would accept.

        Validated by the same function the job contract uses, at decode, so a
        spec cannot carry a project name the layout rejects.
        """
        payload = _payload()
        payload["project"] = "My Project"
        spec_path = _write_spec(tmp_path, payload)
        out_dir = tmp_path / "build"

        with pytest.raises(JSONTypeError):
            _ = image_cli.main(_argv(tmp_path, spec_path, out_dir))

        assert not out_dir.exists()

    def test_a_name_containing_a_dot_writes_nothing(self, tmp_path: pathlib.Path) -> None:
        # The dot is the separator `project_of` splits on, so a name carrying
        # one makes the renderer and the reader disagree about where the
        # project ends.
        spec_path = _write_spec(tmp_path, _payload())
        out_dir = tmp_path / "build"
        argv = _argv(tmp_path, spec_path, out_dir)
        argv[argv.index("--name") + 1] = "img.abl-sif-v22"

        with pytest.raises(ValueError, match="must not contain a dot"):
            _ = image_cli.main(argv)

        assert not out_dir.exists()

    def test_an_empty_name_writes_nothing(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_spec(tmp_path, _payload())
        out_dir = tmp_path / "build"
        argv = _argv(tmp_path, spec_path, out_dir)
        argv[argv.index("--name") + 1] = ""

        with pytest.raises(ValueError, match="must not be empty"):
            _ = image_cli.main(argv)

        assert not out_dir.exists()

    def test_a_missing_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_spec(tmp_path, _payload())
        with pytest.raises(ValueError, match="--image-name"):
            _ = image_cli.main(["--spec", str(spec_path), "--out-dir", str(tmp_path / "b")])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_spec(tmp_path, _payload())
        with pytest.raises(ValueError):
            _ = image_cli.main(
                [
                    "--spec",
                    str(spec_path),
                    "--out-dir",
                    str(tmp_path / "b"),
                    "--image-name",
                    "abl.sif",
                    "--nope",
                    "x",
                ]
            )


class TestEntrypoint:
    """``entrypoint`` reads ``sys.argv`` and raises; that only happens when a
    process starts through it, so it is exercised for real rather than
    excluded from coverage.
    """

    def test_it_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, emitted: list[str], argv: list[str]
    ) -> None:
        spec_path = _write_spec(tmp_path, _payload())
        out_dir = tmp_path / "build"
        argv[:] = ["prog", *_argv(tmp_path, spec_path, out_dir)]

        with pytest.raises(SystemExit) as excinfo:
            image_cli.entrypoint()
        assert excinfo.value.code == 0
        assert (out_dir / DEFINITION_NAME).is_file()

    def test_a_refused_spec_exits_two_without_writing(
        self, tmp_path: pathlib.Path, emitted: list[str], errors: list[str], argv: list[str]
    ) -> None:
        """A contract refusal is a refusal, not a crash: status 2, no files."""
        spec_path = _write_spec(tmp_path, _payload(requirements=["torch"]))
        out_dir = tmp_path / "build"
        argv[:] = ["prog", *_argv(tmp_path, spec_path, out_dir)]

        with pytest.raises(SystemExit) as excinfo:
            image_cli.entrypoint()
        assert excinfo.value.code == 2
        assert not out_dir.exists()
        assert any("must pin an exact version" in line for line in errors)
