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
from hpc3.core.image_layout import DEFINITION_NAME, REQUIREMENTS_NAME, SELFCHECK_NAME

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
        "extra_index_urls": ["https://download.pytorch.org/whl/cu124"],
        "requirements": ["torch==2.6.0+cu124", "transformers==4.46.3"],
        "wheels": ["model_trainer_server-0.1.0-py3-none-any.whl"],
        "expected_versions": {"torch": "2.6.0+cu124"},
        "required_symbols": [
            {"module": "model_trainer.cluster.preflight", "attribute": "check_corpus_certified"}
        ],
        "labels": {"org.corvis.captured": "2026-08-25"},
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
    assert (
        image_cli.main(
            [
                "--spec",
                str(spec_path),
                "--out-dir",
                str(out_dir),
                "--image-name",
                "abl.sif",
            ]
        )
        == 0
    )
    return out_dir


class TestItWritesEveryFile:
    """Four artifacts, one document."""

    def test_all_four_files_are_written(self, tmp_path: pathlib.Path, emitted: list[str]) -> None:
        out_dir = _render(tmp_path, _payload())
        written = sorted(p.name for p in out_dir.iterdir())
        assert written == sorted(
            [
                DEFINITION_NAME,
                REQUIREMENTS_NAME,
                image_cli.BUILD_SCRIPT_NAME,
                SELFCHECK_NAME,
            ]
        )
        assert len(emitted) == 6

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
            _ = image_cli.main(
                ["--spec", str(spec_path), "--out-dir", str(out_dir), "--image-name", "abl.sif"]
            )
        assert not out_dir.exists()

    def test_a_bind_mounted_env_prefix_writes_nothing(self, tmp_path: pathlib.Path) -> None:
        spec_path = _write_spec(tmp_path, _payload(env_prefix="/pub/wagnera3/envs/abl"))
        out_dir = tmp_path / "build"
        with pytest.raises(JSONTypeError, match="bind-mounts over"):
            _ = image_cli.main(
                ["--spec", str(spec_path), "--out-dir", str(out_dir), "--image-name", "abl.sif"]
            )
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
        argv[:] = [
            "prog",
            "--spec",
            str(spec_path),
            "--out-dir",
            str(out_dir),
            "--image-name",
            "abl.sif",
        ]

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
        argv[:] = [
            "prog",
            "--spec",
            str(spec_path),
            "--out-dir",
            str(out_dir),
            "--image-name",
            "abl.sif",
        ]

        with pytest.raises(SystemExit) as excinfo:
            image_cli.entrypoint()
        assert excinfo.value.code == 2
        assert not out_dir.exists()
        assert any("must pin an exact version" in line for line in errors)
