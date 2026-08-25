"""Tests for capturing a project's environment into an image spec.

The command's value is that it reuses what the project already declares
rather than restating it: ``env_path`` says which environment to probe, and
``pinned_packages`` become the assertions the built image checks itself
against. A second list of versions is a second thing to keep in step, which
is the drift this whole area exists to remove.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str

from hpc3.cli import image_capture as capture_cli
from hpc3.contracts.image_spec import decode_image_spec
from tests.conftest import FakeRun, project_config, workspace_document, write_workspace

_FREEZE = "\n".join(
    [
        "torch==2.6.0+cu124",
        "transformers==4.46.3",
        "pip==26.2.1",
        "platform-core==0.1.0",
        "model-trainer-server==0.1.0",
    ]
)

_COMMIT = "d11efacd231ef92426eaf92483c33a8504bd770f"


def _args(tmp_path: pathlib.Path, **overrides: str) -> list[str]:
    """Build a complete argument list with optional overrides.

    Args:
        tmp_path: Working directory holding the workspace document.
        **overrides: Flag values to replace.

    Returns:
        The argument list.
    """
    values = {
        "--config": str(tmp_path / "hpc3.json"),
        "--project": "abl",
        "--commit": _COMMIT,
        "--base-image": "python:3.11.16-slim-bookworm",
        "--env-prefix": "/opt/env",
        "--first-party": "platform_core,model-trainer-server",
        "--symbols": "model_trainer.cluster.preflight:check_corpus_certified",
        "--extra-index-url": "https://download.pytorch.org/whl/cu124",
        "--out": str(tmp_path / "specs" / "abl-image.json"),
    }
    values.update(overrides)
    return [token for flag, value in values.items() for token in (flag, value)]


def _workspace(tmp_path: pathlib.Path) -> None:
    """Write a workspace declaring one project with pinned versions.

    Args:
        tmp_path: Directory to write into.
    """
    config: JSONValue = project_config(
        env_path="/pub/wagnera3/envs/abl-pinned",
        pinned_packages={"torch": "2.6.0+cu124", "transformers": "4.46.3"},
    )
    _ = write_workspace(tmp_path / "hpc3.json", workspace_document(projects={"abl": config}))


def _capture(tmp_path: pathlib.Path, fake_run: FakeRun, **overrides: str) -> pathlib.Path:
    """Run the command against a fake probe and return the written path.

    Args:
        tmp_path: Working directory.
        fake_run: The fake command runner.
        **overrides: Flag values to replace.

    Returns:
        Path to the spec the command wrote.
    """
    _workspace(tmp_path)
    fake_run.add("bin/python", stdout=_FREEZE)
    assert capture_cli.main(_args(tmp_path, **overrides)) == 0
    return tmp_path / "specs" / "abl-image.json"


class TestParseSymbols:
    """An entry this cannot read becomes an assertion that never runs."""

    def test_one_pair(self) -> None:
        assert capture_cli.parse_symbols("pkg.mod:thing") == [
            {"module": "pkg.mod", "attribute": "thing"}
        ]

    def test_several_pairs_keep_their_order(self) -> None:
        parsed = capture_cli.parse_symbols("a:one, b:two")
        assert [check["module"] for check in parsed] == ["a", "b"]

    @pytest.mark.parametrize("raw", ["pkg.mod", ":thing", "pkg.mod:", "  :  "])
    def test_a_malformed_entry_is_refused(self, raw: str) -> None:
        with pytest.raises(ValueError, match="module:attribute"):
            _ = capture_cli.parse_symbols(raw)

    def test_an_empty_list_is_refused(self) -> None:
        """An image asserting nothing cannot detect its own staleness."""
        with pytest.raises(ValueError, match="at least one"):
            _ = capture_cli.parse_symbols("  ,  ")


class TestCapture:
    """What the command writes, read back through the contract."""

    def test_the_written_spec_decodes(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        out = _capture(tmp_path, fake_run)
        spec = decode_image_spec(load_json_str(out.read_text(encoding="utf-8")))
        assert spec["git_commit"] == _COMMIT

    def test_third_party_becomes_requirements(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        spec = decode_image_spec(
            load_json_str(_capture(tmp_path, fake_run).read_text(encoding="utf-8"))
        )
        assert spec["requirements"] == ["torch==2.6.0+cu124", "transformers==4.46.3"]

    def test_first_party_becomes_wheels(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        spec = decode_image_spec(
            load_json_str(_capture(tmp_path, fake_run).read_text(encoding="utf-8"))
        )
        assert spec["wheels"] == [
            "model_trainer_server-0.1.0-py3-none-any.whl",
            "platform_core-0.1.0-py3-none-any.whl",
        ]

    def test_pinned_packages_become_the_version_assertions(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Reused, not restated: a second list is a second thing to keep."""
        spec = decode_image_spec(
            load_json_str(_capture(tmp_path, fake_run).read_text(encoding="utf-8"))
        )
        assert spec["expected_versions"] == {"torch": "2.6.0+cu124", "transformers": "4.46.3"}

    def test_the_labels_name_the_project_and_its_source_environment(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        spec = decode_image_spec(
            load_json_str(_capture(tmp_path, fake_run).read_text(encoding="utf-8"))
        )
        assert spec["labels"] == {
            "org.corvis.project": "abl",
            "org.corvis.env-source": "/pub/wagnera3/envs/abl-pinned",
        }

    def test_it_probes_the_projects_own_environment(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _ = _capture(tmp_path, fake_run)
        assert any(
            "/pub/wagnera3/envs/abl-pinned/bin/python" in " ".join(call.argv)
            for call in fake_run.calls
        )

    def test_it_creates_the_output_directory(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        out = _capture(tmp_path, fake_run)
        assert out.parent.is_dir()

    def test_it_reports_what_it_probed(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _ = _capture(tmp_path, fake_run)
        assert emitted[0] == (
            "probed /pub/wagnera3/envs/abl-pinned: 5 distribution(s), 2 requirement(s), 2 wheel(s)"
        )


class TestItRefusesRatherThanWrites:
    """A spec that cannot be trusted must not reach a build."""

    def test_an_env_prefix_under_a_bind_mounted_root_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _workspace(tmp_path)
        fake_run.add("bin/python", stdout=_FREEZE)
        with pytest.raises(JSONTypeError, match="bind-mounts over"):
            _ = capture_cli.main(_args(tmp_path, **{"--env-prefix": "/pub/envs/abl"}))
        assert not (tmp_path / "specs").exists()

    def test_a_first_party_name_matching_nothing_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _workspace(tmp_path)
        fake_run.add("bin/python", stdout=_FREEZE)
        with pytest.raises(AppError) as excinfo:
            _ = capture_cli.main(_args(tmp_path, **{"--first-party": "typo-package"}))
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PROBE_UNREADABLE

    def test_an_unknown_project_is_refused(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _workspace(tmp_path)
        with pytest.raises(AppError) as excinfo:
            _ = capture_cli.main(_args(tmp_path, **{"--project": "nope"}))
        assert excinfo.value.code is Hpc3ErrorCode.WORKSPACE_PROJECT_UNKNOWN

    def test_a_missing_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        _workspace(tmp_path)
        with pytest.raises(ValueError, match="--symbols"):
            _ = capture_cli.main(
                [
                    "--config",
                    str(tmp_path / "hpc3.json"),
                    "--project",
                    "abl",
                    "--commit",
                    _COMMIT,
                    "--base-image",
                    "python:3.11-slim",
                    "--env-prefix",
                    "/opt/env",
                    "--first-party",
                    "platform_core",
                    "--extra-index-url",
                    "https://example.invalid",
                    "--out",
                    str(tmp_path / "s.json"),
                ]
            )


class TestEntrypoint:
    """``entrypoint`` reads sys.argv, so it is exercised for real."""

    def test_it_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str], argv: list[str]
    ) -> None:
        _workspace(tmp_path)
        fake_run.add("bin/python", stdout=_FREEZE)
        argv[:] = ["prog", *_args(tmp_path)]

        with pytest.raises(SystemExit) as excinfo:
            capture_cli.entrypoint()
        assert excinfo.value.code == 0
        assert (tmp_path / "specs" / "abl-image.json").is_file()
