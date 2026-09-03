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
        "torch==2.6.0+cu124==",
        "transformers==4.46.3==",
        "pip==26.2.1==",
        "platform-core==0.1.0==py3-none-any",
        "model-trainer-server==0.1.0==py3-none-any",
    ]
)
"""What the probe prints: ``name==version==wheel_tag``, one per line.

The third-party lines carry no tag on purpose. Nothing needs one for them --
they become ``==`` requirement lines the build resolves from an index -- and
a conda-installed package genuinely has no ``WHEEL`` metadata to report. Only
the first-party distributions, which become wheel FILES the staging step has
to find, carry a tag.
"""

_COMMIT = "d11efacd231ef92426eaf92483c33a8504bd770f"


BASE_IMAGE = "python:3.11.16-slim-bookworm@sha256:" + "b3" * 32
"""A digest-pinned base, because the spec contract refuses a bare tag.

Composed rather than written out so the line fits, and so the 64-character
digest is obviously synthetic rather than mistaken for a real one.
"""


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
        "--base-image": BASE_IMAGE,
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


class TestParseExtraIndexUrls:
    """No extra index is an answer, and more than one has to be sayable.

    Unlike ``--symbols``, emptiness here is legitimate: a project drawing
    every wheel from PyPI genuinely has no additional index. The flag used
    to be mandatory AND single-valued, which made both of those unsayable.
    """

    def test_the_flag_being_absent_is_no_extra_index(self) -> None:
        """``None`` is what the parser sees when the flag was never given."""
        assert capture_cli.parse_extra_index_urls(None) == []

    def test_one_index(self) -> None:
        assert capture_cli.parse_extra_index_urls("https://download.pytorch.org/whl/cu124") == [
            "https://download.pytorch.org/whl/cu124"
        ]

    def test_several_indexes_keep_their_order(self) -> None:
        """The field is ordered because pip consults them in order."""
        assert capture_cli.parse_extra_index_urls("https://a.invalid, https://b.invalid") == [
            "https://a.invalid",
            "https://b.invalid",
        ]

    @pytest.mark.parametrize("raw", ["", "   ", " , ", ",,"])
    def test_a_value_carrying_no_index_is_no_index(self, raw: str) -> None:
        """Separators without content say the same thing as omission."""
        assert capture_cli.parse_extra_index_urls(raw) == []


class TestOnboardingAProjectThatIsNotRegisteredYet:
    """The route that exists because registration now requires an image.

    A project cannot be registered until it declares an image digest, and the
    digest comes from a build driven by the spec this command writes. Reading
    the whole workspace to reach one string made that circular: a single
    unimaged project refused the read for the very command whose output would
    have fixed it. ``--env-path`` reads only the connection instead.
    """

    def test_a_workspace_whose_project_is_unimaged_is_still_readable(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The deadlock, asserted directly.

        The registry here would be refused by ``decode_workspace``. Onboarding
        never decodes it, so the capture that produces the missing image can
        run.

        Args:
            tmp_path: Working directory.
            fake_run: Recorded remote runner.
            emitted: Captured summary lines.
        """
        unimaged = project_config(gpu=None, partition="free")
        del unimaged["image"]
        _ = write_workspace(
            tmp_path / "hpc3.json", workspace_document(projects={"newcomer": unimaged})
        )
        fake_run.add("bin/python", stdout=_FREEZE)

        code = capture_cli.main(
            _args(tmp_path, **{"--project": "newcomer", "--env-path": "/pub/envs/newcomer"})
        )

        assert code == 0

    def test_the_probe_runs_on_the_host_not_inside_an_image(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """There is no image to enter yet; that is what onboarding means.

        Args:
            tmp_path: Working directory.
            fake_run: Recorded remote runner.
            emitted: Captured summary lines.
        """
        _workspace(tmp_path)
        fake_run.add("bin/python", stdout=_FREEZE)

        _ = capture_cli.main(_args(tmp_path, **{"--env-path": "/pub/envs/newcomer"}))

        probes = [c.remote_command for c in fake_run.calls if "bin/python" in c.remote_command]
        assert len(probes) == 1
        assert "apptainer" not in probes[0]
        assert "/pub/envs/newcomer" in probes[0]

    def test_the_spec_asserts_the_versions_it_captured(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """An unregistered project has declared no pins, so the probe is the source.

        Asserting nothing is not available: the spec contract refuses an empty
        ``expected_versions`` because an image that asserts no versions cannot
        detect its own staleness. Asserting what was captured is a real check
        that the build reproduced the environment it was taken from.

        Args:
            tmp_path: Working directory.
            fake_run: Recorded remote runner.
            emitted: Captured summary lines.
        """
        _workspace(tmp_path)
        fake_run.add("bin/python", stdout=_FREEZE)

        _ = capture_cli.main(_args(tmp_path, **{"--env-path": "/pub/envs/newcomer"}))

        spec = decode_image_spec(
            load_json_str((tmp_path / "specs" / "abl-image.json").read_text(encoding="utf-8"))
        )
        assert spec["expected_versions"] == {
            "torch": "2.6.0+cu124",
            "transformers": "4.46.3",
        }

    def test_a_registered_project_pinning_nothing_captures_what_it_has(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Not an onboarding case, and it was broken before this change.

        ``pinned_packages`` may legitimately be empty -- ``rusted`` declares
        exactly that, its payload being a compiled binary. Capturing such a
        project produced an empty ``expected_versions``, which the spec
        contract refuses, so the command failed on a valid registration.

        Args:
            tmp_path: Working directory.
            fake_run: Recorded remote runner.
            emitted: Captured summary lines.
        """
        config = project_config(env_path="/pub/envs/abl", pinned_packages={})
        _ = write_workspace(tmp_path / "hpc3.json", workspace_document(projects={"abl": config}))
        fake_run.add("bin/python", stdout=_FREEZE)

        assert capture_cli.main(_args(tmp_path)) == 0

        spec = decode_image_spec(
            load_json_str((tmp_path / "specs" / "abl-image.json").read_text(encoding="utf-8"))
        )
        assert spec["expected_versions"] == {
            "torch": "2.6.0+cu124",
            "transformers": "4.46.3",
        }

    def test_the_registered_route_still_reuses_the_projects_pins(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Without --env-path nothing about the version bump changes.

        Args:
            tmp_path: Working directory.
            fake_run: Recorded remote runner.
            emitted: Captured summary lines.
        """
        _workspace(tmp_path)
        fake_run.add("bin/python", stdout=_FREEZE)

        _ = capture_cli.main(_args(tmp_path))

        spec = decode_image_spec(
            load_json_str((tmp_path / "specs" / "abl-image.json").read_text(encoding="utf-8"))
        )
        assert spec["expected_versions"] != {}


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

    def test_omitting_the_index_flag_captures_no_extra_index(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The whole command runs without the flag, and records the truth.

        This is the case the old ``require_flag`` made impossible. A CPU-only
        project has no additional index, and the only value available to
        satisfy a mandatory flag was PyPI itself -- so the spec recorded the
        default index as an addition to the default index.
        """
        _workspace(tmp_path)
        fake_run.add("bin/python", stdout=_FREEZE)
        tokens = _args(tmp_path)
        flag_at = tokens.index("--extra-index-url")
        del tokens[flag_at : flag_at + 2]
        assert "--extra-index-url" not in tokens

        assert capture_cli.main(tokens) == 0

        out = tmp_path / "specs" / "abl-image.json"
        spec = decode_image_spec(load_json_str(out.read_text(encoding="utf-8")))
        assert spec["extra_index_urls"] == []

    def test_two_indexes_survive_the_round_trip_in_order(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The contract is a list; capture used to be able to write only one."""
        out = _capture(
            tmp_path,
            fake_run,
            **{"--extra-index-url": "https://first.invalid,https://second.invalid"},
        )
        spec = decode_image_spec(load_json_str(out.read_text(encoding="utf-8")))
        assert spec["extra_index_urls"] == ["https://first.invalid", "https://second.invalid"]

    def test_pinned_packages_become_the_version_assertions(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Reused, not restated: a second list is a second thing to keep."""
        spec = decode_image_spec(
            load_json_str(_capture(tmp_path, fake_run).read_text(encoding="utf-8"))
        )
        assert spec["expected_versions"] == {"torch": "2.6.0+cu124", "transformers": "4.46.3"}

    def test_the_project_is_a_field_and_the_label_names_the_source_environment(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """The project moved out of the free-form labels into a real field.

        It lived only in ``labels`` until 2026-09-03, where nothing requires
        or validates it, so every later command re-took the project from its
        own flag and the defence against a typo was a registry lookup a
        project mid-onboarding cannot satisfy. One source, typed once at
        capture, read by the renderer.

        Args:
            tmp_path: Working directory.
            fake_run: Recorded remote runner.
            emitted: Captured summary lines.
        """
        spec = decode_image_spec(
            load_json_str(_capture(tmp_path, fake_run).read_text(encoding="utf-8"))
        )

        assert spec["project"] == "abl"
        assert spec["labels"] == {"org.corvis.env-source": "/pub/wagnera3/envs/abl-pinned"}

    def test_it_probes_the_projects_own_environment(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        _ = _capture(tmp_path, fake_run)
        assert any(
            "/pub/wagnera3/envs/abl-pinned/bin/python" in " ".join(call.argv)
            for call in fake_run.calls
        )

    def test_an_imaged_projects_environment_is_probed_inside_the_image(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Because that is the only filesystem it is on.

        Once a project adopts an image its ``env_path`` is a container path:
        ``/opt/env`` exists inside the ``.sif`` and nowhere on the cluster. A
        host probe therefore fails on every project past its first build, so
        this command worked exactly once per project and every version bump
        since hand-edited ``git_commit`` in the spec it was meant to generate.
        """
        _ = _capture(tmp_path, fake_run)
        probes = [c.remote_command for c in fake_run.calls if "bin/python" in c.remote_command]
        assert len(probes) == 1
        assert "apptainer exec" in probes[0]

    def test_a_cpu_project_is_probed_inside_its_image_too(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """This asserted the opposite until 2026-09-03.

        A CPU project could then declare no image, so its ``env_path`` was a
        real host directory and the probe ran on the host. Every project
        declares an image now, so a registered project can never be imageless
        and its ``env_path`` is always a container path -- probing the host
        would look for ``/opt/env`` on the cluster filesystem, where it does
        not exist.
        """
        config: JSONValue = project_config(
            env_path="/opt/env",
            pinned_packages={"torch": "2.6.0+cu124", "transformers": "4.46.3"},
            gpu=None,
        )
        _ = write_workspace(tmp_path / "hpc3.json", workspace_document(projects={"abl": config}))
        fake_run.add("bin/python", stdout=_FREEZE)
        assert capture_cli.main(_args(tmp_path)) == 0

        probes = [c.remote_command for c in fake_run.calls if "bin/python" in c.remote_command]
        assert len(probes) == 1
        assert "apptainer" in probes[0]

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
