"""Tests for environment identity, not environment existence.

The failure these guard against was live on the cluster while they were
written: ``/pub/wagnera3/envs/abl`` and ``/pub/wagnera3/envs/abl-pinned`` both
existed, both passed the directory check, and they differ by transformers
4.46.3 vs 5.15.1 and torch 2.6.0+cu124 vs 2.11.0+cu128. Seven characters in a
path, a major version underneath, and a run that completes with a number not
comparable to the arms it was meant to extend.

The version strings below are the measured ones from those two environments.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.image import ImageReference
from hpc3.contracts.pins import encode_pinned_packages, normalise_name, require_pinned_packages
from hpc3.core.env_probe import (
    check_pins,
    parse_installed,
    probe_command,
    verify_env_packages,
)
from tests.conftest import ABL_PINNED_DISTRIBUTIONS, FakeRun

_UNPINNED_ENV = "torch==2.11.0+cu128\ntransformers==5.15.1\nnumpy==2.3.1\n"
"""What ``/pub/wagnera3/envs/abl`` reports -- the environment one typo away."""

_ABL_PINS = {"torch": "2.6.0+cu124", "transformers": "4.46.3"}

_IMAGE = ImageReference(
    path="/pub/wagnera3/images/v3/abl.sif",
    sha256="3bd2e694857821c383c5e90b1c23b63c706ece20069a6bf34431512ef1c041d4",
    binds=["/pub/wagnera3"],
)
"""The v3 ablation image, whose environment is at the container path /opt/env.

A real reference rather than a placeholder: /opt/env is exactly the shape
that exists nowhere on the cluster filesystem, which is the case a host
probe answers wrongly rather than not at all.
"""


class TestNormaliseName:
    def test_the_underscore_spelling_matches_the_hyphen_spelling(self) -> None:
        """importlib.metadata reports whatever the distribution called itself."""
        assert normalise_name("typing_extensions") == normalise_name("typing-extensions")

    def test_case_is_folded(self) -> None:
        assert normalise_name("Torch") == "torch"

    def test_dots_are_separators_too(self) -> None:
        assert normalise_name("zope.interface") == "zope-interface"

    def test_a_run_of_separators_collapses(self) -> None:
        assert normalise_name("a__.-b") == "a-b"


class TestProbeCommand:
    def test_it_runs_the_environments_own_interpreter_by_absolute_path(self) -> None:
        """Through PATH it would answer about whatever the login shell activated."""
        assert probe_command("/pub/envs/abl-pinned").startswith(
            "'/pub/envs/abl-pinned/bin/python' -c "
        )

    def test_the_command_carries_no_newline(self) -> None:
        """A newline would be split by the remote shell, not by Python."""
        assert "\n" not in probe_command("/pub/envs/abl-pinned")

    def test_it_asks_importlib_metadata_rather_than_importing(self) -> None:
        """A package present but broken must still be reported, not raise."""
        assert "importlib.metadata" in probe_command("/e")


class TestParseInstalled:
    def test_it_reads_the_measured_environment(self) -> None:
        installed = parse_installed(ABL_PINNED_DISTRIBUTIONS)
        assert installed["torch"]["version"] == "2.6.0+cu124"
        assert installed["transformers"]["version"] == "4.46.3"

    def test_names_are_normalised_on_the_way_in(self) -> None:
        assert "typing-extensions" in parse_installed(ABL_PINNED_DISTRIBUTIONS)

    def test_a_local_version_suffix_survives_intact(self) -> None:
        """+cu124 vs +cu128 is the whole distinction; it must not be trimmed."""
        assert parse_installed("torch==2.6.0+cu124\n")["torch"]["version"] == "2.6.0+cu124"

    def test_blank_lines_are_skipped(self) -> None:
        assert parse_installed("\ntorch==2.6.0\n\n") == {
            "torch": {"version": "2.6.0", "wheel_tag": ""}
        }

    def test_a_line_without_the_separator_is_skipped(self) -> None:
        assert parse_installed("some warning\ntorch==2.6.0\n") == {
            "torch": {"version": "2.6.0", "wheel_tag": ""}
        }

    def test_output_with_no_distribution_at_all_is_refused(self) -> None:
        """A traceback must not read as 'nothing is installed'."""
        with pytest.raises(AppError) as excinfo:
            parse_installed("Traceback (most recent call last):\n  ImportError\n")
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PROBE_UNREADABLE

    def test_empty_output_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_installed("")
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PROBE_UNREADABLE


class TestCheckPins:
    def test_the_pinned_environment_passes(self) -> None:
        check_pins(parse_installed(ABL_PINNED_DISTRIBUTIONS), _ABL_PINS, env_path="/e")

    def test_the_environment_one_typo_away_is_refused(self) -> None:
        """The measured case: envs/abl instead of envs/abl-pinned."""
        with pytest.raises(AppError) as excinfo:
            check_pins(parse_installed(_UNPINNED_ENV), _ABL_PINS, env_path="/pub/envs/abl")
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PACKAGE_MISMATCH

    def test_the_message_names_both_versions_and_the_environment(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_pins(parse_installed(_UNPINNED_ENV), _ABL_PINS, env_path="/pub/envs/abl")
        assert "/pub/envs/abl" in excinfo.value.message
        assert "2.11.0+cu128" in excinfo.value.message
        assert "2.6.0+cu124" in excinfo.value.message

    def test_a_cuda_build_difference_alone_is_a_mismatch(self) -> None:
        """Same torch, different CUDA build, different kernels."""
        with pytest.raises(AppError) as excinfo:
            check_pins(
                {"torch": {"version": "2.6.0+cu118", "wheel_tag": ""}},
                {"torch": "2.6.0+cu124"},
                env_path="/e",
            )
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PACKAGE_MISMATCH

    def test_an_absent_package_is_refused_and_says_so(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_pins(
                {"torch": {"version": "2.6.0+cu124", "wheel_tag": ""}},
                _ABL_PINS,
                env_path="/e",
            )
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PACKAGE_MISMATCH
        assert "does not have transformers installed" in excinfo.value.message

    def test_extra_packages_in_the_environment_are_not_a_problem(self) -> None:
        """A pin declares what must be there, not what must not."""
        installed = parse_installed(ABL_PINNED_DISTRIBUTIONS)
        check_pins(installed, {"torch": "2.6.0+cu124"}, env_path="/e")

    def test_no_pins_accepts_anything(self) -> None:
        check_pins(parse_installed(_UNPINNED_ENV), {}, env_path="/e")


class TestVerifyEnvPackages:
    def test_no_pins_makes_no_remote_call(self, fake_run: FakeRun) -> None:
        """A project with a compiled payload should not pay for a round trip."""
        verify_env_packages("hpc3", "/e", {}, image=None)
        assert fake_run.calls == []

    def test_pins_are_checked_against_the_live_environment(self, fake_run: FakeRun) -> None:
        fake_run.add("importlib.metadata", stdout=ABL_PINNED_DISTRIBUTIONS)
        verify_env_packages("hpc3", "/pub/envs/abl-pinned", _ABL_PINS, image=None)
        assert len(fake_run.calls) == 1

    def test_the_wrong_environment_is_refused(self, fake_run: FakeRun) -> None:
        fake_run.add("importlib.metadata", stdout=_UNPINNED_ENV)
        with pytest.raises(AppError) as excinfo:
            verify_env_packages("hpc3", "/pub/envs/abl", _ABL_PINS, image=None)
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PACKAGE_MISMATCH

    def test_an_images_pins_are_read_from_inside_the_image(self, fake_run: FakeRun) -> None:
        """The environment being pinned lives in the sif, not on the cluster.

        Probing the host for a container path returns an empty listing, which
        `check_pins` reads as "torch is not installed" -- a confident and
        wrong diagnosis of the image, produced by asking the wrong machine.
        """
        fake_run.add("importlib.metadata", stdout=ABL_PINNED_DISTRIBUTIONS)
        verify_env_packages("hpc3", "/opt/env", _ABL_PINS, image=_IMAGE)
        sent = fake_run.calls[0].remote_command
        assert sent.startswith("module load apptainer/1.4.5 && apptainer exec ")
        assert '--bind "/pub/wagnera3:/pub/wagnera3"' in sent
        assert '"/pub/wagnera3/images/v3/abl.sif"' in sent
        assert "/opt/env/bin/python" in sent

    def test_an_images_wrong_environment_is_still_refused(self, fake_run: FakeRun) -> None:
        """Running the probe in the right place does not soften its verdict."""
        fake_run.add("importlib.metadata", stdout=_UNPINNED_ENV)
        with pytest.raises(AppError) as excinfo:
            verify_env_packages("hpc3", "/opt/env", _ABL_PINS, image=_IMAGE)
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PACKAGE_MISMATCH


class TestRequirePinnedPackages:
    def test_it_reads_and_normalises(self) -> None:
        obj: dict[str, JSONValue] = {"p": {"Typing_Extensions": "4.12.2"}}
        assert require_pinned_packages(obj, "p") == {"typing-extensions": "4.12.2"}

    def test_an_empty_map_is_valid(self) -> None:
        """A compiled payload has no Python packages to pin."""
        assert require_pinned_packages({"p": {}}, "p") == {}

    def test_a_missing_field_is_refused(self) -> None:
        """Required even when empty: an unasked question is not an answer."""
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            require_pinned_packages({}, "p")

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            require_pinned_packages({"p": ["torch==2.6.0"]}, "p")

    def test_a_non_string_version_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="version strings"):
            require_pinned_packages({"p": {"torch": 2}}, "p")

    def test_an_empty_version_is_refused(self) -> None:
        """It would compare equal to nothing and fail every run."""
        with pytest.raises(JSONTypeError, match="empty name or version"):
            require_pinned_packages({"p": {"torch": ""}}, "p")

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="empty name or version"):
            require_pinned_packages({"p": {"": "2.6.0"}}, "p")

    def test_it_round_trips_through_encode(self) -> None:
        pinned = require_pinned_packages({"p": {"torch": "2.6.0+cu124"}}, "p")
        assert encode_pinned_packages(pinned) == {"torch": "2.6.0+cu124"}
