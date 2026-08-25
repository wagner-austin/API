"""Tests for turning a live environment into an image spec.

Both rules here exist because a hand transcription gets them wrong, and both
failures are silent. Pinning pip makes the build install a version and
replace it in an order nobody declared; capturing a first-party package as a
requirement makes the build resolve our name from a public index.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core.image_capture import (
    BUILD_PROVIDED,
    capture_layers,
    wheel_filename,
)

_FIRST_PARTY = frozenset({"platform_core", "model-trainer-server"})

_INSTALLED = {
    "torch": "2.6.0+cu124",
    "transformers": "4.46.3",
    "pip": "26.2.1",
    "setuptools": "84.0.0",
    "wheel": "0.48.0",
    "platform-core": "0.1.0",
    "model-trainer-server": "0.1.0",
}


class TestWheelFilename:
    """pip reports one spelling; the file on disk carries another."""

    def test_hyphens_become_underscores(self) -> None:
        assert wheel_filename("platform-core", "0.1.0") == ("platform_core-0.1.0-py3-none-any.whl")

    def test_the_underscore_spelling_gives_the_same_name(self) -> None:
        """A spec author types whichever spelling they saw last."""
        assert wheel_filename("platform_core", "0.1.0") == wheel_filename("platform-core", "0.1.0")

    def test_a_local_version_survives(self) -> None:
        assert wheel_filename("torch", "2.6.0+cu124") == "torch-2.6.0+cu124-py3-none-any.whl"


class TestCaptureLayers:
    """The split a hand transcription gets wrong."""

    def test_third_party_becomes_pinned_requirements(self) -> None:
        requirements, _ = capture_layers(_INSTALLED, _FIRST_PARTY)
        assert requirements == ["torch==2.6.0+cu124", "transformers==4.46.3"]

    def test_first_party_becomes_wheels(self) -> None:
        _, wheels = capture_layers(_INSTALLED, _FIRST_PARTY)
        assert wheels == [
            "model_trainer_server-0.1.0-py3-none-any.whl",
            "platform_core-0.1.0-py3-none-any.whl",
        ]

    def test_first_party_never_appears_as_a_requirement(self) -> None:
        """Otherwise the build resolves our name from a public index."""
        requirements, _ = capture_layers(_INSTALLED, _FIRST_PARTY)
        assert not any(line.startswith("platform-core") for line in requirements)
        assert not any(line.startswith("model-trainer-server") for line in requirements)

    @pytest.mark.parametrize("provided", sorted(BUILD_PROVIDED))
    def test_build_provided_packages_are_excluded(self, provided: str) -> None:
        """The build installs these before reading the requirements file."""
        requirements, _ = capture_layers(_INSTALLED, _FIRST_PARTY)
        assert not any(line.startswith(f"{provided}==") for line in requirements)

    def test_either_spelling_of_a_first_party_name_matches(self) -> None:
        by_hyphen = capture_layers(_INSTALLED, frozenset({"platform-core"}))
        by_underscore = capture_layers(_INSTALLED, frozenset({"platform_core"}))
        assert by_hyphen == by_underscore

    def test_output_is_sorted_so_a_recapture_diffs_cleanly(self) -> None:
        """A spec is a document people diff; reported order is not stable."""
        shuffled = dict(reversed(list(_INSTALLED.items())))
        assert capture_layers(shuffled, _FIRST_PARTY) == capture_layers(_INSTALLED, _FIRST_PARTY)

    def test_a_first_party_name_that_matches_nothing_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            _ = capture_layers(_INSTALLED, frozenset({"platform_core", "typo-package"}))
        assert excinfo.value.code is Hpc3ErrorCode.ENV_PROBE_UNREADABLE

    def test_the_refusal_names_the_missing_distribution(self) -> None:
        with pytest.raises(AppError, match="typo-package"):
            _ = capture_layers(_INSTALLED, frozenset({"typo-package"}))

    def test_an_environment_with_no_first_party_declared_captures_everything(self) -> None:
        """A project whose payload is a plain script has no wheels to build."""
        requirements, wheels = capture_layers(_INSTALLED, frozenset())
        assert wheels == []
        assert "platform-core==0.1.0" in requirements
