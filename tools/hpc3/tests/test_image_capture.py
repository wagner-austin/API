"""Tests for turning a live environment into an image spec.

Both rules here exist because a hand transcription gets them wrong, and both
failures are silent. Pinning pip makes the build install a version and
replace it in an order nobody declared; capturing a first-party package as a
requirement makes the build resolve our name from a public index.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core.env_probe import InstalledDistribution
from hpc3.core.image_capture import (
    BUILD_PROVIDED,
    capture_layers,
    wheel_filename,
)

_FIRST_PARTY = frozenset({"platform_core", "model-trainer-server"})


def _reported(version: str, wheel_tag: str = "py3-none-any") -> InstalledDistribution:
    """Build one probe record.

    Args:
        version: Exact version.
        wheel_tag: PEP 425 tag; the pure-Python one unless a test needs
            otherwise.

    Returns:
        The record.
    """
    return InstalledDistribution(version=version, wheel_tag=wheel_tag)


_INSTALLED: dict[str, InstalledDistribution] = {
    "torch": _reported("2.6.0+cu124"),
    "transformers": _reported("4.46.3"),
    "pip": _reported("26.2.1"),
    "setuptools": _reported("84.0.0"),
    "wheel": _reported("0.48.0"),
    "platform-core": _reported("0.1.0"),
    "model-trainer-server": _reported("0.1.0"),
}


class TestWheelFilename:
    """pip reports one spelling; the file on disk carries another."""

    def test_hyphens_become_underscores(self) -> None:
        assert wheel_filename("platform-core", "0.1.0", "py3-none-any") == (
            "platform_core-0.1.0-py3-none-any.whl"
        )

    def test_the_underscore_spelling_gives_the_same_name(self) -> None:
        """A spec author types whichever spelling they saw last."""
        assert wheel_filename("platform_core", "0.1.0", "py3-none-any") == wheel_filename(
            "platform-core", "0.1.0", "py3-none-any"
        )

    def test_a_local_version_survives(self) -> None:
        assert (
            wheel_filename("torch", "2.6.0+cu124", "cp311-cp311-linux_x86_64")
            == "torch-2.6.0+cu124-cp311-cp311-linux_x86_64.whl"
        )


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
        shuffled: dict[str, InstalledDistribution] = {
            name: _INSTALLED[name] for name in reversed(list(_INSTALLED))
        }
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


class TestTheWheelTagIsReadNotAssumed:
    """The bound the old constant named, and the project that crossed it.

    ``WHEEL_TAG`` was the literal "py3-none-any" and its docstring said a
    project shipping a compiled extension would need its real tag. cleargbm is
    that project: ``cleargbm_rs`` is compiled Rust, so its wheel is
    ``cp311-cp311-linux_x86_64``, and the captured spec named a file that does
    not exist on disk.
    """

    def test_a_compiled_first_party_wheel_keeps_its_real_tag(self) -> None:
        """The cleargbm case, which the assumed tag got wrong."""
        installed: dict[str, InstalledDistribution] = {
            "cleargbm-rs": _reported("0.1.0", "cp311-cp311-linux_x86_64"),
        }

        _, wheels = capture_layers(installed, frozenset({"cleargbm_rs"}))

        assert wheels == ["cleargbm_rs-0.1.0-cp311-cp311-linux_x86_64.whl"]

    def test_a_pure_python_first_party_wheel_still_gets_its_own_tag(self) -> None:
        """Read, not assumed -- even when the answer is the old constant."""
        installed: dict[str, InstalledDistribution] = {
            "platform-core": _reported("0.1.0", "py3-none-any"),
        }

        _, wheels = capture_layers(installed, frozenset({"platform_core"}))

        assert wheels == ["platform_core-0.1.0-py3-none-any.whl"]

    def test_a_first_party_distribution_with_no_tag_is_refused(self) -> None:
        """Naming a file that will not exist is the failure being prevented.

        A distribution with no WHEEL metadata was not installed from a wheel.
        The staging step would fail on a missing path and say only that; this
        says which distribution and why.
        """
        installed: dict[str, InstalledDistribution] = {
            "platform-core": _reported("0.1.0", ""),
        }

        with pytest.raises(AppError) as excinfo:
            _ = capture_layers(installed, frozenset({"platform_core"}))

        assert excinfo.value.code is Hpc3ErrorCode.WHEEL_TAG_UNKNOWN
        assert "platform-core" in excinfo.value.message

    def test_a_third_party_distribution_with_no_tag_is_fine(self) -> None:
        """It becomes a requirement line, not a file; conda packages look like this."""
        installed: dict[str, InstalledDistribution] = {
            "numpy": _reported("2.3.5", ""),
            "platform-core": _reported("0.1.0", "py3-none-any"),
        }

        requirements, _ = capture_layers(installed, frozenset({"platform_core"}))

        assert requirements == ["numpy==2.3.5"]
