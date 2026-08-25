"""Tests for the rendered build script.

Its three judgment calls are the ones a build gets wrong quietly, so each has
an assertion: pipefail (a piped failure reporting success), an unconditional
commit stamp (an image claiming a previous build's provenance), and required
cache directories (a multi-gigabyte build filling a 50 GB home volume and
failing as though the build were broken).
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue

from hpc3.contracts.image import ImageSpec, decode_image_spec
from hpc3.core.image_build import render_build_script
from hpc3.core.image_layout import COMMIT_NAME, DEFINITION_NAME

_COMMIT = "d11efacd231ef92426eaf92483c33a8504bd770f"


def _spec(**overrides: JSONValue) -> ImageSpec:
    """Build a valid spec with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        The decoded spec.
    """
    base: dict[str, JSONValue] = {
        "base_image": "python:3.11.16-slim-bookworm",
        "env_prefix": "/opt/env",
        "git_commit": _COMMIT,
        "extra_index_urls": [],
        "requirements": ["torch==2.6.0+cu124"],
        "wheels": ["w-0.1.0-py3-none-any.whl"],
        "expected_versions": {"torch": "2.6.0+cu124"},
        "required_symbols": [{"module": "torch", "attribute": "__version__"}],
        "labels": {},
    }
    base.update(overrides)
    return decode_image_spec(base)


class TestFailureIsNotSwallowed:
    """A build that fails must not report success."""

    def test_pipefail_is_set(self) -> None:
        """Without it, a piped apptainer failure exits 0."""
        assert (
            "set -euo pipefail" in render_build_script(_spec(), image_name="abl.sif").splitlines()
        )

    def test_there_is_no_fallback_invocation(self) -> None:
        """A second attempt after a failed build would hide which one ran."""
        rendered = render_build_script(_spec(), image_name="abl.sif")
        assert rendered.count("apptainer build") == 1

    def test_missing_wheels_exit_before_building(self) -> None:
        lines = render_build_script(_spec(), image_name="abl.sif").splitlines()
        guard = lines.index("if [ ! -d wheels ]; then")
        build = lines.index(f"apptainer build --force abl.sif {DEFINITION_NAME}")
        assert guard < build


class TestProvenance:
    """The stamp must belong to this build."""

    def test_the_commit_is_written_every_run(self) -> None:
        line = f"printf '%s\\n' '{_COMMIT}' > {COMMIT_NAME}"
        assert line in render_build_script(_spec(), image_name="abl.sif").splitlines()

    def test_the_commit_is_written_before_the_build(self) -> None:
        lines = render_build_script(_spec(), image_name="abl.sif").splitlines()
        stamp = lines.index(f"printf '%s\\n' '{_COMMIT}' > {COMMIT_NAME}")
        build = lines.index(f"apptainer build --force abl.sif {DEFINITION_NAME}")
        assert stamp < build

    def test_a_different_commit_changes_the_stamp(self) -> None:
        rendered = render_build_script(_spec(git_commit="deadbeef"), image_name="abl.sif")
        assert "'deadbeef'" in rendered
        assert _COMMIT not in rendered


class TestBuildEnvironment:
    """Defaults under $HOME would fill a 50 GB volume."""

    def test_cache_and_tmp_are_required_not_defaulted(self) -> None:
        rendered = render_build_script(_spec(), image_name="abl.sif")
        assert 'APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:?' in rendered
        assert 'APPTAINER_TMPDIR="${APPTAINER_TMPDIR:?' in rendered

    def test_the_image_name_reaches_the_build_and_the_digest(self) -> None:
        rendered = render_build_script(_spec(), image_name="other.sif").splitlines()
        assert f"apptainer build --force other.sif {DEFINITION_NAME}" in rendered
        assert "sha256sum other.sif" in rendered

    def test_it_carries_no_carriage_returns(self) -> None:
        """A CRLF script makes the kernel report the interpreter as missing."""
        assert "\r" not in render_build_script(_spec(), image_name="abl.sif")

    def test_it_ends_with_a_newline(self) -> None:
        assert render_build_script(_spec(), image_name="abl.sif").endswith("\n")
