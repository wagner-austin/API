"""Tests for platform_devpost.__init__ module functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform_devpost import (
    _find_monorepo_root,
    find_hackathons,
    get_codebase_profile,
)
from platform_devpost.testing import (
    FakeDevpostClient,
    hooks,
    make_fake_hackathon,
    make_fake_profile,
    make_fake_theme,
    make_interest_filter,
)
from platform_devpost.types import CodebaseProfile


class TestFindMonorepoRoot:
    """Tests for _find_monorepo_root function."""

    def test_finds_root_with_libs_dir(self, tmp_path: Path) -> None:
        """Test finding monorepo root with libs directory."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = _find_monorepo_root(subdir)

        assert result == tmp_path

    def test_finds_root_from_nested_path(self, tmp_path: Path) -> None:
        """Test finding monorepo root from deeply nested path."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)

        result = _find_monorepo_root(nested)

        assert result == tmp_path

    def test_raises_if_no_libs_dir(self, tmp_path: Path) -> None:
        """Test raises RuntimeError if libs directory not found."""
        with pytest.raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
            _find_monorepo_root(tmp_path)


class TestFindHackathons:
    """Tests for find_hackathons function."""

    def test_find_without_interests(self, tmp_path: Path) -> None:
        """Test find_hackathons without interest filter."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        h1 = make_fake_hackathon(id=1)
        h2 = make_fake_hackathon(id=2)
        fake_client = FakeDevpostClient(hackathons=(h1, h2))
        hooks.devpost_client = lambda: fake_client
        hooks.profile_scanner = lambda root: make_fake_profile(
            capabilities=(),
            technologies=(),
            frameworks=(),
        )

        result = find_hackathons(root=tmp_path)

        assert len(result) == 2

    def test_find_with_interests(self, tmp_path: Path) -> None:
        """Test find_hackathons with interest filter."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        h1 = make_fake_hackathon(id=1, open_state="open")
        h2 = make_fake_hackathon(id=2, open_state="ended")
        fake_client = FakeDevpostClient(hackathons=(h1, h2))
        hooks.devpost_client = lambda: fake_client
        hooks.profile_scanner = lambda root: make_fake_profile()

        interests = make_interest_filter(states=("open",))
        result = find_hackathons(interests=interests, root=tmp_path)

        assert len(result) == 1
        assert result[0].hackathon.id == 1

    def test_find_with_codebase_matching(self, tmp_path: Path) -> None:
        """Test find_hackathons with codebase matching."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        theme = make_fake_theme(name="Machine Learning")
        h = make_fake_hackathon(id=1, themes=(theme,))
        fake_client = FakeDevpostClient(hackathons=(h,))
        hooks.devpost_client = lambda: fake_client
        hooks.profile_scanner = lambda root: make_fake_profile(
            technologies=("python",),
            frameworks=(),
        )

        result = find_hackathons(match_codebase=True, root=tmp_path)

        assert len(result) == 1
        assert result[0].match_score >= 0.0

    def test_find_with_min_score_filter(self, tmp_path: Path) -> None:
        """Test find_hackathons filters by minimum score."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        h1 = make_fake_hackathon(id=1)  # No themes, will have 0 score
        h2 = make_fake_hackathon(id=2)
        fake_client = FakeDevpostClient(hackathons=(h1, h2))
        hooks.devpost_client = lambda: fake_client
        hooks.profile_scanner = lambda root: make_fake_profile()

        result = find_hackathons(min_match_score=0.5, root=tmp_path)

        # Both have 0 score because no themes, so both filtered out
        assert len(result) == 0

    def test_find_without_codebase_matching(self, tmp_path: Path) -> None:
        """Test find_hackathons without codebase matching."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        h1 = make_fake_hackathon(id=1)
        h2 = make_fake_hackathon(id=2)
        fake_client = FakeDevpostClient(hackathons=(h1, h2))
        hooks.devpost_client = lambda: fake_client

        result = find_hackathons(match_codebase=False, root=tmp_path)

        assert len(result) == 2
        # All have 0 score and new_territory when not matching
        for match in result:
            assert match.match_score == 0.0
            assert match.recommendation == "new_territory"

    def test_find_with_auto_detected_root(self) -> None:
        """Test find_hackathons auto-detects monorepo root."""
        h1 = make_fake_hackathon(id=1)
        fake_client = FakeDevpostClient(hackathons=(h1,))
        hooks.devpost_client = lambda: fake_client
        hooks.profile_scanner = lambda root: make_fake_profile()

        # Call without root - should auto-detect
        result = find_hackathons(match_codebase=True)

        assert len(result) == 1


class TestGetCodebaseProfile:
    """Tests for get_codebase_profile function."""

    def test_get_profile_with_explicit_root(self, tmp_path: Path) -> None:
        """Test get_codebase_profile with explicit root."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        expected_profile = make_fake_profile(
            technologies=("python", "rust"),
        )
        hooks.profile_scanner = lambda root: expected_profile

        result = get_codebase_profile(root=tmp_path)

        assert result == expected_profile

    def test_get_profile_calls_scanner(self, tmp_path: Path) -> None:
        """Test get_codebase_profile calls profile_scanner hook."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        services_dir = tmp_path / "services"
        services_dir.mkdir()

        called_with: list[Path] = []

        def capturing_scanner(root: Path) -> CodebaseProfile:
            called_with.append(root)
            return make_fake_profile()

        hooks.profile_scanner = capturing_scanner

        get_codebase_profile(root=tmp_path)

        assert len(called_with) == 1
        assert called_with[0] == tmp_path

    def test_get_profile_auto_detects_root(self) -> None:
        """Test get_codebase_profile auto-detects monorepo root."""
        called_with: list[Path] = []

        def capturing_scanner(root: Path) -> CodebaseProfile:
            called_with.append(root)
            return make_fake_profile()

        hooks.profile_scanner = capturing_scanner

        # Call without root - it should auto-detect
        profile = get_codebase_profile()

        # Should have called scanner with detected path
        assert len(called_with) == 1
        # The detected path should contain "libs" directory
        assert (called_with[0] / "libs").exists()
        # Verify the profile was returned
        assert profile.technologies == ("python",)
