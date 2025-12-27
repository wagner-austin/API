"""Tests for platform_kaggle public API."""

from __future__ import annotations

from pathlib import Path

from platform_kaggle import (
    InterestFilter,
    find_competitions,
    get_codebase_profile,
    hooks,
    make_fake_competition,
    make_fake_competition_pages,
)
from platform_kaggle.testing import FakeKaggleClient, FakeKagglePageFetcher


class TestFindCompetitions:
    """Tests for find_competitions function."""

    def test_find_competitions_without_interests(self, tmp_path: Path) -> None:
        """Test find_competitions returns all competitions when no interests."""
        competitions = (
            make_fake_competition(ref="comp1", title="Competition 1"),
            make_fake_competition(ref="comp2", title="Competition 2"),
        )
        fake_client = FakeKaggleClient(competitions=competitions)
        hooks.kaggle_client = lambda: fake_client

        matches = find_competitions(codebase_root=tmp_path, fetch_descriptions=False)
        assert len(matches) == 2

    def test_find_competitions_with_interests(self, tmp_path: Path) -> None:
        """Test find_competitions filters by interests."""
        competitions = (
            make_fake_competition(ref="comp1", tags=("tabular", "classification")),
            make_fake_competition(ref="comp2", tags=("image", "computer-vision")),
        )
        fake_client = FakeKaggleClient(competitions=competitions)
        hooks.kaggle_client = lambda: fake_client

        interests = InterestFilter(
            include_tags=("tabular",),
            exclude_tags=(),
            min_reward=None,
            categories=None,
        )
        matches = find_competitions(
            interests=interests, codebase_root=tmp_path, fetch_descriptions=False
        )
        assert len(matches) == 1
        assert matches[0].competition.ref == "comp1"

    def test_find_competitions_with_codebase_matching(self, tmp_path: Path) -> None:
        """Test find_competitions matches against codebase capabilities."""
        # Create a minimal pyproject.toml
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "test_lib"
        lib_dir.mkdir()
        pyproject = lib_dir / "pyproject.toml"
        pyproject.write_text("""
[tool.poetry]
name = "test-lib"

[tool.poetry.dependencies]
python = "^3.11"
lightgbm = "^4.0.0"
""")

        competitions = (make_fake_competition(ref="comp1", tags=("tabular", "classification")),)
        fake_client = FakeKaggleClient(competitions=competitions)
        hooks.kaggle_client = lambda: fake_client

        matches = find_competitions(
            match_codebase=True, codebase_root=tmp_path, fetch_descriptions=False
        )
        assert len(matches) == 1
        # Should have matched capabilities from lightgbm detection
        assert matches[0].match_score > 0.0

    def test_find_competitions_without_codebase_matching(self, tmp_path: Path) -> None:
        """Test find_competitions returns default matches when match_codebase=False."""
        competitions = (make_fake_competition(ref="comp1", title="Competition 1"),)
        fake_client = FakeKaggleClient(competitions=competitions)
        hooks.kaggle_client = lambda: fake_client

        matches = find_competitions(match_codebase=False, codebase_root=tmp_path)
        assert len(matches) == 1
        # Default match has score 0.5 and recommendation "good_fit"
        assert matches[0].match_score == 0.5
        assert matches[0].recommendation == "good_fit"
        assert matches[0].matched_capabilities == ()
        assert matches[0].missing_capabilities == ()

    def test_find_competitions_min_match_score(self, tmp_path: Path) -> None:
        """Test find_competitions filters by minimum match score."""
        # Create libs with no capabilities
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "empty_lib"
        lib_dir.mkdir()
        (lib_dir / "pyproject.toml").write_text("""
[tool.poetry]
name = "empty-lib"

[tool.poetry.dependencies]
python = "^3.11"
""")

        # Competition requires many capabilities we don't have
        competitions = (
            make_fake_competition(
                ref="comp1",
                tags=("deep-learning", "pytorch", "gpu"),
            ),
        )
        fake_client = FakeKaggleClient(competitions=competitions)
        hooks.kaggle_client = lambda: fake_client

        # Set up page fetcher with empty pages
        pages = make_fake_competition_pages(description="A deep learning competition")
        fake_fetcher = FakeKagglePageFetcher(
            competition_ids={"comp1": 1},
            pages={1: pages},
        )
        hooks.page_fetcher = lambda: fake_fetcher

        # With high min_score, no matches
        matches = find_competitions(
            match_codebase=True,
            min_match_score=0.9,
            codebase_root=tmp_path,
        )
        assert len(matches) == 0

    def test_find_competitions_auto_detect_root(self) -> None:
        """Test find_competitions auto-detects codebase root when not provided."""
        # This will use the real codebase root (parent of parent of ...)
        competitions = (make_fake_competition(ref="comp1"),)
        fake_client = FakeKaggleClient(competitions=competitions)
        hooks.kaggle_client = lambda: fake_client

        # Set up page fetcher
        pages = make_fake_competition_pages(description="A test competition")
        fake_fetcher = FakeKagglePageFetcher(
            competition_ids={"comp1": 1},
            pages={1: pages},
        )
        hooks.page_fetcher = lambda: fake_fetcher

        # Don't pass codebase_root, let it auto-detect
        matches = find_competitions(match_codebase=True)
        # Should return our competition
        assert len(matches) == 1
        assert matches[0].competition.ref == "comp1"

    def test_find_competitions_active_only_false_skips_deadline_filter(
        self, tmp_path: Path
    ) -> None:
        """Test find_competitions with active_only=False includes expired competitions."""
        # Use a past deadline
        competitions = (make_fake_competition(ref="expired", deadline="2020-01-01"),)
        fake_client = FakeKaggleClient(competitions=competitions)
        hooks.kaggle_client = lambda: fake_client

        # With active_only=False, should include expired competition
        matches = find_competitions(
            interests=None,
            match_codebase=False,
            active_only=False,
            codebase_root=tmp_path,
        )
        assert len(matches) == 1
        assert matches[0].competition.ref == "expired"

    def test_find_competitions_with_fetch_descriptions(self, tmp_path: Path) -> None:
        """Test find_competitions fetches descriptions for better matching."""
        # Create libs with transformers capability
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "test_lib"
        lib_dir.mkdir()
        (lib_dir / "pyproject.toml").write_text("""
[tool.poetry]
name = "test-lib"

[tool.poetry.dependencies]
python = "^3.11"
transformers = "^4.0.0"
""")

        # Competition with tags that would normally match transformers
        competitions = (make_fake_competition(ref="gemma-comp", tags=("text",)),)
        fake_client = FakeKaggleClient(competitions=competitions)
        hooks.kaggle_client = lambda: fake_client

        # Pages that mention Gemma (hard requirement we don't have)
        pages = make_fake_competition_pages(
            description="Use Gemma 3n to build a mobile app",
        )
        # FakeKagglePageFetcher uses competition_ids (slug->id) and pages (id->pages)
        fake_fetcher = FakeKagglePageFetcher(
            competition_ids={"gemma-comp": 12345},
            pages={12345: pages},
        )
        hooks.page_fetcher = lambda: fake_fetcher

        # With fetch_descriptions=True, should detect hard requirement
        matches = find_competitions(
            match_codebase=True,
            codebase_root=tmp_path,
            fetch_descriptions=True,
        )
        assert len(matches) == 1
        # Should have low score due to missing gemma_model hard requirement
        assert matches[0].match_score <= 0.3
        assert "gemma_model" in matches[0].missing_capabilities


class TestGetCodebaseProfile:
    """Tests for get_codebase_profile function."""

    def test_get_codebase_profile_with_root(self, tmp_path: Path) -> None:
        """Test get_codebase_profile with explicit root."""
        # Create minimal structure
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()
        lib_dir = libs_dir / "test_lib"
        lib_dir.mkdir()
        (lib_dir / "pyproject.toml").write_text("""
[tool.poetry]
name = "test-lib"

[tool.poetry.dependencies]
python = "^3.11"
pandas = "^2.0.0"
""")

        profile = get_codebase_profile(root=tmp_path)
        # pandas implies csv and excel data formats
        assert "csv" in profile.data_formats
        assert "excel" in profile.data_formats

    def test_get_codebase_profile_auto_detect_root(self) -> None:
        """Test get_codebase_profile auto-detects codebase root."""
        # Don't pass root, let it auto-detect
        # This uses the real monorepo root
        profile = get_codebase_profile()
        # The real codebase has lightgbm in libs/cleargbm
        assert "lightgbm" in profile.ml_backends

    def test_get_codebase_profile_empty_libs(self, tmp_path: Path) -> None:
        """Test get_codebase_profile with empty libs directory."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        profile = get_codebase_profile(root=tmp_path)
        assert profile.capabilities == ()
        assert profile.ml_backends == ()
