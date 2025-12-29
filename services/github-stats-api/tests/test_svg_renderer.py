from __future__ import annotations

from github_stats_api.api.schemas.stats import (
    CapabilitiesResponse,
    Capability,
    LanguageStats,
    UserStats,
)
from github_stats_api.svg_renderer import (
    _calculate_rank,
    _escape_xml,
    _format_number,
    build_capabilities_response,
    build_language_stats,
    build_user_stats,
    render_capabilities_card,
    render_langs_card,
    render_stats_card,
)


class TestBuildUserStats:
    """Tests for build_user_stats function."""

    def test_build_user_stats_basic(self) -> None:
        """Test building user stats from data."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
        }

        stats = build_user_stats(data)

        assert stats["username"] == "testuser"
        assert stats["name"] == "Test User"
        assert stats["total_commits"] == 100
        assert stats["total_prs"] == 20
        assert stats["total_issues"] == 10
        assert stats["total_stars"] == 50
        assert stats["total_contributions"] == 150
        assert stats["rank"] in ("S+", "S", "A+", "A", "B+", "B", "C")
        assert 0 <= stats["rank_percentile"] <= 100

    def test_build_user_stats_high_activity_gets_high_rank(self) -> None:
        """Test that high activity results in high rank."""
        data: dict[str, int | str] = {
            "login": "superstar",
            "name": "Super Star",
            "total_commits": 50000,
            "total_prs": 5000,
            "total_issues": 2000,
            "total_stars": 100000,
            "total_contributions": 60000,
        }

        stats = build_user_stats(data)

        assert stats["rank"] in ("S+", "S", "A+")

    def test_build_user_stats_low_activity_gets_low_rank(self) -> None:
        """Test that low activity results in low rank."""
        data: dict[str, int | str] = {
            "login": "newbie",
            "name": "New User",
            "total_commits": 1,
            "total_prs": 0,
            "total_issues": 0,
            "total_stars": 0,
            "total_contributions": 1,
        }

        stats = build_user_stats(data)

        assert stats["rank"] in ("B", "C")

    def test_build_user_stats_handles_missing_name(self) -> None:
        """Test that missing name falls back to login."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 20,
        }

        stats = build_user_stats(data)

        assert stats["name"] == "testuser"


class TestBuildLanguageStats:
    """Tests for build_language_stats function."""

    def test_build_language_stats_basic(self) -> None:
        """Test building language stats from data."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": 50000, "color": "#3572A5"},
            {"name": "JavaScript", "size": 30000, "color": "#f1e05a"},
            {"name": "TypeScript", "size": 20000, "color": "#3178c6"},
        ]

        stats, total = build_language_stats(languages)

        assert total == 100000
        assert len(stats) == 3
        assert stats[0]["name"] == "Python"
        assert stats[0]["percentage"] == 50.0
        assert stats[1]["name"] == "JavaScript"
        assert stats[1]["percentage"] == 30.0
        assert stats[2]["name"] == "TypeScript"
        assert stats[2]["percentage"] == 20.0

    def test_build_language_stats_empty(self) -> None:
        """Test building language stats from empty data."""
        stats, total = build_language_stats([])

        assert total == 0
        assert stats == []

    def test_build_language_stats_filters_invalid(self) -> None:
        """Test that invalid entries are filtered out."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": 100, "color": "#3572A5"},
            {"name": "", "size": 50, "color": "#000"},  # Empty name
            {"name": "JS", "size": 0, "color": "#f1e05a"},  # Zero size
        ]

        stats, total = build_language_stats(languages)

        assert total == 100
        assert len(stats) == 1
        assert stats[0]["name"] == "Python"


class TestRenderStatsCard:
    """Tests for render_stats_card function."""

    def test_render_stats_card_basic(self) -> None:
        """Test rendering stats card."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="default",
            hide_border=False,
            show_icons=True,
            hide=(),
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Test User" in svg
        assert "Total Stars" in svg
        assert "Total Commits" in svg

    def test_render_stats_card_hides_stats(self) -> None:
        """Test that hide parameter hides stats."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="default",
            hide_border=False,
            show_icons=True,
            hide=("stars", "commits"),
            disable_animations=False,
        )

        assert "Total Stars" not in svg
        assert "Total Commits" not in svg
        assert "Total PRs" in svg

    def test_render_stats_card_respects_theme(self) -> None:
        """Test that theme colors are applied."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="dracula",
            hide_border=False,
            show_icons=True,
            hide=(),
            disable_animations=False,
        )

        assert "#282a36" in svg  # dracula bg color
        assert "#ff79c6" in svg  # dracula title color


class TestRenderLangsCard:
    """Tests for render_langs_card function."""

    def test_render_langs_card_default_layout(self) -> None:
        """Test rendering langs card with default layout."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
            {"name": "TypeScript", "size": 30000, "percentage": 30.0, "color": "#3178c6"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=100000,
            theme_name="default",
            hide_border=False,
            layout="default",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Python" in svg
        assert "TypeScript" in svg

    def test_render_langs_card_compact_layout(self) -> None:
        """Test rendering langs card with compact layout."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=50000,
            theme_name="dracula",
            hide_border=True,
            layout="compact",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "Python" in svg

    def test_render_langs_card_donut_layout(self) -> None:
        """Test rendering langs card with donut layout."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
            {"name": "JavaScript", "size": 50000, "percentage": 50.0, "color": "#f1e05a"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=100000,
            theme_name="default",
            hide_border=False,
            layout="donut",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert 'height="250"' in svg  # donut layout height

    def test_render_langs_card_pie_layout(self) -> None:
        """Test rendering langs card with pie layout."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 50000, "percentage": 50.0, "color": "#3572A5"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=50000,
            theme_name="default",
            hide_border=False,
            layout="pie",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert 'height="250"' in svg  # pie layout height

    def test_render_langs_card_zero_percentage_segment(self) -> None:
        """Test rendering langs card with zero percentage language."""
        languages: list[LanguageStats] = [
            {"name": "Python", "size": 100, "percentage": 100.0, "color": "#3572A5"},
            {"name": "Other", "size": 0, "percentage": 0.0, "color": "#858585"},
        ]

        svg = render_langs_card(
            username="testuser",
            languages=languages,
            total_size=100,
            theme_name="default",
            hide_border=False,
            layout="compact",
            langs_count=8,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "Python" in svg


class TestRenderStatsCardVariations:
    """Additional tests for render_stats_card edge cases."""

    def test_render_stats_card_without_icons(self) -> None:
        """Test rendering stats card without icons."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="default",
            hide_border=False,
            show_icons=False,
            hide=(),
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "Total Stars" in svg
        # Icons should not appear in stat lines when show_icons=False
        assert "⭐" not in svg

    def test_render_stats_card_hides_prs_and_issues(self) -> None:
        """Test hiding prs and issues stats."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "B+",
            "rank_percentile": 45.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="default",
            hide_border=True,
            show_icons=True,
            hide=("prs", "issues"),
            disable_animations=False,
        )

        assert "Total PRs" not in svg
        assert "Total Issues" not in svg
        assert "Total Stars" in svg

    def test_render_stats_card_hides_contribs(self) -> None:
        """Test hiding contribs stat."""
        stats: UserStats = {
            "username": "testuser",
            "name": "Test User",
            "total_commits": 100,
            "total_prs": 20,
            "total_issues": 10,
            "total_stars": 50,
            "total_contributions": 150,
            "rank": "A",
            "rank_percentile": 30.0,
        }

        svg = render_stats_card(
            stats=stats,
            theme_name="default",
            hide_border=False,
            show_icons=True,
            hide=("contribs",),
            disable_animations=False,
        )

        assert "Contributed to" not in svg
        assert "Total Stars" in svg


class TestCalculateRank:
    """Tests for _calculate_rank function.

    Rank calculation uses: score = commits*1 + prs*2 + issues*1 + stars*4
    Then: percentile = 100 - log10(score+1) * 15

    Thresholds:
    - S+: percentile <= 1 (score >= ~4 million)
    - S: percentile <= 12.5 (score >= ~680k)
    - A+: percentile <= 25 (score >= 100k)
    - A: percentile <= 37.5 (score >= ~15k)
    - B+: percentile <= 50 (score >= ~2k)
    - B: percentile <= 62.5 (score >= ~300)
    - C: everything else
    """

    def test_calculate_rank_s_plus(self) -> None:
        """Test S+ rank calculation."""
        # Need score >= 4 million for S+ (percentile <= 1)
        # Using stars=1,000,000 gives score = 4,000,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=1_000_000,
        )
        assert rank == "S+"
        assert percentile <= 1

    def test_calculate_rank_s(self) -> None:
        """Test S rank calculation."""
        # Need score ~680k-4M for S (percentile 1-12.5)
        # Using stars=200,000 gives score = 800,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=200_000,
        )
        assert rank == "S"
        assert 1 < percentile <= 12.5

    def test_calculate_rank_a_plus(self) -> None:
        """Test A+ rank calculation."""
        # Need score ~100k-680k for A+ (percentile 12.5-25)
        # Using stars=50,000 gives score = 200,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=50_000,
        )
        assert rank == "A+"
        assert 12.5 < percentile <= 25

    def test_calculate_rank_a(self) -> None:
        """Test A rank calculation."""
        # Need score ~15k-100k for A (percentile 25-37.5)
        # Using stars=10,000 gives score = 40,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=10_000,
        )
        assert rank == "A"
        assert 25 < percentile <= 37.5

    def test_calculate_rank_b_plus(self) -> None:
        """Test B+ rank calculation."""
        # Need score ~2k-15k for B+ (percentile 37.5-50)
        # Using stars=1,000 gives score = 4,000
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=1_000,
        )
        assert rank == "B+"
        assert 37.5 < percentile <= 50

    def test_calculate_rank_b(self) -> None:
        """Test B rank calculation."""
        # Need score ~300-2k for B (percentile 50-62.5)
        # Using stars=200 gives score = 800
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=200,
        )
        assert rank == "B"
        assert 50 < percentile <= 62.5

    def test_calculate_rank_c(self) -> None:
        """Test C rank calculation."""
        # Low activity gives C rank (percentile > 62.5)
        rank, percentile = _calculate_rank(
            commits=1,
            prs=0,
            issues=0,
            stars=0,
        )
        assert rank == "C"
        assert percentile > 62.5

    def test_calculate_rank_zero_activity(self) -> None:
        """Test rank calculation with zero activity."""
        rank, percentile = _calculate_rank(
            commits=0,
            prs=0,
            issues=0,
            stars=0,
        )
        assert rank == "C"
        assert percentile == 100.0


class TestFormatNumber:
    """Tests for _format_number function."""

    def test_format_number_millions(self) -> None:
        """Test formatting numbers in millions."""
        assert _format_number(1_000_000) == "1.0M"
        assert _format_number(2_500_000) == "2.5M"
        assert _format_number(10_000_000) == "10.0M"

    def test_format_number_thousands(self) -> None:
        """Test formatting numbers in thousands."""
        assert _format_number(1_000) == "1.0k"
        assert _format_number(2_500) == "2.5k"
        assert _format_number(999_999) == "1000.0k"

    def test_format_number_small(self) -> None:
        """Test formatting small numbers."""
        assert _format_number(0) == "0"
        assert _format_number(1) == "1"
        assert _format_number(999) == "999"


class TestEscapeXml:
    """Tests for _escape_xml function."""

    def test_escape_xml_ampersand(self) -> None:
        """Test escaping ampersand."""
        assert _escape_xml("A & B") == "A &amp; B"

    def test_escape_xml_less_than(self) -> None:
        """Test escaping less than."""
        assert _escape_xml("A < B") == "A &lt; B"

    def test_escape_xml_greater_than(self) -> None:
        """Test escaping greater than."""
        assert _escape_xml("A > B") == "A &gt; B"

    def test_escape_xml_quotes(self) -> None:
        """Test escaping quotes."""
        assert _escape_xml('A "B" C') == "A &quot;B&quot; C"
        assert _escape_xml("A 'B' C") == "A &apos;B&apos; C"

    def test_escape_xml_no_special_chars(self) -> None:
        """Test no escaping needed."""
        assert _escape_xml("Hello World") == "Hello World"


class TestBuildUserStatsTypeValidation:
    """Tests for build_user_stats type validation."""

    def test_build_user_stats_non_int_commits(self) -> None:
        """Test handling non-int commits."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": "not_an_int",
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["total_commits"] == 0

    def test_build_user_stats_non_int_prs(self) -> None:
        """Test handling non-int prs."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": 10,
            "total_prs": "not_an_int",
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["total_prs"] == 0

    def test_build_user_stats_non_int_issues(self) -> None:
        """Test handling non-int issues."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": "not_an_int",
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["total_issues"] == 0

    def test_build_user_stats_non_int_stars(self) -> None:
        """Test handling non-int stars."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": "not_an_int",
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["total_stars"] == 0

    def test_build_user_stats_non_str_name(self) -> None:
        """Test handling non-str name."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": 123,
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        # Should fall back to login
        assert stats["name"] == "testuser"

    def test_build_user_stats_non_str_login(self) -> None:
        """Test handling non-str login."""
        data: dict[str, int | str] = {
            "login": 123,
            "name": "Test User",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": 10,
        }

        stats = build_user_stats(data)
        assert stats["username"] == ""

    def test_build_user_stats_non_int_contributions(self) -> None:
        """Test handling non-int contributions."""
        data: dict[str, int | str] = {
            "login": "testuser",
            "name": "Test",
            "total_commits": 10,
            "total_prs": 5,
            "total_issues": 2,
            "total_stars": 1,
            "total_contributions": "not_an_int",
        }

        stats = build_user_stats(data)
        assert stats["total_contributions"] == 0


class TestBuildLanguageStatsTypeValidation:
    """Tests for build_language_stats type validation."""

    def test_build_language_stats_non_str_color(self) -> None:
        """Test handling non-str color."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": 100, "color": 123},
        ]

        stats, _total = build_language_stats(languages)
        assert len(stats) == 1
        assert stats[0]["color"] == "#858585"  # Default color

    def test_build_language_stats_negative_size(self) -> None:
        """Test handling negative size."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": -100, "color": "#3572A5"},
            {"name": "JavaScript", "size": 100, "color": "#f1e05a"},
        ]

        stats, _total = build_language_stats(languages)
        assert len(stats) == 1
        assert stats[0]["name"] == "JavaScript"

    def test_build_language_stats_non_int_size(self) -> None:
        """Test handling non-int size."""
        languages: list[dict[str, int | str]] = [
            {"name": "Python", "size": "not_an_int", "color": "#3572A5"},
            {"name": "JavaScript", "size": 100, "color": "#f1e05a"},
        ]

        stats, _total = build_language_stats(languages)
        assert len(stats) == 1
        assert stats[0]["name"] == "JavaScript"

    def test_build_language_stats_non_str_name(self) -> None:
        """Test handling non-str name."""
        languages: list[dict[str, int | str]] = [
            {"name": 123, "size": 100, "color": "#3572A5"},
            {"name": "JavaScript", "size": 100, "color": "#f1e05a"},
        ]

        stats, _total = build_language_stats(languages)
        assert len(stats) == 1
        assert stats[0]["name"] == "JavaScript"


class TestBuildCapabilitiesResponse:
    """Tests for build_capabilities_response function."""

    def test_build_capabilities_response_basic(self) -> None:
        """Test building capabilities response."""
        cap: Capability = {
            "name": "xgboost_tabular",
            "strength": "strong",
            "tags": ("ml", "tabular", "xgboost"),
            "description": "XGBoost gradient boosting",
        }

        response = build_capabilities_response(
            repo="owner/repo",
            capabilities=(cap,),
            ml_backends=("xgboost", "lightgbm"),
            frameworks=("fastapi",),
            data_formats=("csv", "parquet"),
            task_types=("binary_classification",),
        )

        assert response["repo"] == "owner/repo"
        assert len(response["capabilities"]) == 1
        assert response["capabilities"][0]["name"] == "xgboost_tabular"
        assert response["ml_backends"] == ("xgboost", "lightgbm")
        assert response["frameworks"] == ("fastapi",)
        assert response["data_formats"] == ("csv", "parquet")
        assert response["task_types"] == ("binary_classification",)

    def test_build_capabilities_response_empty(self) -> None:
        """Test building capabilities response with no capabilities."""
        response = build_capabilities_response(
            repo="owner/repo",
            capabilities=(),
            ml_backends=(),
            frameworks=(),
            data_formats=(),
            task_types=(),
        )

        assert response["repo"] == "owner/repo"
        assert response["capabilities"] == ()
        assert response["ml_backends"] == ()


class TestRenderCapabilitiesCard:
    """Tests for render_capabilities_card function."""

    def test_render_capabilities_card_basic(self) -> None:
        """Test rendering capabilities card."""
        cap: Capability = {
            "name": "xgboost_tabular",
            "strength": "strong",
            "tags": ("ml", "tabular"),
            "description": "XGBoost gradient boosting",
        }

        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (cap,),
            "ml_backends": ("xgboost",),
            "frameworks": ("fastapi",),
            "data_formats": ("csv",),
            "task_types": ("binary_classification",),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="default",
            hide_border=False,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Codebase Capabilities" in svg
        assert "Xgboost Tabular" in svg  # Title-cased
        assert "strong" in svg
        assert "xgboost" in svg
        assert "Binary Classification" in svg

    def test_render_capabilities_card_multiple_capabilities(self) -> None:
        """Test rendering capabilities card with multiple capabilities."""
        caps: tuple[Capability, ...] = (
            {
                "name": "xgboost_tabular",
                "strength": "strong",
                "tags": ("ml",),
                "description": "XGBoost",
            },
            {
                "name": "fastapi_rest",
                "strength": "moderate",
                "tags": ("web",),
                "description": "FastAPI REST",
            },
            {
                "name": "pytorch_cv",
                "strength": "basic",
                "tags": ("cv",),
                "description": "PyTorch CV",
            },
        )

        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": caps,
            "ml_backends": ("xgboost", "pytorch"),
            "frameworks": ("fastapi",),
            "data_formats": ("csv", "parquet"),
            "task_types": ("binary_classification", "image_classification"),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="dracula",
            hide_border=True,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "Xgboost Tabular" in svg
        assert "Fastapi Rest" in svg
        assert "Pytorch Cv" in svg
        assert "strong" in svg
        assert "moderate" in svg
        assert "basic" in svg
        # Check strength classes
        assert "strength-strong" in svg
        assert "strength-moderate" in svg
        assert "strength-basic" in svg

    def test_render_capabilities_card_no_capabilities(self) -> None:
        """Test rendering capabilities card with no capabilities."""
        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="default",
            hide_border=False,
            disable_animations=False,
        )

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "Codebase Capabilities" in svg

    def test_render_capabilities_card_with_theme(self) -> None:
        """Test that theme colors are applied."""
        cap: Capability = {
            "name": "test_cap",
            "strength": "strong",
            "tags": (),
            "description": "Test",
        }

        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (cap,),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="dracula",
            hide_border=False,
            disable_animations=False,
        )

        assert "#282a36" in svg  # dracula bg color
        assert "#ff79c6" in svg  # dracula title color

    def test_render_capabilities_card_many_task_types(self) -> None:
        """Test rendering card with more than 6 task types shows +N more."""
        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (
                "type1",
                "type2",
                "type3",
                "type4",
                "type5",
                "type6",
                "type7",
                "type8",
            ),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="default",
            hide_border=False,
            disable_animations=False,
        )

        assert "+2 more" in svg

    def test_render_capabilities_card_hide_border(self) -> None:
        """Test hiding border on capabilities card."""
        response: CapabilitiesResponse = {
            "repo": "owner/repo",
            "capabilities": (),
            "ml_backends": (),
            "frameworks": (),
            "data_formats": (),
            "task_types": (),
        }

        svg = render_capabilities_card(
            response=response,
            theme_name="default",
            hide_border=True,
            disable_animations=False,
        )

        assert 'stroke-opacity="0"' in svg
