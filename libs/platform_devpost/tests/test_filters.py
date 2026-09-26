"""Tests for platform_devpost.filters module."""

from __future__ import annotations

from platform_devpost.filters import filter_hackathons
from platform_devpost.testing import make_fake_hackathon, make_fake_theme, make_interest_filter
from platform_devpost.types import DisplayedLocation, Hackathon, HackathonState, Theme


def _make_hackathon_with_themes(
    hackathon_id: int,
    themes: tuple[Theme, ...],
    state: HackathonState = HackathonState.OPEN,
    featured: bool = False,
) -> Hackathon:
    """Create a hackathon with specific themes for testing."""
    return Hackathon(
        id=hackathon_id,
        title=f"Hackathon {hackathon_id}",
        url=f"https://example.com/{hackathon_id}",
        thumbnail_url="https://example.com/img.jpg",
        organization_name="Org",
        displayed_location=DisplayedLocation(icon="globe", location="Online"),
        open_state=state,
        time_left_to_submission="1 day",
        submission_period_dates="Jan 1-2",
        themes=themes,
        prize_amount="$0",
        registrations_count=0,
        featured=featured,
        winners_announced=False,
        invite_only=False,
    )


class TestFilterHackathons:
    """Tests for filter_hackathons function."""

    def test_empty_filter_returns_all(self) -> None:
        """Test empty filter returns all hackathons."""
        h1 = make_fake_hackathon(id=1)
        h2 = make_fake_hackathon(id=2)
        interests = make_interest_filter()

        result = filter_hackathons((h1, h2), interests)
        assert len(result) == 2

    def test_include_themes_filters(self) -> None:
        """Test include_themes filters hackathons."""
        ai_theme = make_fake_theme(id=1, name="AI/ML")
        web_theme = make_fake_theme(id=2, name="Web Development")

        h1 = _make_hackathon_with_themes(1, (ai_theme,))
        h2 = _make_hackathon_with_themes(2, (web_theme,))
        h3 = _make_hackathon_with_themes(3, (ai_theme, web_theme))

        interests = make_interest_filter(include_themes=("AI",))
        result = filter_hackathons((h1, h2, h3), interests)

        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 3

    def test_exclude_themes_filters(self) -> None:
        """Test exclude_themes removes matching hackathons."""
        finance_theme = make_fake_theme(id=1, name="Finance")
        ai_theme = make_fake_theme(id=2, name="AI/ML")

        h1 = _make_hackathon_with_themes(1, (finance_theme,))
        h2 = _make_hackathon_with_themes(2, (ai_theme,))

        interests = make_interest_filter(exclude_themes=("Finance",))
        result = filter_hackathons((h1, h2), interests)

        assert len(result) == 1
        assert result[0].id == 2

    def test_state_filter(self) -> None:
        """Test states filter hackathons by state."""
        h1 = make_fake_hackathon(id=1, open_state=HackathonState.OPEN)
        h2 = make_fake_hackathon(id=2, open_state=HackathonState.ENDED)
        h3 = make_fake_hackathon(id=3, open_state=HackathonState.UPCOMING)

        interests = make_interest_filter(states=(HackathonState.OPEN, HackathonState.UPCOMING))
        result = filter_hackathons((h1, h2, h3), interests)

        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 3

    def test_featured_only_filter(self) -> None:
        """Test featured_only filters to featured hackathons."""
        h1 = make_fake_hackathon(id=1, featured=True)
        h2 = make_fake_hackathon(id=2, featured=False)
        h3 = make_fake_hackathon(id=3, featured=True)

        interests = make_interest_filter(featured_only=True)
        result = filter_hackathons((h1, h2, h3), interests)

        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 3

    def test_combined_filters(self) -> None:
        """Test multiple filters applied together."""
        ai_theme = make_fake_theme(id=1, name="AI")

        h1 = _make_hackathon_with_themes(1, (ai_theme,), state=HackathonState.OPEN, featured=True)
        h2 = _make_hackathon_with_themes(2, (ai_theme,), state=HackathonState.ENDED, featured=True)
        h3 = _make_hackathon_with_themes(3, (), state=HackathonState.OPEN, featured=True)

        interests = make_interest_filter(
            include_themes=("AI",),
            states=(HackathonState.OPEN,),
            featured_only=True,
        )
        result = filter_hackathons((h1, h2, h3), interests)

        assert len(result) == 1
        assert result[0].id == 1

    def test_filter_empty_hackathons(self) -> None:
        """Test filtering empty tuple returns empty."""
        interests = make_interest_filter(include_themes=("AI",))
        result = filter_hackathons((), interests)
        assert len(result) == 0

    def test_no_matches_returns_empty(self) -> None:
        """Test filter with no matches returns empty."""
        h1 = make_fake_hackathon(id=1, open_state=HackathonState.ENDED)
        interests = make_interest_filter(states=(HackathonState.OPEN,))

        result = filter_hackathons((h1,), interests)
        assert len(result) == 0

    def test_theme_matching_case_insensitive(self) -> None:
        """Test theme matching is case insensitive."""
        ai_theme = make_fake_theme(id=1, name="AI/ML")
        h1 = _make_hackathon_with_themes(1, (ai_theme,))

        interests = make_interest_filter(include_themes=("ai",))
        result = filter_hackathons((h1,), interests)

        assert len(result) == 1

    def test_theme_matching_partial(self) -> None:
        """Test theme matching works with partial match."""
        ml_theme = make_fake_theme(id=1, name="Machine Learning")
        h1 = _make_hackathon_with_themes(1, (ml_theme,))

        interests = make_interest_filter(include_themes=("machine",))
        result = filter_hackathons((h1,), interests)

        assert len(result) == 1

    def test_empty_exclude_themes_allows_all(self) -> None:
        """Test empty exclude_themes allows all hackathons."""
        h1 = make_fake_hackathon(id=1)
        interests = make_interest_filter(exclude_themes=())

        result = filter_hackathons((h1,), interests)
        assert len(result) == 1
