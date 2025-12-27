"""Tests for platform_devpost.testing module."""

from __future__ import annotations

from platform_devpost.testing import (
    FakeDevpostApi,
    FakeDevpostClient,
    hooks,
    make_fake_capability,
    make_fake_displayed_location,
    make_fake_hackathon,
    make_fake_profile,
    make_fake_theme,
    make_interest_filter,
    reset_hooks,
)


class TestHooksContainer:
    """Tests for HooksContainer class."""

    def test_hooks_devpost_api_factory_callable(self) -> None:
        """Test hooks.devpost_api_factory is callable and returns API."""
        api = hooks.devpost_api_factory()
        # API should have fetch_hackathons method
        assert callable(api.fetch_hackathons)

    def test_hooks_devpost_client_callable(self) -> None:
        """Test hooks.devpost_client is callable and returns client."""
        client = hooks.devpost_client()
        # Client should have list_hackathons method
        assert callable(client.list_hackathons)

    def test_hooks_profile_scanner_callable(self) -> None:
        """Test hooks.profile_scanner is callable."""
        # Verify profile_scanner is a callable
        assert callable(hooks.profile_scanner)

    def test_hooks_devpost_client_returns_working_client(self) -> None:
        """Test hooks.devpost_client returns client with list_hackathons."""
        # Set up fake API
        fake_api = FakeDevpostApi(hackathons=(make_fake_hackathon(id=1),))
        hooks.devpost_api_factory = lambda: fake_api

        client = hooks.devpost_client()
        result = client.list_hackathons()
        assert len(result) == 1
        assert result[0].id == 1


class TestResetHooks:
    """Tests for reset_hooks function."""

    def test_reset_hooks_restores_production(self) -> None:
        """Test reset_hooks restores production implementations."""
        original_client_factory = hooks.devpost_client

        # Modify by calling reset (which sets to production)
        reset_hooks()

        # Verify it's callable and returns a client
        client = hooks.devpost_client()
        # Verify list_hackathons is callable
        assert callable(client.list_hackathons)

        # Call reset again to ensure consistency
        reset_hooks()

        # Should still work
        assert hooks.devpost_client == original_client_factory


class TestFakeDevpostApi:
    """Tests for FakeDevpostApi class."""

    def test_default_constructor(self) -> None:
        """Test FakeDevpostApi with default empty hackathons."""
        api = FakeDevpostApi()
        response = api.fetch_hackathons()
        assert len(response.hackathons) == 0

    def test_with_hackathons(self) -> None:
        """Test FakeDevpostApi with configured hackathons."""
        hackathon = make_fake_hackathon(id=1, title="Test Hack")
        api = FakeDevpostApi(hackathons=(hackathon,))
        response = api.fetch_hackathons()
        assert len(response.hackathons) == 1
        assert response.hackathons[0].id == 1

    def test_fetch_with_page(self) -> None:
        """Test fetch_hackathons with page parameter."""
        api = FakeDevpostApi()
        response = api.fetch_hackathons(page=2)
        assert response.meta.per_page == 10

    def test_fetch_with_search(self) -> None:
        """Test fetch_hackathons with search parameter."""
        h1 = make_fake_hackathon(id=1, title="AI Challenge")
        h2 = make_fake_hackathon(id=2, title="Web Dev Hack")
        api = FakeDevpostApi(hackathons=(h1, h2))

        response = api.fetch_hackathons(search="AI")
        assert len(response.hackathons) == 1
        assert response.hackathons[0].id == 1

    def test_fetch_calls_recorded(self) -> None:
        """Test that fetch calls are recorded."""
        api = FakeDevpostApi()
        api.fetch_hackathons(page=1, search="test")
        api.fetch_hackathons(page=2)

        assert len(api._fetch_calls) == 2
        assert api._fetch_calls[0] == {"page": 1, "search": "test"}
        assert api._fetch_calls[1] == {"page": 2, "search": None}


class TestFakeDevpostClient:
    """Tests for FakeDevpostClient class."""

    def test_default_constructor(self) -> None:
        """Test FakeDevpostClient with default empty hackathons."""
        client = FakeDevpostClient()
        hackathons = client.list_hackathons()
        assert len(hackathons) == 0

    def test_with_hackathons(self) -> None:
        """Test FakeDevpostClient with configured hackathons."""
        hackathon = make_fake_hackathon(id=1, title="Test")
        client = FakeDevpostClient(hackathons=(hackathon,))
        result = client.list_hackathons()
        assert len(result) == 1

    def test_list_with_search(self) -> None:
        """Test list_hackathons with search filter."""
        h1 = make_fake_hackathon(id=1, title="AI Hack")
        h2 = make_fake_hackathon(id=2, title="Web Hack")
        client = FakeDevpostClient(hackathons=(h1, h2))

        result = client.list_hackathons(search="AI")
        assert len(result) == 1
        assert result[0].id == 1

    def test_list_with_state(self) -> None:
        """Test list_hackathons with state filter."""
        h1 = make_fake_hackathon(id=1, open_state="open")
        h2 = make_fake_hackathon(id=2, open_state="ended")
        client = FakeDevpostClient(hackathons=(h1, h2))

        result = client.list_hackathons(state="open")
        assert len(result) == 1
        assert result[0].id == 1

    def test_get_hackathon_found(self) -> None:
        """Test get_hackathon returns hackathon when found."""
        hackathon = make_fake_hackathon(id=42)
        client = FakeDevpostClient(hackathons=(hackathon,))

        result = client.get_hackathon(42)
        assert result == hackathon
        assert result.id == 42

    def test_get_hackathon_found_after_iteration(self) -> None:
        """Test get_hackathon finds hackathon after iterating."""
        h1 = make_fake_hackathon(id=1)
        h2 = make_fake_hackathon(id=2)
        h3 = make_fake_hackathon(id=3)
        client = FakeDevpostClient(hackathons=(h1, h2, h3))

        # Find the last hackathon to ensure we iterate through all
        result = client.get_hackathon(3)
        assert result == h3
        assert result.id == 3

    def test_get_hackathon_not_found(self) -> None:
        """Test get_hackathon returns None when not found."""
        client = FakeDevpostClient()
        result = client.get_hackathon(999)
        assert result is None

    def test_get_hackathon_not_found_with_hackathons(self) -> None:
        """Test get_hackathon returns None when ID not in list."""
        h1 = make_fake_hackathon(id=1)
        h2 = make_fake_hackathon(id=2)
        client = FakeDevpostClient(hackathons=(h1, h2))

        result = client.get_hackathon(999)
        assert result is None

    def test_list_calls_recorded(self) -> None:
        """Test that list calls are recorded."""
        client = FakeDevpostClient()
        client.list_hackathons(search="test", state="open")
        client.list_hackathons()

        assert len(client._list_calls) == 2
        assert client._list_calls[0] == {"search": "test", "state": "open"}
        assert client._list_calls[1] == {"search": None, "state": None}


class TestMakeFakeTheme:
    """Tests for make_fake_theme factory."""

    def test_default_values(self) -> None:
        """Test make_fake_theme with default values."""
        theme = make_fake_theme()
        assert theme.id == 1
        assert theme.name == "Test Theme"

    def test_custom_values(self) -> None:
        """Test make_fake_theme with custom values."""
        theme = make_fake_theme(id=99, name="AI/ML")
        assert theme.id == 99
        assert theme.name == "AI/ML"


class TestMakeFakeDisplayedLocation:
    """Tests for make_fake_displayed_location factory."""

    def test_default_values(self) -> None:
        """Test make_fake_displayed_location with default values."""
        loc = make_fake_displayed_location()
        assert loc.icon == "globe"
        assert loc.location == "Online"

    def test_custom_values(self) -> None:
        """Test make_fake_displayed_location with custom values."""
        loc = make_fake_displayed_location(icon="map", location="NYC")
        assert loc.icon == "map"
        assert loc.location == "NYC"


class TestMakeFakeHackathon:
    """Tests for make_fake_hackathon factory."""

    def test_default_values(self) -> None:
        """Test make_fake_hackathon with default values."""
        h = make_fake_hackathon()
        assert h.id == 1
        assert h.title == "Test Hackathon"
        assert h.open_state == "open"
        assert h.featured is False
        # Verify default location is created
        assert h.displayed_location.icon == "globe"
        assert h.displayed_location.location == "Online"

    def test_custom_values(self) -> None:
        """Test make_fake_hackathon with custom values."""
        h = make_fake_hackathon(
            id=42,
            title="AI Challenge",
            open_state="upcoming",
            featured=True,
        )
        assert h.id == 42
        assert h.title == "AI Challenge"
        assert h.open_state == "upcoming"
        assert h.featured is True

    def test_with_explicit_location(self) -> None:
        """Test make_fake_hackathon with explicit displayed_location."""
        custom_loc = make_fake_displayed_location(icon="pin", location="NYC")
        h = make_fake_hackathon(displayed_location=custom_loc)
        assert h.displayed_location.icon == "pin"
        assert h.displayed_location.location == "NYC"


class TestMakeFakeCapability:
    """Tests for make_fake_capability factory."""

    def test_default_values(self) -> None:
        """Test make_fake_capability with default values."""
        cap = make_fake_capability()
        assert cap.name == "test_capability"
        assert cap.strength == "moderate"
        assert cap.tags == ("test",)

    def test_custom_values(self) -> None:
        """Test make_fake_capability with custom values."""
        cap = make_fake_capability(
            name="web_dev",
            strength="strong",
            tags=("web", "frontend"),
        )
        assert cap.name == "web_dev"
        assert cap.strength == "strong"
        assert cap.tags == ("web", "frontend")


class TestMakeFakeProfile:
    """Tests for make_fake_profile factory."""

    def test_default_values(self) -> None:
        """Test make_fake_profile with default values."""
        profile = make_fake_profile()
        assert profile.capabilities == ()
        assert profile.technologies == ("python",)
        assert profile.frameworks == ("flask",)

    def test_custom_values(self) -> None:
        """Test make_fake_profile with custom values."""
        cap = make_fake_capability(name="ml")
        profile = make_fake_profile(
            capabilities=(cap,),
            technologies=("python", "javascript"),
            frameworks=("flask", "react"),
        )
        assert len(profile.capabilities) == 1
        assert profile.technologies == ("python", "javascript")
        assert profile.frameworks == ("flask", "react")


class TestMakeInterestFilter:
    """Tests for make_interest_filter factory."""

    def test_default_values(self) -> None:
        """Test make_interest_filter with default values."""
        f = make_interest_filter()
        assert f.include_themes == ()
        assert f.exclude_themes == ()
        assert f.states is None
        assert f.featured_only is False

    def test_custom_values(self) -> None:
        """Test make_interest_filter with custom values."""
        f = make_interest_filter(
            include_themes=("AI",),
            exclude_themes=("Finance",),
            states=("open",),
            featured_only=True,
        )
        assert f.include_themes == ("AI",)
        assert f.exclude_themes == ("Finance",)
        assert f.states == ("open",)
        assert f.featured_only is True
