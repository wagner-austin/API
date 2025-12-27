"""Tests for platform_devpost.client module."""

from __future__ import annotations

from platform_devpost.client import DevpostClient
from platform_devpost.testing import FakeDevpostApi, hooks, make_fake_hackathon


class TestDevpostClient:
    """Tests for DevpostClient class."""

    def test_list_hackathons_returns_all(self) -> None:
        """Test list_hackathons returns all hackathons."""
        h1 = make_fake_hackathon(id=1, title="Hack 1")
        h2 = make_fake_hackathon(id=2, title="Hack 2")
        fake_api = FakeDevpostApi(hackathons=(h1, h2))
        hooks.devpost_api_factory = lambda: fake_api

        client = DevpostClient()
        result = client.list_hackathons()

        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

    def test_list_hackathons_with_search(self) -> None:
        """Test list_hackathons filters by search."""
        h1 = make_fake_hackathon(id=1, title="AI Challenge")
        h2 = make_fake_hackathon(id=2, title="Web Dev Hack")
        fake_api = FakeDevpostApi(hackathons=(h1, h2))
        hooks.devpost_api_factory = lambda: fake_api

        client = DevpostClient()
        result = client.list_hackathons(search="AI")

        assert len(result) == 1
        assert result[0].id == 1

    def test_list_hackathons_with_state(self) -> None:
        """Test list_hackathons filters by state."""
        h1 = make_fake_hackathon(id=1, open_state="open")
        h2 = make_fake_hackathon(id=2, open_state="ended")
        fake_api = FakeDevpostApi(hackathons=(h1, h2))
        hooks.devpost_api_factory = lambda: fake_api

        client = DevpostClient()
        result = client.list_hackathons(state="open")

        assert len(result) == 1
        assert result[0].id == 1

    def test_get_hackathon_found(self) -> None:
        """Test get_hackathon returns hackathon when found."""
        h1 = make_fake_hackathon(id=42, title="Target Hack")
        h2 = make_fake_hackathon(id=99, title="Other Hack")
        fake_api = FakeDevpostApi(hackathons=(h1, h2))
        hooks.devpost_api_factory = lambda: fake_api

        client = DevpostClient()
        result = client.get_hackathon(42)

        # Verify result is the expected hackathon, not None
        assert result == h1
        assert result.id == 42
        assert result.title == "Target Hack"

    def test_get_hackathon_not_found(self) -> None:
        """Test get_hackathon returns None when not found."""
        h1 = make_fake_hackathon(id=1)
        fake_api = FakeDevpostApi(hackathons=(h1,))
        hooks.devpost_api_factory = lambda: fake_api

        client = DevpostClient()
        result = client.get_hackathon(999)

        assert result is None
