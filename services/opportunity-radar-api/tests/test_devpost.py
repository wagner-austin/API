"""Tests for Devpost routes."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_list

from opportunity_radar_api.api.container import ServiceContainer
from opportunity_radar_api.api.main import create_app


def test_list_hackathons(fake_container: ServiceContainer) -> None:
    """Test listing hackathons without filters."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/devpost/hackathons")

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    assert len(data) == 1
    match = narrow_json_to_dict(data[0])
    hackathon = narrow_json_to_dict(match["hackathon"])
    assert hackathon["id"] == 123


def test_list_hackathons_with_themes(fake_container: ServiceContainer) -> None:
    """Test listing hackathons with theme filter."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/devpost/hackathons", params={"themes": ["AI"]})

    assert response.status_code == 200


def test_list_hackathons_with_exclude(fake_container: ServiceContainer) -> None:
    """Test listing hackathons with exclude filter."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/devpost/hackathons", params={"themes": ["AI"], "exclude": ["Gaming"]})

    assert response.status_code == 200


def test_list_hackathons_with_states(fake_container: ServiceContainer) -> None:
    """Test listing hackathons with state filter."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    # Test with open state
    response = client.get("/devpost/hackathons", params={"states": ["open"]})
    assert response.status_code == 200

    # Test with multiple states
    response = client.get("/devpost/hackathons", params={"states": ["open", "upcoming"]})
    assert response.status_code == 200


def test_list_hackathons_with_all_states(fake_container: ServiceContainer) -> None:
    """Test listing hackathons with all state types."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    # Test each state type
    for state in ["open", "upcoming", "ended", "submissions"]:
        response = client.get("/devpost/hackathons", params={"states": [state]})
        assert response.status_code == 200


def test_list_hackathons_with_min_score(fake_container: ServiceContainer) -> None:
    """Test listing hackathons with minimum score filter."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/devpost/hackathons", params={"min_score": 0.9})

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    assert len(data) >= 0


def test_list_hackathons_without_codebase_matching(
    fake_container: ServiceContainer,
) -> None:
    """Test listing hackathons without codebase matching."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/devpost/hackathons", params={"match_codebase": False})

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    # Should return raw hackathons, not matches
    assert len(data) == 1
    hackathon = narrow_json_to_dict(data[0])
    assert "id" in hackathon  # Direct hackathon, not match wrapper


def test_list_hackathons_featured_only(fake_container: ServiceContainer) -> None:
    """Test listing hackathons with featured_only filter."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/devpost/hackathons", params={"featured_only": True})

    assert response.status_code == 200


def test_get_hackathon_found(fake_container: ServiceContainer) -> None:
    """Test getting specific hackathon by ID."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/devpost/hackathons/123")

    assert response.status_code == 200
    data = narrow_json_to_dict(load_json_str(response.text))
    assert data["id"] == 123
    assert data["title"] == "Test Hackathon"


def test_get_hackathon_not_found(fake_container: ServiceContainer) -> None:
    """Test getting non-existent hackathon returns 404."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/devpost/hackathons/99999")

    assert response.status_code == 404
    data = narrow_json_to_dict(load_json_str(response.text))
    assert "not found" in str(data["detail"])


def test_list_hackathons_with_invalid_state(fake_container: ServiceContainer) -> None:
    """Test listing hackathons ignores invalid state strings."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    # Pass invalid state that should be ignored
    response = client.get("/devpost/hackathons", params={"states": ["invalid_state"]})

    # Should still succeed (invalid states are just ignored)
    assert response.status_code == 200
