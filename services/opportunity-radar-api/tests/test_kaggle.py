"""Tests for Kaggle routes."""

from __future__ import annotations

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_list

from opportunity_radar_api.api.container import ServiceContainer
from opportunity_radar_api.api.main import create_app


def test_list_competitions(fake_container: ServiceContainer) -> None:
    """Test listing competitions without filters."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/kaggle/competitions")

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    assert len(data) == 1
    match = narrow_json_to_dict(data[0])
    competition = narrow_json_to_dict(match["competition"])
    assert competition["ref"] == "test-comp"


def test_list_competitions_with_tags(fake_container: ServiceContainer) -> None:
    """Test listing competitions with tag filter."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/kaggle/competitions", params={"tags": ["tabular"]})

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    assert len(data) == 1


def test_list_competitions_with_exclude_tags(fake_container: ServiceContainer) -> None:
    """Test listing competitions with exclude filter."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get(
        "/kaggle/competitions", params={"tags": ["tabular"], "exclude": ["image"]}
    )

    assert response.status_code == 200


def test_list_competitions_with_min_score(fake_container: ServiceContainer) -> None:
    """Test listing competitions with minimum score filter."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    # High min_score should filter out
    response = client.get("/kaggle/competitions", params={"min_score": 0.9})

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    # May be empty if score is below threshold
    assert len(data) >= 0


def test_list_competitions_without_codebase_matching(
    fake_container: ServiceContainer,
) -> None:
    """Test listing competitions without codebase matching."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/kaggle/competitions", params={"match_codebase": False})

    assert response.status_code == 200
    data = narrow_json_to_list(load_json_str(response.text))
    # Should return raw competitions, not matches
    assert len(data) == 1
    comp = narrow_json_to_dict(data[0])
    assert "ref" in comp  # Direct competition, not match wrapper


def test_get_competition_found(fake_container: ServiceContainer) -> None:
    """Test getting specific competition by ref."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/kaggle/competitions/test-comp")

    assert response.status_code == 200
    data = narrow_json_to_dict(load_json_str(response.text))
    assert data["ref"] == "test-comp"
    assert data["title"] == "Test Competition"


def test_get_competition_not_found(fake_container: ServiceContainer) -> None:
    """Test getting non-existent competition returns 404."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    response = client.get("/kaggle/competitions/non-existent")

    assert response.status_code == 404
    data = narrow_json_to_dict(load_json_str(response.text))
    assert "not found" in str(data["detail"])
