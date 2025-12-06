from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from turkic_api.api.main import create_app


def test_app_factory_and_healthz_endpoint() -> None:
    app = create_app()
    client: TestClient = TestClient(app)

    r = client.get("/healthz")
    # The healthz (liveness) endpoint always returns 200 with status ok
    assert r.status_code == 200
    body: dict[str, str] = r.json()
    assert body == {"status": "ok"}


def test_readyz_with_default_redis_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test /readyz uses default redis provider when none is configured.

    This exercises the default redis path in main.py line 120.
    """
    from pathlib import Path

    def _exists(_: Path) -> bool:
        return True

    monkeypatch.setattr("pathlib.Path.exists", _exists, raising=False)

    # Create app without redis_provider to use the default path
    app = create_app()
    client: TestClient = TestClient(app)

    # The default redis will fail to connect, returning degraded status
    r = client.get("/readyz")
    body: dict[str, str | None] = r.json()
    # Either ready (if redis is available) or degraded (if not)
    assert body["status"] in ("ready", "degraded")
