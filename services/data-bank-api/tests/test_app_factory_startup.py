from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from platform_core.testing import make_fake_env

from data_bank_api.api.main import create_app


def test_app_factory_and_health_endpoints(tmp_path: Path) -> None:
    env = make_fake_env()
    env.set("API_UPLOAD_KEYS", "u1")
    env.set("REDIS_URL", "redis://ignored")
    # `create_app` builds the storage eagerly, so the data root must be a
    # place this process may actually write. It used to inherit the loader's
    # hardcoded `/data/files`: writable on Windows, where it resolves under
    # the current drive, and PermissionError on Linux, where it is a
    # directory at the filesystem root. Added 2025-12-05 and green ever
    # since, because until now it had only ever been run on Windows.
    env.set("DATA_ROOT", str(tmp_path / "files"))
    app = create_app()
    client: TestClient = TestClient(app)

    r1 = client.get("/healthz")
    assert r1.status_code == 200
    # Avoid JSON parsing here to satisfy strict typing policies in tests
    # while still validating the contract succinctly.
    assert '"status"' in r1.text
    assert '"ok"' in r1.text

    r2 = client.get("/readyz")
    assert r2.status_code in (200, 503)
    assert '"status"' in r2.text
