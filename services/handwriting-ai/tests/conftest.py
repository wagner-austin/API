from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis


def _write_mnist_raw(root: Path, n: int = 8) -> None:
    """Create minimal MNIST raw data files for testing."""
    raw = (root / "MNIST" / "raw").resolve()
    raw.mkdir(parents=True, exist_ok=True)

    img_path = raw / "train-images-idx3-ubyte.gz"
    rows = 28
    cols = 28
    total = int(n) * rows * cols
    header = (
        (2051).to_bytes(4, "big")
        + int(n).to_bytes(4, "big")
        + rows.to_bytes(4, "big")
        + cols.to_bytes(4, "big")
    )
    payload = bytes([0]) * total
    with gzip.open(img_path, "wb") as f:
        f.write(header)
        f.write(payload)

    lbl_path = raw / "train-labels-idx1-ubyte.gz"
    header_l = (2049).to_bytes(4, "big") + int(n).to_bytes(4, "big")
    labels = bytes([i % 10 for i in range(int(n))])
    with gzip.open(lbl_path, "wb") as f:
        f.write(header_l)
        f.write(labels)


class MnistRawWriter:
    """Callable class for creating MNIST raw test data."""

    def __call__(self, root: Path, n: int = 8) -> None:
        _write_mnist_raw(root, n)


def _make_mnist_raw_writer() -> MnistRawWriter:
    """Factory function for MnistRawWriter fixture."""
    return MnistRawWriter()


write_mnist_raw = pytest.fixture(_make_mnist_raw_writer)


@pytest.fixture(autouse=True)
def _readyz_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a typed Redis stub and REDIS_URL for /readyz in tests.

    This keeps readiness checks deterministic without requiring a real Redis.
    Individual tests may override these as needed.
    """
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    import handwriting_ai.api.routes.health as health_mod

    # Return a fresh fake per call, pre-seeded with one worker so /readyz passes by default
    def _rf(url: str) -> RedisStrProto:
        r = FakeRedis()
        r.sadd("rq:workers", "w1")
        return r

    monkeypatch.setattr(health_mod, "redis_for_kv", _rf)


@pytest.fixture(autouse=True)
def _mock_data_bank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide data bank API credentials and stub ArtifactStore for tests."""
    monkeypatch.setenv("APP__DATA_BANK_API_URL", "http://test-db")
    monkeypatch.setenv("APP__DATA_BANK_API_KEY", "test-key")

    class _FakeStore:
        def __init__(self, base_url: str, api_key: str) -> None:
            pass

        def upload_artifact(
            self,
            dir_path: Path,
            *,
            artifact_name: str,
            request_id: str,
        ) -> dict[str, str | int | None]:
            return {
                "file_id": "fake-file-id",
                "size": 1,
                "sha256": "x",
                "content_type": "application/gzip",
                "created_at": None,
            }

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)


@pytest.fixture()
def digits_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    """Provide a typed Redis stub for digits jobs and capture published events."""
    stub = FakeRedis()
    import handwriting_ai.jobs.digits as digits_mod

    def _redis_for_kv(_: str) -> FakeRedis:
        return stub

    monkeypatch.setattr(digits_mod, "redis_for_kv", _redis_for_kv, raising=True)
    return stub
