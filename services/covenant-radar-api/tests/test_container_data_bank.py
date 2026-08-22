"""Tests for service container dependency injection."""

from __future__ import annotations

from pathlib import Path

from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from covenant_radar_api.core import ServiceContainer
from covenant_radar_api.core.config import Settings


def test_container_get_model_file_id_xgboost(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _get_model_file_id returns correct file ID for XGBoost backend."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_model_file_id="active_xgb.ubj",
    )
    assert container._get_model_file_id() == "active_xgb.ubj"
    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})


def test_container_get_model_file_id_mlp(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _get_model_file_id returns correct file ID for MLP backend."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.pt"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="mlp",
        data_bank_model_file_id="active_mlp.pt",
    )
    assert container._get_model_file_id() == "active_mlp.pt"
    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})


def test_container_get_model_file_id_lstm(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _get_model_file_id returns correct file ID for LSTM backend."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.pt"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="lstm",
        data_bank_model_file_id="active_lstm.pt",
    )
    assert container._get_model_file_id() == "active_lstm.pt"
    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})


def test_container_get_model_file_id_lightgbm(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _get_model_file_id returns correct file ID for LightGBM backend."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.txt"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="lightgbm",
        data_bank_model_file_id="active_lgbm.txt",
    )
    assert container._get_model_file_id() == "active_lgbm.txt"
    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})


def test_container_download_model_from_data_bank_not_configured(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _download_model_from_data_bank returns False when not configured."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="",  # Not configured
        data_bank_key="",
    )
    result = container._download_model_from_data_bank(tmp_path / "model.ubj")
    assert result is False
    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})


def test_container_download_model_from_data_bank_success(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _download_model_from_data_bank successfully downloads model."""
    from platform_core.data_bank_client import HeadInfo

    from covenant_radar_api.core import _test_hooks

    # Track download calls
    download_calls: list[tuple[str, Path]] = []

    class FakeDataBankDownloader:
        """Fake downloader that writes a file."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            download_calls.append((file_id, dest))
            dest.write_bytes(b"fake model content")
            return HeadInfo(size=18, etag="abc123", content_type="application/octet-stream")

    def fake_factory(base_url: str, api_key: str) -> FakeDataBankDownloader:
        return FakeDataBankDownloader()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    dest_path = tmp_path / "models" / "active_xgb.ubj"
    result = container._download_model_from_data_bank(dest_path)

    assert result is True
    assert len(download_calls) == 1
    assert download_calls[0][0] == "active_xgb.ubj"
    assert download_calls[0][1] == dest_path
    assert dest_path.exists()
    assert dest_path.read_bytes() == b"fake model content"

    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_download_model_from_data_bank_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _download_model_from_data_bank returns False when file not found."""
    from platform_core.data_bank_client import HeadInfo, NotFoundError

    from covenant_radar_api.core import _test_hooks

    class FakeDataBankDownloaderNotFound:
        """Fake downloader that raises NotFoundError."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            raise NotFoundError(f"File {file_id} not found")

    def fake_factory(base_url: str, api_key: str) -> FakeDataBankDownloaderNotFound:
        return FakeDataBankDownloaderNotFound()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    dest_path = tmp_path / "models" / "active_xgb.ubj"
    result = container._download_model_from_data_bank(dest_path)

    assert result is False
    assert not dest_path.exists()

    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_download_model_from_data_bank_client_error(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test _download_model_from_data_bank returns False on client errors."""
    from platform_core.data_bank_client import DataBankClientError, HeadInfo

    from covenant_radar_api.core import _test_hooks

    class FakeDataBankDownloaderError:
        """Fake downloader that raises DataBankClientError."""

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            raise DataBankClientError("transport error: Connection refused")

    def fake_factory(base_url: str, api_key: str) -> FakeDataBankDownloaderError:
        return FakeDataBankDownloaderError()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    dest_path = tmp_path / "models" / "active_xgb.ubj"
    result = container._download_model_from_data_bank(dest_path)

    assert result is False
    assert not dest_path.exists()

    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})
    _test_hooks.data_bank_client_factory = orig_factory


def test_default_data_bank_client_factory_creates_client() -> None:
    """Test _default_data_bank_client_factory creates a DataBankClient."""
    from covenant_radar_api.core._test_hooks import _default_data_bank_client_factory

    client = _default_data_bank_client_factory(
        "https://data-bank.example.com",
        "test-api-key",
    )
    assert client.__class__.__name__ == "DataBankClient"
