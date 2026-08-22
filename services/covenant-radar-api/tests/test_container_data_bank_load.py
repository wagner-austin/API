"""Tests for service container dependency injection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.testing import make_train_config
from covenant_ml.trainer_fit import (
    save_model,
    train_model,
)
from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from numpy.typing import NDArray
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from covenant_radar_api.core import ServiceContainer
from covenant_radar_api.core.config import Settings


def test_container_load_model_now_tries_data_bank_when_configured(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now downloads from data-bank when local file missing."""
    from platform_core.data_bank_client import HeadInfo

    from covenant_radar_api.core import _test_hooks

    x_train: NDArray[np.float64] = np.zeros((4, 8), dtype=np.float64)
    x_train[0, 0] = 2.0
    x_train[1, 0] = 3.0
    x_train[2, 0] = 5.0
    x_train[3, 0] = 6.0
    y_train: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    y_train[2] = 1
    y_train[3] = 1
    config = make_train_config(
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    xgb_model = train_model(x_train, y_train, config)
    staging_path = tmp_path / "staging" / "model.ubj"
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(xgb_model, str(staging_path))
    model_bytes = staging_path.read_bytes()

    class FakeDownloader:
        """Fake downloader that writes model bytes to dest path."""

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
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(model_bytes)
            return HeadInfo(
                size=len(model_bytes),
                etag="fake-etag",
                content_type="application/octet-stream",
            )

    def fake_factory(base_url: str, api_key: str) -> FakeDownloader:
        return FakeDownloader()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0},
        region_encoder={"North America": 0},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    result = container.load_model_now()

    assert result is True
    assert container.get_model_info()["is_loaded"] is True

    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_load_model_now_returns_false_when_data_bank_fails(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now returns False when data-bank download fails."""
    from platform_core.data_bank_client import DataBankClientError, HeadInfo

    from covenant_radar_api.core import _test_hooks

    class FakeDownloaderFail:
        """Fake downloader that raises error."""

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
            raise DataBankClientError("connection refused")

    def fake_factory(base_url: str, api_key: str) -> FakeDownloaderFail:
        return FakeDownloaderFail()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    result = container.load_model_now()

    assert result is False
    assert container.get_model_info()["is_loaded"] is False

    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_load_model_now_skips_when_no_file_id(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now skips data-bank download when file_id is empty."""
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="",
    )

    result = container.load_model_now()

    assert result is False
    assert container.get_model_info()["is_loaded"] is False

    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})


def test_container_get_model_tries_data_bank_when_configured(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model downloads from data-bank when local file missing."""
    from platform_core.data_bank_client import HeadInfo

    from covenant_radar_api.core import _test_hooks

    x_train: NDArray[np.float64] = np.zeros((4, 8), dtype=np.float64)
    x_train[0, 0] = 2.0
    x_train[1, 0] = 3.0
    x_train[2, 0] = 5.0
    x_train[3, 0] = 6.0
    y_train: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
    y_train[2] = 1
    y_train[3] = 1
    config = make_train_config(
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    xgb_model = train_model(x_train, y_train, config)
    staging_path = tmp_path / "staging" / "model.ubj"
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(xgb_model, str(staging_path))
    model_bytes = staging_path.read_bytes()

    class FakeDownloader:
        """Fake downloader that writes model bytes to dest path."""

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
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(model_bytes)
            return HeadInfo(
                size=len(model_bytes),
                etag="fake-etag",
                content_type="application/octet-stream",
            )

    def fake_factory(base_url: str, api_key: str) -> FakeDownloader:
        return FakeDownloader()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0},
        region_encoder={"North America": 0},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    loaded_model = container.get_model()

    x_test: NDArray[np.float64] = np.zeros((1, 8), dtype=np.float64)
    prediction = loaded_model.predict_proba(x_test)
    assert prediction.shape == (1, 2)
    assert container.get_model_info()["is_loaded"] is True

    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_get_model_raises_when_data_bank_fails(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model raises FileNotFoundError when data-bank download fails."""
    from platform_core.data_bank_client import DataBankClientError, HeadInfo

    from covenant_radar_api.core import _test_hooks

    class FakeDownloaderFail:
        """Fake downloader that raises error."""

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
            raise DataBankClientError("connection refused")

    def fake_factory(base_url: str, api_key: str) -> FakeDownloaderFail:
        return FakeDownloaderFail()

    orig_factory = _test_hooks.data_bank_client_factory
    _test_hooks.data_bank_client_factory = fake_factory

    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(tmp_path / "nonexistent_model.ubj"),
        model_output_dir=tmp_path,
        sector_encoder={},
        region_encoder={},
        ml_backend="xgboost",
        data_bank_url="https://data-bank.example.com",
        data_bank_key="test-api-key",
        data_bank_model_file_id="active_xgb.ubj",
    )

    with pytest.raises(FileNotFoundError, match="Train a model first"):
        container.get_model()

    container.close()
    fake_kv_client.assert_only_called({"sadd", "close"})
    _test_hooks.data_bank_client_factory = orig_factory


def test_container_get_model_lightgbm_backend_loads_successfully(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model loads LightGBM model via hook when file exists."""
    from covenant_radar_api.worker import _test_hooks as worker_hooks

    model_path = tmp_path / "test_lightgbm.txt"
    model_path.write_text("fake lightgbm model")

    class FakeLightGBMPredictor:
        """Fake predictor for LightGBM model."""

        def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
            """Return fake predictions."""
            n_samples = int(x.shape[0])
            return np.column_stack(
                [
                    np.full(n_samples, 0.3, dtype=np.float64),
                    np.full(n_samples, 0.7, dtype=np.float64),
                ]
            )

    fake_predictor = FakeLightGBMPredictor()
    orig_loader = worker_hooks.lightgbm_loader
    worker_hooks.lightgbm_loader = lambda model_path: fake_predictor

    try:
        container = ServiceContainer(
            settings=test_settings,
            redis=fake_kv_client,
            db_conn=InMemoryConnection(in_memory_store),
            redis_rq=fake_rq_client,
            model_path=str(model_path),
            model_output_dir=tmp_path,
            sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
            region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
            ml_backend="lightgbm",
        )

        loaded_model = container.get_model()

        x_test: NDArray[np.float64] = np.zeros((1, 8), dtype=np.float64)
        prediction = loaded_model.predict_proba(x_test)
        assert prediction.shape == (1, 2)
        assert container.get_model_info()["is_loaded"] is True

        container.close()
        fake_kv_client.assert_only_called({"sadd", "close"})
    finally:
        worker_hooks.lightgbm_loader = orig_loader
