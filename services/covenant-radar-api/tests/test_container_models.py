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
from lightgbm.basic import LightGBMError
from numpy.typing import NDArray
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from covenant_radar_api.core import ServiceContainer
from covenant_radar_api.core.config import Settings

from .conftest import ContainerAndStore


def test_container_load_model_now_returns_false_when_file_missing(
    container_with_store: ContainerAndStore,
) -> None:
    """Test load_model_now returns False when model file doesn't exist."""
    # Don't create model file - it should not exist
    result = container_with_store.container.load_model_now()
    assert result is False
    # Model should still not be loaded
    assert container_with_store.container.get_model_info()["is_loaded"] is False


def test_container_load_model_now_returns_true_when_file_exists(
    container_with_store: ContainerAndStore,
) -> None:
    """Test load_model_now returns True and loads model when file exists."""
    # Create a real model file at the expected path
    model_path = Path(container_with_store.container.get_model_info()["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

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
    model = train_model(x_train, y_train, config)
    save_model(model, str(model_path))

    # Now load model
    result = container_with_store.container.load_model_now()
    assert result is True
    assert container_with_store.container.get_model_info()["is_loaded"] is True


def test_container_load_mlp_model_raises_file_not_found(
    container_with_store: ContainerAndStore,
    tmp_path: Path,
) -> None:
    """Test _load_mlp_model raises FileNotFoundError when metadata missing."""
    container = container_with_store.container
    # Create model file but not metadata
    model_path = tmp_path / "test_mlp.pt"
    model_path.write_bytes(b"fake model")

    with pytest.raises(FileNotFoundError) as exc_info:
        container._load_mlp_model(str(model_path))

    assert "active_mlp_meta.json" in str(exc_info.value)


def test_container_load_model_now_mlp_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now raises FileNotFoundError when MLP metadata missing."""
    # Create a model file but no metadata so load_model_now fails
    model_path = tmp_path / "test_mlp.pt"
    model_path.write_bytes(b"fake mlp model")

    # Create container with MLP backend
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="mlp",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.load_model_now()

    assert "active_mlp_meta.json" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_get_model_mlp_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model raises FileNotFoundError when MLP metadata missing."""
    # Create model file but no metadata
    model_path = tmp_path / "test_mlp.pt"
    model_path.write_bytes(b"fake mlp model")

    # Create container with MLP backend
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="mlp",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.get_model()

    assert "active_mlp_meta.json" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_load_lstm_model_raises_file_not_found(
    container_with_store: ContainerAndStore,
    tmp_path: Path,
) -> None:
    """Test _load_lstm_model raises FileNotFoundError when metadata missing."""
    container = container_with_store.container
    # Create model file but not metadata
    model_path = tmp_path / "test_lstm.pt"
    model_path.write_bytes(b"fake model")

    with pytest.raises(FileNotFoundError) as exc_info:
        container._load_lstm_model(str(model_path))

    assert "active_lstm_meta.json" in str(exc_info.value)


def test_container_load_lightgbm_model_raises_lightgbm_error(
    container_with_store: ContainerAndStore,
    tmp_path: Path,
) -> None:
    """Test _load_lightgbm_model raises LightGBMError when model missing."""
    container = container_with_store.container
    model_path = str(tmp_path / "test_lightgbm.txt")

    with pytest.raises(LightGBMError) as exc_info:
        container._load_lightgbm_model(model_path)

    assert "test_lightgbm.txt" in str(exc_info.value)


def test_container_load_model_now_lstm_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now raises FileNotFoundError when LSTM metadata missing."""
    # Create a model file but no metadata
    model_path = tmp_path / "test_lstm.pt"
    model_path.write_bytes(b"fake lstm model")

    # Create container with LSTM backend
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="lstm",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.load_model_now()

    assert "active_lstm_meta.json" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_load_model_now_lightgbm_backend_returns_false_when_missing(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now returns False when LightGBM model missing."""
    # Create container with LightGBM backend but no model file
    model_path = tmp_path / "test_lightgbm.txt"
    # Don't create the file

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

    result = container.load_model_now()

    assert result is False
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_load_model_now_lightgbm_backend_returns_true(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test load_model_now returns True when LightGBM model loads successfully.

    Uses test hook injection to provide a fake model loader, covering
    the lightgbm branch in load_model_now (container.py line 326).
    """
    from covenant_radar_api.worker import _test_hooks as worker_hooks

    # Create model file
    model_path = tmp_path / "test_lightgbm.txt"
    model_path.write_text("fake lightgbm model")

    # Create fake predictor
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

        result = container.load_model_now()

        assert result is True
        assert container.get_model_info()["is_loaded"] is True
        container.close()
    finally:
        worker_hooks.lightgbm_loader = orig_loader


def test_container_get_model_lstm_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model raises FileNotFoundError when LSTM metadata missing."""
    # Create model file but no metadata
    model_path = tmp_path / "test_lstm.pt"
    model_path.write_bytes(b"fake lstm model")

    # Create container with LSTM backend
    container = ServiceContainer(
        settings=test_settings,
        redis=fake_kv_client,
        db_conn=InMemoryConnection(in_memory_store),
        redis_rq=fake_rq_client,
        model_path=str(model_path),
        model_output_dir=tmp_path,
        sector_encoder={"Technology": 0, "Finance": 1, "Healthcare": 2},
        region_encoder={"North America": 0, "Europe": 1, "Asia": 2},
        ml_backend="lstm",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        container.get_model()

    assert "active_lstm_meta.json" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True


def test_container_get_model_lightgbm_backend_raises_file_not_found(
    tmp_path: Path,
    in_memory_store: InMemoryStore,
    fake_kv_client: FakeRedis,
    fake_rq_client: FakeRedisBytesClient,
    test_settings: Settings,
) -> None:
    """Test get_model raises FileNotFoundError when LightGBM model missing.

    When the model file doesn't exist and data-bank is not configured,
    get_model raises FileNotFoundError with a descriptive message.
    """
    # Create container with LightGBM backend but no model file
    model_path = tmp_path / "test_lightgbm.txt"
    # Don't create the file

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

    with pytest.raises(FileNotFoundError) as exc_info:
        container.get_model()

    # Check that the error message includes the path and helpful info
    assert "test_lightgbm.txt" in str(exc_info.value)
    assert "Train a model first" in str(exc_info.value)
    container.close()
    # sadd is called during fixture setup (adds worker to rq:workers)
    fake_kv_client.assert_only_called({"sadd", "close"})
    assert fake_rq_client.closed is True
