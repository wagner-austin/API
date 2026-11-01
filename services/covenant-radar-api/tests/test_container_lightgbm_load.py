"""The container loads a LightGBM model through its worker loader hook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from covenant_persistence.testing import InMemoryConnection, InMemoryStore
from numpy.typing import NDArray
from platform_core.config.covenant_radar import MLBackend
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from covenant_radar_api.core import ServiceContainer
from covenant_radar_api.core.config import Settings


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
            ml_backend=MLBackend.LIGHTGBM,
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
