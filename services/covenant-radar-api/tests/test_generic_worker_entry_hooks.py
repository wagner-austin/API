"""Tests for the generic streaming worker entry hooks.

Every hook here reads something the entry point cannot construct itself: a
fitted state, a station map, a model, a Gemini client. The loaders validate
on the way in, because a malformed state surfaces otherwise as an index
error deep inside feature extraction, naming an array rather than the file
that was wrong.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from covenant_ml.datasets.types import encode_temporal_feature_state
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str

from covenant_radar_api.generic_worker_entry_hooks import (
    _real_logger_factory,
    _real_model_loader,
    _real_station_map_loader,
    _real_temporal_state_loader,
    _real_text_generator_factory,
)
from covenant_radar_api.integrations.google_ai._test_hooks import (
    use_fake_gemini,
    use_real_gemini,
)
from tests.domains.weather._test_weather_fixtures import make_flat_state


class TestTemporalStateLoader:
    """The fitted state round-trips through its published codec."""

    def test_loads_a_state_written_by_the_encoder(self, tmp_path: Path) -> None:
        """encode_temporal_feature_state output reads back unchanged.

        This is the contract that makes a fitted state deployable: training
        writes it, the worker reads it, and neither side knows the other.
        """
        state = make_flat_state(hot_threshold=4.0, cold_threshold=-3.0, mean=1.5)
        path = tmp_path / "state.json"
        path.write_text(dump_json_str(encode_temporal_feature_state(state)), encoding="utf-8")

        loaded = _real_temporal_state_loader(path)

        assert loaded["n_locations"] == state["n_locations"]
        assert loaded["thresholds"]["hot_threshold"] == state["thresholds"]["hot_threshold"]
        assert loaded["thresholds"]["cold_threshold"] == state["thresholds"]["cold_threshold"]
        assert loaded["seasonal_cycle"]["mean"] == state["seasonal_cycle"]["mean"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """An absent state file names the path rather than starting empty."""
        with pytest.raises(FileNotFoundError):
            _real_temporal_state_loader(tmp_path / "absent.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """A truncated file fails at load, not at the first observation."""
        path = tmp_path / "state.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(InvalidJsonError):
            _real_temporal_state_loader(path)

    def test_non_object_payload_raises(self, tmp_path: Path) -> None:
        """A JSON array is valid JSON but not a state."""
        path = tmp_path / "state.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(JSONTypeError):
            _real_temporal_state_loader(path)

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        """A state without n_locations is rejected where it is read."""
        state = make_flat_state()
        encoded = encode_temporal_feature_state(state)
        del encoded["n_locations"]
        path = tmp_path / "state.json"
        path.write_text(dump_json_str(encoded), encoding="utf-8")

        with pytest.raises(ValueError, match="n_locations"):
            _real_temporal_state_loader(path)


class TestStationMapLoader:
    """The station map is validated to integer indices."""

    def test_loads_station_indices(self, tmp_path: Path) -> None:
        """A well-formed map reads back as station to index."""
        path = tmp_path / "stations.json"
        path.write_text(dump_json_str({"station-a": 0, "station-b": 1}), encoding="utf-8")

        loaded = _real_station_map_loader(path)

        assert loaded == {"station-a": 0, "station-b": 1}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """An absent map names the path."""
        with pytest.raises(FileNotFoundError):
            _real_station_map_loader(tmp_path / "absent.json")

    def test_empty_map_raises(self, tmp_path: Path) -> None:
        """An empty map would featurise nothing, so it fails at startup.

        Left to run, every observation would raise KeyError on its station
        and the worker would dead-letter the entire stream.
        """
        path = tmp_path / "stations.json"
        path.write_text("{}", encoding="utf-8")

        with pytest.raises(JSONTypeError, match="empty"):
            _real_station_map_loader(path)

    def test_non_integer_index_raises(self, tmp_path: Path) -> None:
        """A string index is rejected here, not inside numpy indexing."""
        path = tmp_path / "stations.json"
        path.write_text(dump_json_str({"station-a": "zero"}), encoding="utf-8")

        with pytest.raises(JSONTypeError):
            _real_station_map_loader(path)

    def test_non_object_payload_raises(self, tmp_path: Path) -> None:
        """A JSON array is not a station map."""
        path = tmp_path / "stations.json"
        path.write_text("[0, 1]", encoding="utf-8")

        with pytest.raises(JSONTypeError):
            _real_station_map_loader(path)


class TestLoggerFactory:
    """The logger factory returns something the entry point can write to."""

    def test_returns_a_logger_that_accepts_messages(self) -> None:
        """info and error are the two methods the entry point calls."""
        logger = _real_logger_factory("test-generic-entry")

        logger.info("started")
        logger.error("stopped")


class TestRealModelLoader:
    """The model hook loads a real model file."""

    def test_loads_a_saved_xgboost_model(self, tmp_path: Path) -> None:
        """A model written by XGBoost reads back and predicts.

        Exercises the production loader against a real artifact rather than
        asserting the hook is merely callable, which would not run its body.
        """
        import numpy as np
        import xgboost as xgb
        from numpy.typing import NDArray

        rng = np.random.default_rng(0)
        x: NDArray[np.float64] = rng.normal(size=(60, 5)).astype(np.float64)
        first_column: NDArray[np.float64] = np.asarray(x[:, 0], dtype=np.float64)
        y: NDArray[np.int64] = (first_column > 0).astype(np.int64)
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, device="cpu")
        model.fit(x, y)
        model_path = tmp_path / "model.ubj"
        model.save_model(str(model_path))

        loaded = _real_model_loader(str(model_path))
        proba = loaded.predict_proba(x[:4])

        assert proba.shape == (4, 2)


class TestRealTextGeneratorFactory:
    """The text generator hook builds a Gemini-backed generator."""

    def test_builds_a_generator_without_calling_the_api(self) -> None:
        """Construction is offline; only generate_text would reach Gemini.

        The client factory is itself hooked, so this covers the production
        wiring without a network call or an API key.
        """
        use_fake_gemini()
        try:
            generator = _real_text_generator_factory("test-key", "gemini-2.0-flash")

            assert generator.generate_text("summarise this") != ""
        finally:
            use_real_gemini()
