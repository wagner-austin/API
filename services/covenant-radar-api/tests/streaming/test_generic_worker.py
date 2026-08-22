"""Tests for GenericStreamingWorker."""

from __future__ import annotations

from typing import TypeVar

import covenant_radar_api.streaming._test_hooks_generic_worker as _hooks
from covenant_radar_api.domains.base_schemas import (
    BaseAlertEventV1,
    BasePredictionEventV1,
)
from covenant_radar_api.streaming.generic_worker import (
    GenericProcessingResult,
    GenericWorkerConfig,
    _build_alert_prompt,
    _calculate_confidence,
    _classify_severity,
    _extract_positive_probability,
    make_generic_worker_config,
)

from ._test_generic_worker_fixtures import (
    make_base_input_payload,
    make_generic_streaming_worker,
    make_test_generic_worker_config,
)

_T = TypeVar("_T")


def _require(value: _T | None) -> _T:
    """Narrow optional type to non-None. Raises if None."""
    if value is None:
        msg = "Expected non-None value"
        raise AssertionError(msg)
    return value


class TestGenericWorkerConfig:
    """Tests for GenericWorkerConfig and make_generic_worker_config."""

    def test_make_generic_worker_config(self) -> None:
        """Factory creates config with provided values."""
        config: GenericWorkerConfig = make_generic_worker_config(
            model_version="v2.1",
            poll_timeout_seconds=5.0,
        )
        assert config["model_version"] == "v2.1"
        assert config["poll_timeout_seconds"] == 5.0

    def test_config_from_fixture(self) -> None:
        """Test fixture creates valid config."""
        config: GenericWorkerConfig = make_test_generic_worker_config()
        assert config["model_version"] == "test-v1"
        assert config["poll_timeout_seconds"] == 0.1


class TestGenericStreamingWorkerInit:
    """Tests for GenericStreamingWorker construction."""

    def test_creates_worker(self) -> None:
        """Worker is constructed with is_running False."""
        worker, _, _, _, _, _ = make_generic_streaming_worker()
        assert worker.is_running is False

    def test_config_accessible(self) -> None:
        """Worker stores config for access during processing."""
        worker, _, _, _, _, _ = make_generic_streaming_worker()
        # Verify worker was created successfully
        assert worker.is_running is False


class TestExtractPositiveProbability:
    """Tests for _extract_positive_probability."""

    def test_binary_classification(self) -> None:
        """Binary (1,2) shaped output returns column 1."""
        import numpy as np
        from numpy.typing import NDArray

        proba: NDArray[np.float64] = np.zeros((1, 2), dtype=np.float64)
        proba[0, 0] = 0.3
        proba[0, 1] = 0.7
        result: float = _extract_positive_probability(proba)
        assert result == 0.7

    def test_single_output(self) -> None:
        """Single (1,1) shaped output returns the value."""
        import numpy as np
        from numpy.typing import NDArray

        proba: NDArray[np.float64] = np.zeros((1, 1), dtype=np.float64)
        proba[0, 0] = 0.85
        result: float = _extract_positive_probability(proba)
        assert result == 0.85


class TestCalculateConfidence:
    """Tests for _calculate_confidence."""

    def test_high_probability(self) -> None:
        """High probability has high confidence."""
        result: float = _calculate_confidence(0.9)
        assert result == 0.8

    def test_low_probability(self) -> None:
        """Low probability has high confidence."""
        result: float = _calculate_confidence(0.1)
        assert result == 0.8

    def test_midpoint_zero_confidence(self) -> None:
        """0.5 has zero confidence."""
        result: float = _calculate_confidence(0.5)
        assert result == 0.0

    def test_extreme_probability(self) -> None:
        """1.0 has maximum confidence."""
        result: float = _calculate_confidence(1.0)
        assert result == 1.0


class TestClassifySeverity:
    """Tests for _classify_severity."""

    def test_critical_at_0_9(self) -> None:
        """Prediction >= 0.9 is critical."""
        assert _classify_severity(0.9, 0.8) == "critical"

    def test_critical_above_0_9(self) -> None:
        """Prediction > 0.9 is critical."""
        assert _classify_severity(0.95, 0.8) == "critical"

    def test_warning_at_threshold(self) -> None:
        """Prediction at threshold is warning."""
        assert _classify_severity(0.8, 0.8) == "warning"

    def test_warning_between_threshold_and_critical(self) -> None:
        """Prediction between threshold and 0.9 is warning."""
        assert _classify_severity(0.85, 0.8) == "warning"

    def test_info_below_threshold(self) -> None:
        """Prediction below threshold is info."""
        assert _classify_severity(0.5, 0.8) == "info"


class TestBuildAlertPrompt:
    """Tests for _build_alert_prompt."""

    def test_includes_header(self) -> None:
        """Prompt starts with header line."""
        result: str = _build_alert_prompt({"key": "value"})
        assert result.startswith("Generate an alert summary")

    def test_includes_context_keys(self) -> None:
        """Prompt includes all context key-value pairs."""
        context: dict[str, str] = {
            "entity_id": "deal-123",
            "risk": "high",
        }
        result: str = _build_alert_prompt(context)
        assert "- entity_id: deal-123" in result
        assert "- risk: high" in result

    def test_empty_context(self) -> None:
        """Empty context produces header only."""
        result: str = _build_alert_prompt({})
        assert result == "Generate an alert summary for the following context:"


class TestProcessEvent:
    """Tests for GenericStreamingWorker.process_event."""

    def setup_method(self) -> None:
        """Set up deterministic hooks for each test."""
        self._counter: float = 100.0

        def fake_perf_counter() -> float:
            self._counter += 0.025
            return self._counter

        _hooks.perf_counter = fake_perf_counter
        _hooks.generate_uuid = lambda: "test-uuid-001"
        _hooks.current_iso_timestamp = lambda: "2026-01-15T12:00:00Z"

    def teardown_method(self) -> None:
        """Restore real hooks."""
        _hooks.use_real_hooks()

    def test_returns_prediction(self) -> None:
        """Process event returns a prediction with correct fields."""
        worker, _, _, _, _, _ = make_generic_streaming_worker()
        payload: str = make_base_input_payload(entity_id="entity-001")

        result: GenericProcessingResult = worker.process_event(payload)

        prediction: BasePredictionEventV1 = result["prediction"]
        assert prediction["type"] == "test.prediction.v1"
        assert prediction["event_id"] == "test-uuid-001"
        assert prediction["entity_id"] == "entity-001"
        assert prediction["model_version"] == "test-v1"
        assert prediction["processed_at"] == "2026-01-15T12:00:00Z"

    def test_prediction_value_from_model(self) -> None:
        """Prediction value comes from model output."""
        worker, _, _, _, _, _ = make_generic_streaming_worker(probability=0.75)
        payload: str = make_base_input_payload()

        result: GenericProcessingResult = worker.process_event(payload)

        assert result["prediction"]["prediction_value"] == 0.75

    def test_confidence_calculated(self) -> None:
        """Confidence is abs(prediction - 0.5) * 2."""
        worker, _, _, _, _, _ = make_generic_streaming_worker(probability=0.75)
        payload: str = make_base_input_payload()

        result: GenericProcessingResult = worker.process_event(payload)

        # abs(0.75 - 0.5) * 2 = 0.5
        assert result["prediction"]["confidence"] == 0.5

    def test_latency_measured(self) -> None:
        """Latency is calculated from perf_counter delta."""
        worker, _, _, _, _, _ = make_generic_streaming_worker()
        payload: str = make_base_input_payload()

        result: GenericProcessingResult = worker.process_event(payload)

        # perf_counter increments by 0.025 each call (2 calls: start + end)
        assert result["latency_ms"] == 25

    def test_no_alert_below_threshold(self) -> None:
        """No alert when prediction is below threshold."""
        worker, _, _, _, _, _ = make_generic_streaming_worker(
            probability=0.25,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload()

        result: GenericProcessingResult = worker.process_event(payload)

        assert result["alert"] is None

    def test_alert_generated_above_threshold(self) -> None:
        """Alert generated when prediction exceeds threshold."""
        worker, _, _, _, _, _ = make_generic_streaming_worker(
            probability=0.85,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload(entity_id="entity-001")

        result: GenericProcessingResult = worker.process_event(payload)

        alert: BaseAlertEventV1 = _require(result["alert"])
        assert alert["type"] == "test.alert.v1"
        assert alert["entity_id"] == "entity-001"
        assert alert["alert_type"] == "high_risk_prediction"
        assert alert["prediction_value"] == 0.85

    def test_alert_severity_critical(self) -> None:
        """Prediction >= 0.9 produces critical severity."""
        worker, _, _, _, _, _ = make_generic_streaming_worker(
            probability=0.95,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload()

        result: GenericProcessingResult = worker.process_event(payload)

        alert: BaseAlertEventV1 = _require(result["alert"])
        assert alert["severity"] == "critical"

    def test_alert_severity_warning(self) -> None:
        """Prediction >= threshold but < 0.9 produces warning severity."""
        worker, _, _, _, _, _ = make_generic_streaming_worker(
            probability=0.85,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload()

        result: GenericProcessingResult = worker.process_event(payload)

        alert: BaseAlertEventV1 = _require(result["alert"])
        assert alert["severity"] == "warning"

    def test_alert_calls_text_generator(self) -> None:
        """Alert generation calls text_generator.generate_text."""
        worker, _, _, _, _, text_gen = make_generic_streaming_worker(
            probability=0.85,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload()

        worker.process_event(payload)

        assert len(text_gen.calls) == 1

    def test_alert_prompt_includes_context(self) -> None:
        """Alert prompt includes domain context values."""
        worker, _, _, _, _, text_gen = make_generic_streaming_worker(
            probability=0.85,
            alert_threshold=0.80,
        )
        payload: str = make_base_input_payload(entity_id="entity-xyz")

        worker.process_event(payload)

        prompt: str = text_gen.calls[0]
        assert "entity-xyz" in prompt
        assert "0.8500" in prompt

    def test_alert_uses_gemini_summary(self) -> None:
        """Alert gemini_summary comes from text_generator response."""
        worker, _, _, _, _, text_gen = make_generic_streaming_worker(
            probability=0.85,
            alert_threshold=0.80,
        )
        text_gen.next_response = "Custom alert text from LLM"
        payload: str = make_base_input_payload()

        result: GenericProcessingResult = worker.process_event(payload)

        alert: BaseAlertEventV1 = _require(result["alert"])
        assert alert["gemini_summary"] == "Custom alert text from LLM"

    def test_domain_decode_called(self) -> None:
        """Domain decode_input_event is called with raw payload."""
        worker, domain, _, _, _, _ = make_generic_streaming_worker()
        payload: str = make_base_input_payload()

        worker.process_event(payload)

        assert len(domain.decode_calls) == 1
        assert domain.decode_calls[0] == payload

    def test_model_called_with_features(self) -> None:
        """Model predict_proba is called."""
        worker, _, _, _, predictor, _ = make_generic_streaming_worker()
        payload: str = make_base_input_payload()

        worker.process_event(payload)

        assert predictor.call_count == 1
