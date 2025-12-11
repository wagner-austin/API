"""Integration tests for evaluate routes."""

from __future__ import annotations

from typing import Literal

from covenant_domain import Covenant, CovenantId, DealId, Measurement
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_list

from covenant_radar_api.api.routes.evaluate import build_router

from .conftest import ContainerAndStore

# Period constants for tests
Q1_START = "2024-01-01"
Q1_END = "2024-03-31"


def _create_test_client(cas: ContainerAndStore) -> TestClient:
    """Create test client with real container."""
    app = FastAPI()
    router = build_router(cas.container)
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _add_test_covenant(
    cas: ContainerAndStore,
    cov_id: str,
    deal_id: str,
    formula: str,
    threshold: int,
    direction: Literal["<=", ">="],
) -> None:
    """Add a test covenant to store."""
    cas.store.covenants[cov_id] = Covenant(
        id=CovenantId(value=cov_id),
        deal_id=DealId(value=deal_id),
        name="Test Covenant",
        formula=formula,
        threshold_value_scaled=threshold,
        threshold_direction=direction,
        frequency="QUARTERLY",
    )
    cas.store._covenant_order.append(cov_id)


def _add_measurement(
    cas: ContainerAndStore,
    deal_id: str,
    start: str,
    end: str,
    metric: str,
    value: int,
) -> None:
    """Add a test measurement to store."""
    cas.store.measurements.append(
        Measurement(
            deal_id=DealId(value=deal_id),
            period_start_iso=start,
            period_end_iso=end,
            metric_name=metric,
            metric_value_scaled=value,
        )
    )


class TestEvaluateEndpoint:
    """Tests for POST /evaluate."""

    def test_evaluate_single_covenant_ok(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation with single covenant that passes."""
        _add_test_covenant(container_with_store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "debt", 10_000_000)
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "ebitda", 5_000_000)

        client = _create_test_client(container_with_store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        result = narrow_json_to_dict(data[0])
        assert result["status"] == "OK"
        covenant_id = narrow_json_to_dict(result["covenant_id"])
        assert covenant_id["value"] == "c1"
        # 10M / 5M = 2.0, threshold is 4.0, so OK
        assert result["calculated_value_scaled"] == 2_000_000

    def test_evaluate_single_covenant_breach(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation with single covenant that breaches."""
        _add_test_covenant(container_with_store, "c1", "d1", "debt / ebitda", 3_000_000, "<=")
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "debt", 20_000_000)
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "ebitda", 5_000_000)

        client = _create_test_client(container_with_store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        result = narrow_json_to_dict(data[0])
        assert result["status"] == "BREACH"
        covenant_id = narrow_json_to_dict(result["covenant_id"])
        assert covenant_id["value"] == "c1"
        # 20M / 5M = 4.0 > 3.0 threshold, so BREACH

    def test_evaluate_near_breach(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation with near-breach status."""
        # Threshold is 3.0, tolerance is 10%, so near-breach band is 2.7-3.0
        _add_test_covenant(container_with_store, "c1", "d1", "debt / ebitda", 3_000_000, "<=")
        # 14M / 5M = 2.8, which is in near-breach band
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "debt", 14_000_000)
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "ebitda", 5_000_000)

        client = _create_test_client(container_with_store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        result = narrow_json_to_dict(data[0])
        assert result["status"] == "NEAR_BREACH"
        covenant_id = narrow_json_to_dict(result["covenant_id"])
        assert covenant_id["value"] == "c1"

    def test_evaluate_multiple_covenants(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation with multiple covenants."""
        _add_test_covenant(container_with_store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_test_covenant(container_with_store, "c2", "d1", "ebitda / interest", 2_000_000, ">=")
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "debt", 10_000_000)
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "ebitda", 5_000_000)
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "interest", 1_000_000)

        client = _create_test_client(container_with_store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 2

    def test_evaluate_no_covenants(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation when deal has no covenants."""
        client = _create_test_client(container_with_store)

        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 0

    def test_evaluate_missing_metric(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation when required metric is missing."""
        _add_test_covenant(container_with_store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "debt", 10_000_000)
        # Missing ebitda measurement

        client = _create_test_client(container_with_store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        # Should fail with KeyError (metric not found)
        assert response.status_code == 500

    def test_evaluate_invalid_json(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation with invalid JSON request."""
        client = _create_test_client(container_with_store)

        response = client.post("/evaluate", content=b"not valid json")

        assert response.status_code == 500

    def test_evaluate_missing_field(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation with missing required field."""
        client = _create_test_client(container_with_store)

        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01"
            }""",
        )

        # Missing period_end_iso and tolerance_ratio_scaled
        assert response.status_code == 500

    def test_evaluate_results_saved(self, container_with_store: ContainerAndStore) -> None:
        """Test that evaluation results are saved to repository."""
        _add_test_covenant(container_with_store, "c1", "d1", "debt / ebitda", 4_000_000, "<=")
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "debt", 10_000_000)
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "ebitda", 5_000_000)

        client = _create_test_client(container_with_store)
        client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        # Verify results were saved
        assert len(container_with_store.store.covenant_results) == 1
        assert container_with_store.store.covenant_results[0]["covenant_id"]["value"] == "c1"
        assert container_with_store.store.covenant_results[0]["status"] == "OK"

    def test_evaluate_greater_than_direction(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation with >= threshold direction."""
        # Interest coverage must be >= 2.0
        _add_test_covenant(container_with_store, "c1", "d1", "ebitda / interest", 2_000_000, ">=")
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "ebitda", 6_000_000)
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "interest", 2_000_000)

        client = _create_test_client(container_with_store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        assert len(data) == 1
        result = narrow_json_to_dict(data[0])
        # 6M / 2M = 3.0 >= 2.0 threshold, so OK
        assert result["status"] == "OK"
        covenant_id = narrow_json_to_dict(result["covenant_id"])
        assert covenant_id["value"] == "c1"
        assert result["calculated_value_scaled"] == 3_000_000

    def test_evaluate_greater_than_breach(self, container_with_store: ContainerAndStore) -> None:
        """Test evaluation with >= threshold direction breach."""
        # Interest coverage must be >= 2.0
        _add_test_covenant(container_with_store, "c1", "d1", "ebitda / interest", 2_000_000, ">=")
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "ebitda", 3_000_000)
        _add_measurement(container_with_store, "d1", Q1_START, Q1_END, "interest", 2_000_000)

        client = _create_test_client(container_with_store)
        response = client.post(
            "/evaluate",
            content=b"""{
                "deal_id": "d1",
                "period_start_iso": "2024-01-01",
                "period_end_iso": "2024-03-31",
                "tolerance_ratio_scaled": 100000
            }""",
        )

        assert response.status_code == 200
        data = narrow_json_to_list(load_json_str(response.text))
        result = narrow_json_to_dict(data[0])
        # 3M / 2M = 1.5 < 2.0 threshold, so BREACH
        assert result["status"] == "BREACH"
        covenant_id = narrow_json_to_dict(result["covenant_id"])
        assert covenant_id["value"] == "c1"
