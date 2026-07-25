"""Tests for dashboard route.

Tests verify that the dashboard HTML is correctly served with all required
JavaScript functions for fetching deals and running predictions.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from covenant_radar_api.api.main import create_app

from .conftest import ContainerAndStore


class TestDashboardRoute:
    """Tests for GET /dashboard endpoint."""

    def test_dashboard_returns_html(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that /dashboard returns HTML response with status 200."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_escapes_all_api_derived_interpolations(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Every API-derived value is escaped before reaching innerHTML.

        deal_name and borrower are free text accepted by POST /deals, so an
        unescaped interpolation renders attacker-supplied markup in whichever
        operator opens the dashboard. It also breaks plain rendering for any
        deal name containing & or <.

        Args:
            container_with_store: Fixture providing the wired container.
        """
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        html = client.get("/dashboard").text

        assert "function escapeHtml(" in html

        # No raw API property may reach a template literal; each is escaped at
        # the point it is bound to a local.
        raw_property_interpolations = (
            "${p.deal_name",
            "${p.borrower",
            "${p.risk_tier",
            "${data.model_path",
            "${data.model_id",
            "${job.status}",
            "${job.job_id",
        )
        for raw in raw_property_interpolations:
            assert raw not in html

        escaped_bindings = (
            "const displayName = escapeHtml(",
            "const displayBorrower = escapeHtml(",
            "const tier = escapeHtml(",
            "const pathVal = escapeHtml(",
            "const modelId = escapeHtml(",
            "const status = escapeHtml(job.status)",
            "const jobId = escapeHtml(",
        )
        for binding in escaped_bindings:
            assert binding in html

        # The job id also reaches a URL, which needs encoding rather than
        # HTML escaping.
        assert "/ml/jobs/${encodeURIComponent(jobId)}" in html

    def test_dashboard_loads_chart_js_locally_not_from_a_cdn(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Chart.js is served by this app, not fetched from a third party.

        jsDelivr's /npm/ path is dynamically generated and they document that
        Subresource Integrity must not be used with it, so the CDN script could
        not be pinned. Vendoring removes the third-party trust and lets the
        dashboard work without external network access.

        Args:
            container_with_store: Fixture providing the wired container.
        """
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        html = client.get("/dashboard").text

        assert "cdn.jsdelivr.net" not in html
        assert '<script src="/dashboard/chart.umd.min.js"></script>' in html

        asset = client.get("/dashboard/chart.umd.min.js")

        assert asset.status_code == 200
        assert "javascript" in asset.headers["content-type"]
        assert "Chart.js v4.4.1" in asset.text

    def test_dashboard_contains_title(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard HTML contains the expected title."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "<title>Covenant Radar Dashboard</title>" in html

    def test_dashboard_contains_fetch_deals_function(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard contains fetchDeals JavaScript function."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "async function fetchDeals()" in html
        assert "fetch('/deals')" in html

    def test_dashboard_contains_predict_deal_function(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard contains predictDeal JavaScript function."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "async function predictDeal(dealId)" in html
        assert "fetch('/ml/predict'" in html

    def test_dashboard_contains_fetch_and_predict_all_deals_function(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard contains fetchAndPredictAllDeals function."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "async function fetchAndPredictAllDeals()" in html

    def test_dashboard_contains_refresh_with_predictions_function(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard contains refreshWithPredictions function."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "async function refreshWithPredictions()" in html
        assert "await fetchAndPredictAllDeals()" in html

    def test_dashboard_calls_refresh_with_predictions_on_load(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard calls refreshWithPredictions on DOMContentLoaded."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "refreshWithPredictions()" in html
        assert "DOMContentLoaded" in html

    def test_dashboard_refresh_button_calls_refresh_with_predictions(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that the Refresh button calls refreshWithPredictions."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert 'onclick="refreshWithPredictions()"' in html

    def test_dashboard_contains_risk_distribution_section(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard contains risk distribution UI elements."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "Risk Distribution" in html
        assert "count-critical" in html
        assert "count-high" in html
        assert "count-medium" in html
        assert "count-low" in html

    def test_dashboard_contains_predictions_list_section(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard contains predictions list UI element."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "predictions-list" in html
        assert "Recent Predictions" in html

    def test_dashboard_contains_chart_containers(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that dashboard contains chart canvas elements."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        assert "trend-chart" in html
        assert "risk-chart" in html
        # Assert the chart library is actually loaded, rather than that some
        # string resembling its name appears somewhere in the page.
        assert '<script src="/dashboard/chart.umd.min.js"></script>' in html

    def test_dashboard_predictions_display_deal_name(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that prediction rendering uses deal_name for display."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        # Check that renderPredictions uses deal_name
        assert "p.deal_name" in html
        assert "displayName" in html

    def test_dashboard_predictions_display_borrower(
        self,
        container_with_store: ContainerAndStore,
    ) -> None:
        """Test that prediction rendering includes borrower info."""
        client: TestClient = TestClient(create_app(container_with_store.container.settings))

        response = client.get("/dashboard")
        html = response.text

        # Check that renderPredictions uses borrower
        assert "p.borrower" in html
        assert "displayBorrower" in html
