"""Tests for capabilities: scanning for observability and streaming libraries."""

from __future__ import annotations

from pathlib import Path

from platform_kaggle.capabilities import scan_codebase


class TestScanCodebaseObservability:
    """Scanning for observability and streaming libraries."""

    def test_scan_codebase_with_ddtrace(self, tmp_path: Path) -> None:
        """Test scanning codebase with Datadog ddtrace dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "api_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "api-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
ddtrace = "^2.0.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "datadog_apm" in cap_names

    def test_scan_codebase_with_prometheus(self, tmp_path: Path) -> None:
        """Test scanning codebase with prometheus-client dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "metrics_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "metrics-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
prometheus-client = "^0.17.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "prometheus_metrics" in cap_names

    def test_scan_codebase_with_opentelemetry(self, tmp_path: Path) -> None:
        """Test scanning codebase with OpenTelemetry dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "otel_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "otel-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
opentelemetry-api = "^1.20.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "opentelemetry" in cap_names

    def test_scan_codebase_with_confluent_kafka(self, tmp_path: Path) -> None:
        """Test scanning codebase with confluent-kafka dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "streaming_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "streaming-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
confluent-kafka = "^2.3.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "confluent_kafka" in cap_names

    def test_scan_codebase_with_kafka_python(self, tmp_path: Path) -> None:
        """Test scanning codebase with kafka-python dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "kafka_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "kafka-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
kafka-python = "^2.0.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "kafka_python" in cap_names

    def test_scan_codebase_with_redis(self, tmp_path: Path) -> None:
        """Test scanning codebase with redis dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "cache_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "cache-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
redis = "^5.0.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "redis" in cap_names

    def test_scan_codebase_with_pika(self, tmp_path: Path) -> None:
        """Test scanning codebase with pika (RabbitMQ) dependency."""
        libs_dir = tmp_path / "libs"
        libs_dir.mkdir()

        services_dir = tmp_path / "services"
        services_dir.mkdir()

        service_dir = services_dir / "mq_service"
        service_dir.mkdir()

        pyproject = service_dir / "pyproject.toml"
        pyproject.write_text(
            """
[tool.poetry]
name = "mq-service"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
pika = "^1.3.0"
"""
        )

        profile = scan_codebase(tmp_path)

        cap_names = [c.name for c in profile.capabilities]
        assert "rabbitmq" in cap_names
