"""Tests for Datadog tracing module."""

from __future__ import annotations

from covenant_radar_api.integrations.datadog import _test_hooks
from covenant_radar_api.integrations.datadog.tracing import (
    DatadogConfig,
    TracingState,
    get_tracing_state,
    make_default_datadog_config,
    reset_tracing_state,
    setup_datadog_tracing,
)


class TestSetupDatadogTracing:
    """Tests for setup_datadog_tracing function."""

    def test_setup_tracing_calls_hook(self) -> None:
        """Test that setup_datadog_tracing calls the tracing hook."""
        call_args: list[tuple[str, str, str]] = []

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            call_args.append((service, env, version))
            return True

        _test_hooks.tracing_setup = fake_tracing_setup

        result = setup_datadog_tracing(
            service="test-service",
            env="test-env",
            version="1.0.0",
        )

        assert len(call_args) == 1
        assert call_args[0] == ("test-service", "test-env", "1.0.0")
        assert result["configured"] is True
        assert result["service"] == "test-service"
        assert result["env"] == "test-env"
        assert result["version"] == "1.0.0"

    def test_setup_tracing_returns_state(self) -> None:
        """Test that setup returns TracingState."""

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            return True

        _test_hooks.tracing_setup = fake_tracing_setup

        result = setup_datadog_tracing(
            service="my-service",
            env="production",
            version="2.0.0",
        )

        assert result["configured"] is True
        assert result["service"] == "my-service"
        assert result["env"] == "production"
        assert result["version"] == "2.0.0"

    def test_setup_tracing_skips_if_already_configured(self) -> None:
        """Test that setup is skipped if already configured."""
        call_count = 0

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        _test_hooks.tracing_setup = fake_tracing_setup

        # First call
        result1 = setup_datadog_tracing(
            service="first-service",
            env="dev",
            version="1.0.0",
        )
        assert result1["configured"] is True
        assert result1["service"] == "first-service"
        assert call_count == 1

        # Second call - should be skipped
        result2 = setup_datadog_tracing(
            service="second-service",
            env="prod",
            version="2.0.0",
        )
        assert result2["configured"] is True
        assert result2["service"] == "first-service"  # Still first
        assert call_count == 1  # Not called again

    def test_setup_tracing_records_failure(self) -> None:
        """Test that failed setup is recorded in state."""

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            return False

        _test_hooks.tracing_setup = fake_tracing_setup

        result = setup_datadog_tracing(
            service="failing-service",
            env="dev",
            version="1.0.0",
        )

        assert result["configured"] is False
        assert result["service"] == "failing-service"


class TestGetTracingState:
    """Tests for get_tracing_state function."""

    def test_get_state_before_setup(self) -> None:
        """Test get_tracing_state returns unconfigured state initially."""
        reset_tracing_state()
        state = get_tracing_state()

        assert state["configured"] is False
        assert state["service"] == ""
        assert state["env"] == ""
        assert state["version"] == ""

    def test_get_state_after_setup(self) -> None:
        """Test get_tracing_state returns configured state after setup."""

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            return True

        _test_hooks.tracing_setup = fake_tracing_setup

        setup_datadog_tracing(
            service="configured-service",
            env="staging",
            version="3.0.0",
        )

        state = get_tracing_state()

        assert state["configured"] is True
        assert state["service"] == "configured-service"
        assert state["env"] == "staging"
        assert state["version"] == "3.0.0"


class TestResetTracingState:
    """Tests for reset_tracing_state function."""

    def test_reset_clears_state(self) -> None:
        """Test that reset clears the tracing state."""

        def fake_tracing_setup(service: str, env: str, version: str) -> bool:
            return True

        _test_hooks.tracing_setup = fake_tracing_setup

        # Setup first
        setup_datadog_tracing(
            service="my-service",
            env="prod",
            version="1.0.0",
        )
        assert get_tracing_state()["configured"] is True

        # Reset
        reset_tracing_state()

        state = get_tracing_state()
        assert state["configured"] is False
        assert state["service"] == ""
        assert state["env"] == ""
        assert state["version"] == ""


class TestMakeDefaultDatadogConfig:
    """Tests for make_default_datadog_config function."""

    def test_returns_valid_config(self) -> None:
        """Test that default config has all required fields."""
        config = make_default_datadog_config()

        assert config["enabled"] is False
        assert config["service"] == "covenant-radar-api"
        assert config["env"] == "dev"
        assert config["version"] == "0.0.0"
        assert config["agent_host"] == "localhost"
        assert config["dogstatsd_port"] == 8125
        assert config["trace_enabled"] is True


class TestTracingStateTypedDict:
    """Tests for TracingState TypedDict."""

    def test_tracing_state_fields(self) -> None:
        """Test TracingState has expected fields."""
        state: TracingState = {
            "configured": True,
            "service": "test",
            "env": "dev",
            "version": "1.0.0",
        }

        assert state["configured"] is True
        assert state["service"] == "test"
        assert state["env"] == "dev"
        assert state["version"] == "1.0.0"


class TestDatadogConfigTypedDict:
    """Tests for DatadogConfig TypedDict."""

    def test_datadog_config_fields(self) -> None:
        """Test DatadogConfig has expected fields."""
        config: DatadogConfig = {
            "enabled": True,
            "service": "my-service",
            "env": "production",
            "version": "2.0.0",
            "agent_host": "datadog-agent",
            "dogstatsd_port": 9125,
            "trace_enabled": True,
        }

        assert config["enabled"] is True
        assert config["service"] == "my-service"
        assert config["env"] == "production"
        assert config["version"] == "2.0.0"
        assert config["agent_host"] == "datadog-agent"
        assert config["dogstatsd_port"] == 9125
        assert config["trace_enabled"] is True
