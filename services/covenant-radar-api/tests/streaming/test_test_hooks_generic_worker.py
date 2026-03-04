"""Tests for streaming generic worker test hooks module."""

from __future__ import annotations

import re

from covenant_radar_api.streaming._test_hooks_generic_worker import (
    FakeTextGenerator,
    TextGeneratorProtocol,
    _real_current_iso_timestamp,
    _real_generate_uuid,
    _real_perf_counter,
    current_iso_timestamp,
    generate_uuid,
    perf_counter,
    use_real_hooks,
)


class TestRealPerfCounter:
    """Tests for _real_perf_counter."""

    def test_returns_positive_float(self) -> None:
        """Performance counter returns a positive float."""
        result: float = _real_perf_counter()
        assert result > 0.0

    def test_monotonic(self) -> None:
        """Successive calls return increasing values."""
        first: float = _real_perf_counter()
        second: float = _real_perf_counter()
        assert second >= first


class TestRealGenerateUuid:
    """Tests for _real_generate_uuid."""

    def test_returns_uuid_format(self) -> None:
        """Returns string in UUID4 hyphenated format."""
        result: str = _real_generate_uuid()
        assert len(result) == 36
        # UUID format: 8-4-4-4-12
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            result,
        )

    def test_unique(self) -> None:
        """Successive calls return different UUIDs."""
        first: str = _real_generate_uuid()
        second: str = _real_generate_uuid()
        assert first != second


class TestRealCurrentIsoTimestamp:
    """Tests for _real_current_iso_timestamp."""

    def test_returns_iso_format(self) -> None:
        """Returns ISO 8601 timestamp with Z suffix."""
        result: str = _real_current_iso_timestamp()
        assert "T" in result
        assert result.endswith("Z")

    def test_contains_date_parts(self) -> None:
        """Timestamp contains year, month, day, hour, minute, second."""
        result: str = _real_current_iso_timestamp()
        # Format: YYYY-MM-DDTHH:MM:SSZ
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            result,
        )


class TestModuleLevelHooks:
    """Tests for module-level hook defaults."""

    def test_perf_counter_returns_float(self) -> None:
        """Module-level perf_counter returns a positive float."""
        result: float = perf_counter()
        assert result > 0.0

    def test_generate_uuid_returns_string(self) -> None:
        """Module-level generate_uuid returns a 36-char UUID."""
        result: str = generate_uuid()
        assert len(result) == 36

    def test_current_iso_timestamp_returns_string(self) -> None:
        """Module-level current_iso_timestamp returns ISO timestamp."""
        result: str = current_iso_timestamp()
        assert "T" in result
        assert result.endswith("Z")


class TestFakeTextGenerator:
    """Tests for FakeTextGenerator."""

    def test_satisfies_protocol(self) -> None:
        """FakeTextGenerator satisfies TextGeneratorProtocol."""
        generator: TextGeneratorProtocol = FakeTextGenerator()
        result: str = generator.generate_text("test prompt")
        assert result == "Fake alert summary"

    def test_records_calls(self) -> None:
        """Calls are recorded in order."""
        generator = FakeTextGenerator()
        generator.generate_text("prompt one")
        generator.generate_text("prompt two")

        assert len(generator.calls) == 2
        assert generator.calls[0] == "prompt one"
        assert generator.calls[1] == "prompt two"

    def test_returns_configured_response(self) -> None:
        """Returns the configured next_response."""
        generator = FakeTextGenerator()
        generator.next_response = "Custom alert text"

        result: str = generator.generate_text("any prompt")

        assert result == "Custom alert text"

    def test_default_response(self) -> None:
        """Default response is 'Fake alert summary'."""
        generator = FakeTextGenerator()
        assert generator.next_response == "Fake alert summary"

    def test_empty_calls_initially(self) -> None:
        """Call list is empty on construction."""
        generator = FakeTextGenerator()
        assert generator.calls == []


class TestUseRealHooks:
    """Tests for use_real_hooks."""

    def test_restores_perf_counter(self) -> None:
        """use_real_hooks restores perf_counter to real implementation."""
        import covenant_radar_api.streaming._test_hooks_generic_worker as hooks

        # Override with a fake
        hooks.perf_counter = lambda: 999.0

        # Restore
        use_real_hooks()

        # Should be back to real implementation
        assert hooks.perf_counter is _real_perf_counter

    def test_restores_generate_uuid(self) -> None:
        """use_real_hooks restores generate_uuid to real implementation."""
        import covenant_radar_api.streaming._test_hooks_generic_worker as hooks

        # Override with a fake
        hooks.generate_uuid = lambda: "fake-uuid"

        # Restore
        use_real_hooks()

        assert hooks.generate_uuid is _real_generate_uuid

    def test_restores_current_iso_timestamp(self) -> None:
        """use_real_hooks restores current_iso_timestamp to real implementation."""
        import covenant_radar_api.streaming._test_hooks_generic_worker as hooks

        # Override with a fake
        hooks.current_iso_timestamp = lambda: "fake-timestamp"

        # Restore
        use_real_hooks()

        assert hooks.current_iso_timestamp is _real_current_iso_timestamp
