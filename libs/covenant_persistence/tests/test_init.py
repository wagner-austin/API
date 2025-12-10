"""Tests for package initialization and exports."""

from __future__ import annotations

from covenant_persistence import (
    ConnectCallable,
    ConnectionProtocol,
    CovenantRepository,
    CovenantResultRepository,
    CursorProtocol,
    DealRepository,
    MeasurementRepository,
    PostgresCovenantRepository,
    PostgresCovenantResultRepository,
    PostgresDealRepository,
    PostgresMeasurementRepository,
    connect,
)


class TestPackageExports:
    """Tests that verify all expected symbols are exported and usable."""

    def test_connect_callable_is_protocol(self) -> None:
        """ConnectCallable is a Protocol type."""
        assert ConnectCallable.__class__.__name__ == "_ProtocolMeta"

    def test_connection_protocol_is_protocol(self) -> None:
        """ConnectionProtocol is a Protocol type."""
        assert ConnectionProtocol.__class__.__name__ == "_ProtocolMeta"

    def test_cursor_protocol_is_protocol(self) -> None:
        """CursorProtocol is a Protocol type."""
        assert CursorProtocol.__class__.__name__ == "_ProtocolMeta"

    def test_connect_is_callable(self) -> None:
        """connect function is callable."""
        assert callable(connect)

    def test_deal_repository_is_protocol(self) -> None:
        """DealRepository is a Protocol type."""
        assert DealRepository.__class__.__name__ == "_ProtocolMeta"

    def test_covenant_repository_is_protocol(self) -> None:
        """CovenantRepository is a Protocol type."""
        assert CovenantRepository.__class__.__name__ == "_ProtocolMeta"

    def test_measurement_repository_is_protocol(self) -> None:
        """MeasurementRepository is a Protocol type."""
        assert MeasurementRepository.__class__.__name__ == "_ProtocolMeta"

    def test_covenant_result_repository_is_protocol(self) -> None:
        """CovenantResultRepository is a Protocol type."""
        assert CovenantResultRepository.__class__.__name__ == "_ProtocolMeta"

    def test_postgres_deal_repository_is_class(self) -> None:
        """PostgresDealRepository is a class."""
        assert PostgresDealRepository.__class__.__name__ == "type"

    def test_postgres_covenant_repository_is_class(self) -> None:
        """PostgresCovenantRepository is a class."""
        assert PostgresCovenantRepository.__class__.__name__ == "type"

    def test_postgres_measurement_repository_is_class(self) -> None:
        """PostgresMeasurementRepository is a class."""
        assert PostgresMeasurementRepository.__class__.__name__ == "type"

    def test_postgres_covenant_result_repository_is_class(self) -> None:
        """PostgresCovenantResultRepository is a class."""
        assert PostgresCovenantResultRepository.__class__.__name__ == "type"
