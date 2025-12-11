"""Tests for seeding _test_hooks module."""

from __future__ import annotations

import re

from covenant_radar_api.seeding import _test_hooks


class TestDefaultUuidGenerator:
    """Tests for default UUID generator."""

    def test_returns_string_of_uuid_length(self) -> None:
        """Test default generator returns a string of UUID length."""
        result = _test_hooks._default_uuid_generator()
        assert len(result) == 36

    def test_returns_valid_uuid_format(self) -> None:
        """Test default generator returns valid UUID format."""
        result = _test_hooks._default_uuid_generator()
        # UUID format: 8-4-4-4-12 hex characters
        uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        # fullmatch returns the match if entire string matches, else None
        # Use a concrete assertion on the result
        assert uuid_pattern.fullmatch(result), f"UUID {result} does not match expected format"

    def test_generates_unique_values(self) -> None:
        """Test default generator produces unique values."""
        uuid1 = _test_hooks._default_uuid_generator()
        uuid2 = _test_hooks._default_uuid_generator()
        assert uuid1 != uuid2


class TestModuleLevelHooks:
    """Tests for module-level hook variables."""

    def test_connection_factory_is_callable(self) -> None:
        """Test connection_factory hook can be called."""
        # Verify it's a callable by checking it has __call__
        assert callable(_test_hooks.connection_factory)

    def test_uuid_generator_is_callable(self) -> None:
        """Test uuid_generator hook can be called."""
        assert callable(_test_hooks.uuid_generator)

    def test_uuid_generator_default_produces_valid_uuid(self) -> None:
        """Test uuid_generator default implementation produces valid UUID."""
        result = _test_hooks.uuid_generator()
        # Standard UUID is 36 characters (32 hex + 4 dashes)
        assert len(result) == 36
        # Check it has correct dash positions
        assert result[8] == "-"
        assert result[13] == "-"
        assert result[18] == "-"
        assert result[23] == "-"


class TestLoadPsycopgModule:
    """Tests for psycopg module loader."""

    def test_returns_real_module_when_hook_is_none(self) -> None:
        """Test _load_psycopg_module returns real psycopg when hook is None."""
        import pytest

        orig_hook = _test_hooks.load_psycopg_module_hook
        _test_hooks.load_psycopg_module_hook = None

        try:
            module = _test_hooks._load_psycopg_module()
            # Verify module has connect method (required by protocol)
            # We call it with invalid DSN expecting OperationalError
            psycopg = __import__("psycopg")
            operational_error: type[Exception] = psycopg.OperationalError

            with pytest.raises(operational_error):
                module.connect("host= dbname=x", autocommit=True)
        finally:
            _test_hooks.load_psycopg_module_hook = orig_hook

    def test_uses_hook_when_set(self) -> None:
        """Test _load_psycopg_module uses hook when set."""
        from covenant_persistence.testing import InMemoryConnection, InMemoryStore

        store = InMemoryStore()
        hook_called = [False]

        class FakePsycopgModule:
            """Fake psycopg module for testing."""

            def connect(self, dsn: str, autocommit: bool = False) -> InMemoryConnection:
                hook_called[0] = True
                return InMemoryConnection(store)

        def fake_hook() -> _test_hooks.PsycopgModuleProtocol:
            fake: _test_hooks.PsycopgModuleProtocol = FakePsycopgModule()
            return fake

        orig_hook = _test_hooks.load_psycopg_module_hook
        _test_hooks.load_psycopg_module_hook = fake_hook

        try:
            module = _test_hooks._load_psycopg_module()
            # Call connect to verify it's the fake
            module.connect("test-dsn", autocommit=True)
            assert hook_called[0] is True
        finally:
            _test_hooks.load_psycopg_module_hook = orig_hook


class TestPsycopgConnectAutocommit:
    """Tests for psycopg connect with autocommit."""

    def test_returns_connection_via_hook(self) -> None:
        """Test _psycopg_connect_autocommit returns connection via hook."""
        from covenant_persistence.testing import InMemoryConnection, InMemoryStore

        store = InMemoryStore()

        class FakePsycopgModule:
            """Fake psycopg module for testing."""

            def connect(self, dsn: str, autocommit: bool = False) -> InMemoryConnection:
                return InMemoryConnection(store)

        def fake_hook() -> _test_hooks.PsycopgModuleProtocol:
            fake: _test_hooks.PsycopgModuleProtocol = FakePsycopgModule()
            return fake

        orig_hook = _test_hooks.load_psycopg_module_hook
        _test_hooks.load_psycopg_module_hook = fake_hook

        try:
            # This will now hit the return conn line
            conn = _test_hooks._psycopg_connect_autocommit("test-dsn")
            # Verify we got a connection that works
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
        finally:
            _test_hooks.load_psycopg_module_hook = orig_hook

    def test_calls_real_psycopg_when_hook_is_none(self) -> None:
        """Test _psycopg_connect_autocommit calls real psycopg.connect.

        We test this by providing an invalid DSN and expecting psycopg
        to raise OperationalError, proving the function was called correctly.
        """
        import pytest

        orig_hook = _test_hooks.load_psycopg_module_hook
        _test_hooks.load_psycopg_module_hook = None

        try:
            # Load the error class with proper type annotation
            psycopg = __import__("psycopg")
            operational_error: type[Exception] = psycopg.OperationalError

            # The factory should raise OperationalError for invalid DSN
            with pytest.raises(operational_error):
                _test_hooks._psycopg_connect_autocommit("host= dbname=x")
        finally:
            _test_hooks.load_psycopg_module_hook = orig_hook


class TestExports:
    """Tests for module exports."""

    def test_all_exports_connection_factory(self) -> None:
        """Test ConnectionFactory is in __all__."""
        assert "ConnectionFactory" in _test_hooks.__all__

    def test_all_exports_uuid_generator(self) -> None:
        """Test UuidGenerator is in __all__."""
        assert "UuidGenerator" in _test_hooks.__all__

    def test_all_exports_connection_factory_hook(self) -> None:
        """Test connection_factory is in __all__."""
        assert "connection_factory" in _test_hooks.__all__

    def test_all_exports_uuid_generator_hook(self) -> None:
        """Test uuid_generator is in __all__."""
        assert "uuid_generator" in _test_hooks.__all__

    def test_all_has_exactly_four_exports(self) -> None:
        """Test __all__ has exactly 4 exports."""
        assert len(_test_hooks.__all__) == 4
