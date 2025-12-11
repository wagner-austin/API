"""Tests for seed CLI script."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from covenant_persistence import ConnectionProtocol
from covenant_persistence.testing import InMemoryConnection, InMemoryStore

from covenant_radar_api.seeding import _test_hooks as seeding_hooks
from covenant_radar_api.seeding.runner import SeedResult
from scripts import seed as seed_mod

# =============================================================================
# Fixtures
# =============================================================================


def _make_in_memory_store() -> InMemoryStore:
    """Create fresh in-memory store."""
    return InMemoryStore()


def _reset_seed_script_hooks_impl() -> Generator[None, None, None]:
    """Reset seed script hooks after test."""
    orig_get_env = seed_mod.get_env
    orig_write = seed_mod.write
    yield
    seed_mod.get_env = orig_get_env
    seed_mod.write = orig_write


def _reset_seeding_hooks_impl() -> Generator[None, None, None]:
    """Reset seeding module hooks after test."""
    orig_conn = seeding_hooks.connection_factory
    orig_uuid = seeding_hooks.uuid_generator
    yield
    seeding_hooks.connection_factory = orig_conn
    seeding_hooks.uuid_generator = orig_uuid


in_memory_store = pytest.fixture(_make_in_memory_store)
_reset_seed_script_hooks = pytest.fixture(_reset_seed_script_hooks_impl)
_reset_seeding_hooks = pytest.fixture(_reset_seeding_hooks_impl)


# =============================================================================
# Tests
# =============================================================================


class TestGetDatabaseUrl:
    """Tests for _get_database_url function."""

    def test_returns_url_when_set(
        self,
        _reset_seed_script_hooks: None,
    ) -> None:
        """Test returns DATABASE_URL when environment variable is set."""

        def fake_get_env(key: str) -> str | None:
            if key == "DATABASE_URL":
                return "postgresql://test:test@localhost/test"
            return None

        seed_mod.get_env = fake_get_env

        result = seed_mod._get_database_url()
        assert result == "postgresql://test:test@localhost/test"

    def test_raises_when_not_set(
        self,
        _reset_seed_script_hooks: None,
    ) -> None:
        """Test raises RuntimeError when DATABASE_URL is not set."""

        def fake_get_env(key: str) -> str | None:
            return None

        seed_mod.get_env = fake_get_env

        with pytest.raises(RuntimeError, match="DATABASE_URL environment variable"):
            seed_mod._get_database_url()


class TestPrintResult:
    """Tests for _print_result function."""

    def test_prints_basic_counts(
        self,
        _reset_seed_script_hooks: None,
    ) -> None:
        """Test prints basic counts without verbose."""
        output: list[str] = []

        def fake_write(text: str) -> int:
            output.append(text)
            return len(text)

        seed_mod.write = fake_write

        result: SeedResult = {
            "deals_created": 4,
            "covenants_created": 5,
            "measurements_created": 100,
            "results_created": 25,
        }
        seed_mod._print_result(result, verbose=False)

        full_output = "".join(output)
        assert "Deals: 4" in full_output
        assert "Covenants: 5" in full_output
        assert "Measurements: 100" in full_output
        assert "Results: 25" in full_output
        assert "Total entities" not in full_output

    def test_prints_total_when_verbose(
        self,
        _reset_seed_script_hooks: None,
    ) -> None:
        """Test prints total entities when verbose."""
        output: list[str] = []

        def fake_write(text: str) -> int:
            output.append(text)
            return len(text)

        seed_mod.write = fake_write

        result: SeedResult = {
            "deals_created": 4,
            "covenants_created": 5,
            "measurements_created": 100,
            "results_created": 25,
        }
        seed_mod._print_result(result, verbose=True)

        full_output = "".join(output)
        assert "Total entities: 134" in full_output


class TestMain:
    """Tests for main function."""

    def test_seeds_database_and_returns_zero(
        self,
        in_memory_store: InMemoryStore,
        _reset_seed_script_hooks: None,
        _reset_seeding_hooks: None,
    ) -> None:
        """Test main seeds database and returns exit code 0."""
        output: list[str] = []
        created_conn: list[InMemoryConnection] = []

        def fake_get_env(key: str) -> str | None:
            if key == "DATABASE_URL":
                return "postgresql://test:test@localhost/test"
            return None

        def fake_write(text: str) -> int:
            output.append(text)
            return len(text)

        def fake_connection_factory(dsn: str) -> ConnectionProtocol:
            conn = InMemoryConnection(in_memory_store)
            created_conn.append(conn)
            return conn

        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        seed_mod.get_env = fake_get_env
        seed_mod.write = fake_write
        seeding_hooks.connection_factory = fake_connection_factory
        seeding_hooks.uuid_generator = fake_uuid

        exit_code = seed_mod.main([])

        assert exit_code == 0
        assert len(created_conn) == 1
        assert created_conn[0]._closed is True
        full_output = "".join(output)
        assert "Deals: 12" in full_output

    def test_verbose_flag_long(
        self,
        in_memory_store: InMemoryStore,
        _reset_seed_script_hooks: None,
        _reset_seeding_hooks: None,
    ) -> None:
        """Test --verbose flag enables verbose output."""
        output: list[str] = []

        def fake_get_env(key: str) -> str | None:
            if key == "DATABASE_URL":
                return "postgresql://test:test@localhost/test"
            return None

        def fake_write(text: str) -> int:
            output.append(text)
            return len(text)

        def fake_connection_factory(dsn: str) -> ConnectionProtocol:
            return InMemoryConnection(in_memory_store)

        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        seed_mod.get_env = fake_get_env
        seed_mod.write = fake_write
        seeding_hooks.connection_factory = fake_connection_factory
        seeding_hooks.uuid_generator = fake_uuid

        seed_mod.main(["--verbose"])

        full_output = "".join(output)
        assert "Total entities" in full_output

    def test_verbose_flag_short(
        self,
        in_memory_store: InMemoryStore,
        _reset_seed_script_hooks: None,
        _reset_seeding_hooks: None,
    ) -> None:
        """Test -v flag enables verbose output."""
        output: list[str] = []

        def fake_get_env(key: str) -> str | None:
            if key == "DATABASE_URL":
                return "postgresql://test:test@localhost/test"
            return None

        def fake_write(text: str) -> int:
            output.append(text)
            return len(text)

        def fake_connection_factory(dsn: str) -> ConnectionProtocol:
            return InMemoryConnection(in_memory_store)

        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        seed_mod.get_env = fake_get_env
        seed_mod.write = fake_write
        seeding_hooks.connection_factory = fake_connection_factory
        seeding_hooks.uuid_generator = fake_uuid

        seed_mod.main(["-v"])

        full_output = "".join(output)
        assert "Total entities" in full_output


class TestDefaultFunctions:
    """Tests for default hook implementations."""

    def test_default_get_env_returns_none_for_unset(self) -> None:
        """Test _default_get_env returns None for unset variables."""
        result = seed_mod._default_get_env("__DEFINITELY_NOT_SET_VAR__")
        assert result is None

    def test_default_write_returns_length(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test _default_write writes to stdout and returns length."""
        result = seed_mod._default_write("hello")
        assert result == 5
        captured = capsys.readouterr()
        assert captured.out == "hello"


class TestProtocols:
    """Tests for protocol definitions."""

    def test_write_func_protocol_is_callable_type(self) -> None:
        """Test WriteFunc protocol defines callable behavior."""
        # Verify protocol can be used for type annotation
        func: seed_mod.WriteFunc = seed_mod._default_write
        result = func("test")
        assert result == 4

    def test_get_env_func_protocol_is_callable_type(self) -> None:
        """Test GetEnvFunc protocol defines callable behavior."""
        # Verify protocol can be used for type annotation
        func: seed_mod.GetEnvFunc = seed_mod._default_get_env
        result = func("__NOT_SET__")
        assert result is None


class TestMainGuard:
    """Tests for if __name__ == '__main__' block."""

    def test_main_guard_calls_main(
        self,
        in_memory_store: InMemoryStore,
        _reset_seed_script_hooks: None,
        _reset_seeding_hooks: None,
    ) -> None:
        """Test the if __name__ == '__main__' guard executes main()."""
        import runpy
        from pathlib import Path

        from platform_core.config import _test_hooks as config_hooks

        orig_config_get_env = config_hooks.get_env

        def fake_get_env(key: str) -> str | None:
            if key == "DATABASE_URL":
                return "postgresql://test:test@localhost/test"
            return None

        def fake_connection_factory(dsn: str) -> ConnectionProtocol:
            return InMemoryConnection(in_memory_store)

        uuid_counter = [0]

        def fake_uuid() -> str:
            uuid_counter[0] += 1
            return f"uuid-{uuid_counter[0]}"

        # Set up the hooks before running the module
        # The seed.py script uses platform_core.config._test_hooks.get_env
        config_hooks.get_env = fake_get_env
        seeding_hooks.connection_factory = fake_connection_factory
        seeding_hooks.uuid_generator = fake_uuid

        # Find the seed.py file
        project_root = Path(__file__).parent.parent.parent
        seed_path = str(project_root / "scripts" / "seed.py")

        # Run seed.py as __main__ - expect SystemExit
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(seed_path, run_name="__main__")

        # Restore
        config_hooks.get_env = orig_config_get_env

        # main() returns 0 which becomes SystemExit(0)
        assert exc_info.value.code == 0
