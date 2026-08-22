"""Tests for production hook implementations in platform_email.testing."""

from __future__ import annotations

import io
import sys
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from platform_email._prod_hooks import (
    _prod_cli_get_env,
    _prod_cli_get_now,
    _prod_cli_set_env,
    _prod_console_input,
    _prod_console_output,
    _prod_current_time,
    _prod_file_exists,
    _prod_gmail_credentials_path,
    _prod_gmail_tokens_path,
    _prod_http_delete,
    _prod_http_get,
    _prod_http_patch,
    _prod_http_post,
    _prod_open_browser,
    _prod_outlook_credentials_path,
    _prod_outlook_tokens_path,
    _prod_read_file,
    _prod_read_file_bytes,
    _prod_write_file,
)
from platform_email.testing import (
    reset_hooks,
)


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


class TestProdCurrentTime:
    """Tests for _prod_current_time."""

    def test_returns_reasonable_timestamp(self) -> None:
        """Test that timestamp is reasonable (after 2020)."""
        result = _prod_current_time()
        # Unix timestamp for 2020-01-01, validates it's a reasonable integer
        assert result > 1577836800


class TestProdFileOperations:
    """Tests for file operation hooks."""

    def test_file_exists_returns_true_for_existing(self, tmp_path: Path) -> None:
        """Test file_exists returns True for existing file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content", encoding="utf-8")
        assert _prod_file_exists(str(file_path)) is True

    def test_file_exists_returns_false_for_missing(self) -> None:
        """Test file_exists returns False for missing file."""
        result = _prod_file_exists("/nonexistent/path/to/file.txt")
        assert result is False

    def test_write_and_read_file(self, tmp_path: Path) -> None:
        """Test writing and reading a file."""
        path = str(tmp_path / "test.txt")
        _prod_write_file(path, "Hello, World!")
        content = _prod_read_file(path)
        assert content == "Hello, World!"

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that write_file creates parent directories."""
        path = str(tmp_path / "nested" / "dir" / "test.txt")
        _prod_write_file(path, "Content")
        assert _prod_file_exists(path) is True

    def test_read_file_bytes(self, tmp_path: Path) -> None:
        """Test reading a file as raw bytes."""
        file_path = tmp_path / "binary.dat"
        binary_content = b"\x89PNG\r\n\x1a\n\x00\x01"
        file_path.write_bytes(binary_content)
        result = _prod_read_file_bytes(str(file_path))
        assert result == binary_content


class TestProdConsoleOperations:
    """Tests for console operation hooks."""

    def test_console_output_writes_to_stdout(self) -> None:
        """Test console_output writes to stdout."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            _prod_console_output("Test message")
            output = sys.stdout.getvalue()
            assert "Test message" in output
        finally:
            sys.stdout = old_stdout

    def test_console_input_reads_from_stdin(self) -> None:
        """Test console_input reads from stdin."""
        old_stdin = sys.stdin
        sys.stdin = io.StringIO("test input\n")
        try:
            result = _prod_console_input("Enter: ")
            assert result == "test input"
        finally:
            sys.stdin = old_stdin


class TestProdOpenBrowser:
    """Tests for _prod_open_browser."""

    def test_open_browser_is_callable(self) -> None:
        """Test that open_browser is a callable function."""
        # We can't easily test that it opens a browser without mocking,
        # but we can verify the function exists and is callable
        assert callable(_prod_open_browser)

    def test_open_browser_executes_without_error(self) -> None:
        """Test that open_browser can be called.

        Note: In CI environments without a browser, webbrowser.open() may
        fail silently or return False. We just verify it can be called.
        """
        # Using a data URL that won't actually open anything meaningful
        # webbrowser.open() may fail silently in headless environments
        _prod_open_browser("data:text/html,<html></html>")


class TestProdPathHooks:
    """Tests for path hook functions."""

    def test_outlook_tokens_path_contains_expected_filename(self) -> None:
        """Test outlook_tokens_path contains expected filename."""
        result = _prod_outlook_tokens_path()
        assert "email_tokens.json" in result
        assert ".microsoft" in result

    def test_outlook_credentials_path_contains_expected_filename(self) -> None:
        """Test outlook_credentials_path contains expected filename."""
        result = _prod_outlook_credentials_path()
        assert "email_credentials.json" in result
        assert ".microsoft" in result

    def test_gmail_tokens_path_contains_expected_filename(self) -> None:
        """Test gmail_tokens_path contains expected filename."""
        result = _prod_gmail_tokens_path()
        assert "email_tokens.json" in result
        assert ".google" in result

    def test_gmail_credentials_path_contains_expected_filename(self) -> None:
        """Test gmail_credentials_path contains expected filename."""
        result = _prod_gmail_credentials_path()
        assert "email_credentials.json" in result
        assert ".google" in result


class _TestHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for testing."""

    def log_message(self, format: str, *args: str) -> None:
        """Suppress logging."""
        pass

    def do_GET(self) -> None:
        """Handle GET request."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def do_POST(self) -> None:
        """Handle POST request."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"received": true, "body_length": ' + str(len(body)).encode() + b"}")

    def do_PATCH(self) -> None:
        """Handle PATCH request."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"patched": true, "body_length": ' + str(len(body)).encode() + b"}")

    def do_DELETE(self) -> None:
        """Handle DELETE request."""
        self.send_response(204)
        self.end_headers()


def _create_test_server() -> tuple[HTTPServer, int]:
    """Create a test HTTP server on a random port.

    Returns:
        Tuple of (server, port).
    """
    server = HTTPServer(("127.0.0.1", 0), _TestHTTPHandler)
    address = server.server_address
    port_value = address[1]
    if not isinstance(port_value, int):
        msg = f"Expected int port, got {type(port_value).__name__}"
        raise TypeError(msg)
    return server, port_value


@pytest.fixture()
def test_server() -> Generator[str, None, None]:
    """Create a test HTTP server."""
    server, port = _create_test_server()
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


class TestProdHttpGet:
    """Tests for _prod_http_get."""

    def test_get_returns_response_body(self, test_server: str) -> None:
        """Test GET returns response body."""
        result = _prod_http_get(f"{test_server}/test", {"Accept": "application/json"})
        assert "status" in result
        assert "ok" in result

    def test_get_with_headers(self, test_server: str) -> None:
        """Test GET sends headers."""
        result = _prod_http_get(
            f"{test_server}/test",
            {"Accept": "application/json", "X-Custom": "value"},
        )
        assert "status" in result


class TestProdHttpPost:
    """Tests for _prod_http_post."""

    def test_post_returns_response_body(self, test_server: str) -> None:
        """Test POST returns response body."""
        result = _prod_http_post(
            f"{test_server}/test",
            {"Content-Type": "application/json"},
            '{"data": "test"}',
        )
        assert "received" in result
        assert "true" in result

    def test_post_sends_body(self, test_server: str) -> None:
        """Test POST sends request body."""
        body = '{"key": "value"}'
        result = _prod_http_post(
            f"{test_server}/test",
            {"Content-Type": "application/json"},
            body,
        )
        assert f'"body_length": {len(body)}' in result


class TestProdHttpPatch:
    """Tests for _prod_http_patch."""

    def test_patch_returns_response_body(self, test_server: str) -> None:
        """Test PATCH returns response body."""
        result = _prod_http_patch(
            f"{test_server}/test",
            {"Content-Type": "application/json"},
            '{"update": "data"}',
        )
        assert "patched" in result
        assert "true" in result


class TestProdHttpDelete:
    """Tests for _prod_http_delete."""

    def test_delete_succeeds(self, test_server: str) -> None:
        """Test DELETE completes without error."""
        # Should not raise
        _prod_http_delete(f"{test_server}/test", {"Accept": "application/json"})


class TestProdCliGetEnv:
    """Tests for _prod_cli_get_env."""

    def test_get_env_returns_none_for_missing_key(self) -> None:
        """Test get_env returns None for missing key."""
        result = _prod_cli_get_env("NONEXISTENT_KEY_12345")
        assert result is None

    def test_get_env_uses_cache(self) -> None:
        """Test get_env uses internal cache for repeated calls."""
        # Reset cache
        reset_hooks()
        # Set a value via set_env
        _prod_cli_set_env("CACHED_KEY", "cached_value")
        # Get should return the cached value
        result = _prod_cli_get_env("CACHED_KEY")
        assert result == "cached_value"

    def test_get_env_loads_from_dotenv_file(self) -> None:
        """Test get_env loads values from .env file when it exists."""
        import os

        import platform_email.testing as testing_module

        # Get the path where the code will look for .env
        module_dir = os.path.dirname(testing_module.__file__)
        env_path = os.path.join(module_dir, "..", "..", ".env")
        env_path = os.path.normpath(env_path)

        # Reset hooks to clear the loaded flag
        reset_hooks()

        # Create a temporary .env file
        env_content = (
            "TEST_DOTENV_KEY=test_dotenv_value\n# Comment line\nANOTHER_KEY=another_value\n"
        )
        env_existed = os.path.exists(env_path)
        original_content = ""
        if env_existed:
            with open(env_path, encoding="utf-8") as f:
                original_content = f.read()

        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)

            # Now call _prod_cli_get_env which should load the .env file
            result = _prod_cli_get_env("TEST_DOTENV_KEY")

            # Verify the value was loaded
            assert result == "test_dotenv_value"

            # Also verify the second key was loaded
            result2 = _prod_cli_get_env("ANOTHER_KEY")
            assert result2 == "another_value"
        finally:
            # Clean up - restore original state
            if env_existed:
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
            else:
                if os.path.exists(env_path):
                    os.remove(env_path)


class TestProdCliGetEnvNoFile:
    """Tests for _prod_cli_get_env when .env file doesn't exist."""

    def test_get_env_handles_missing_dotenv_file(self) -> None:
        """Test get_env works when .env file doesn't exist."""
        import os

        import platform_email.testing as testing_module

        # Get the path where the code will look for .env
        module_dir = os.path.dirname(testing_module.__file__)
        env_path = os.path.join(module_dir, "..", "..", ".env")
        env_path = os.path.normpath(env_path)

        # Reset hooks to clear the loaded flag
        reset_hooks()

        # Save existing file if present
        env_existed = os.path.exists(env_path)
        original_content = ""
        if env_existed:
            with open(env_path, encoding="utf-8") as f:
                original_content = f.read()
            os.remove(env_path)

        try:
            # Now call _prod_cli_get_env - should handle missing file gracefully
            result = _prod_cli_get_env("NONEXISTENT_KEY")

            # Should return None since file doesn't exist and key not in cache
            assert result is None
        finally:
            # Restore the file if it existed
            if env_existed:
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(original_content)


class TestProdCliSetEnv:
    """Tests for _prod_cli_set_env."""

    def test_set_env_updates_cache(self) -> None:
        """Test set_env updates the internal cache."""
        # Reset to start fresh
        reset_hooks()
        # Set a value
        _prod_cli_set_env("TEST_KEY_CLI", "test_value")
        # Get should return the value
        result = _prod_cli_get_env("TEST_KEY_CLI")
        assert result == "test_value"

    def test_set_env_overwrites_existing(self) -> None:
        """Test set_env can overwrite existing values."""
        reset_hooks()
        _prod_cli_set_env("OVERWRITE_KEY", "first")
        _prod_cli_set_env("OVERWRITE_KEY", "second")
        result = _prod_cli_get_env("OVERWRITE_KEY")
        assert result == "second"


class TestProdCliGetNow:
    """Tests for _prod_cli_get_now."""

    def test_get_now_returns_datetime_with_year(self) -> None:
        """Test get_now returns a datetime with expected attributes."""
        result = _prod_cli_get_now()
        # Verify it has datetime attributes (year, month, day, etc.)
        assert result.year >= 2020
        assert 1 <= result.month <= 12
        assert 1 <= result.day <= 31

    def test_get_now_returns_recent_time(self) -> None:
        """Test get_now returns a datetime close to now."""
        from datetime import datetime

        result = _prod_cli_get_now()
        now = datetime.now()
        # Should be within 1 second
        diff = abs((now - result).total_seconds())
        assert diff < 1.0
