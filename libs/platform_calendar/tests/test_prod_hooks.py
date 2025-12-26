"""Tests for production hook implementations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.errors import AppError, CalendarErrorCode

from platform_calendar.testing import (
    _prod_console_output,
    _prod_current_time,
    _prod_file_exists,
    _prod_http_delete,
    _prod_http_patch,
    _prod_load_credentials,
    _prod_load_tokens,
    _prod_read_file,
    _prod_save_tokens,
    _prod_write_file,
)
from platform_calendar.types import OAuthTokens


def _make_fake_env_get(env_vars: dict[str, str]) -> Callable[[str], str | None]:
    """Create a fake get_env function that returns values from provided dict."""

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    return fake_get_env


class TestProdCurrentTime:
    def test_returns_positive_timestamp(self) -> None:
        result = _prod_current_time()
        # Should return current timestamp which is > 0
        assert result > 0


class TestProdReadFile:
    def test_reads_file_content(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content", encoding="utf-8")

        result = _prod_read_file(str(test_file))
        assert result == "file content"


class TestProdWriteFile:
    def test_writes_file_content(self, tmp_path: Path) -> None:
        test_file = tmp_path / "output.txt"

        _prod_write_file(str(test_file), "written content")

        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == "written content"


class TestProdFileExists:
    def test_returns_true_for_existing_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "exists.txt"
        test_file.write_text("content", encoding="utf-8")

        result = _prod_file_exists(str(test_file))
        assert result is True

    def test_returns_false_for_nonexistent_file(self, tmp_path: Path) -> None:
        result = _prod_file_exists(str(tmp_path / "nonexistent.txt"))
        assert result is False


class TestProdConsoleOutput:
    def test_writes_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        _prod_console_output("Hello, World!")
        captured = capsys.readouterr()
        assert captured.out == "Hello, World!\n"


class TestProdLoadTokens:
    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        result = _prod_load_tokens(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        tokens_file = tmp_path / "tokens.json"
        tokens_file.write_text("not valid json", encoding="utf-8")
        result = _prod_load_tokens(str(tokens_file))
        assert result is None

    def test_returns_none_for_non_object_json(self, tmp_path: Path) -> None:
        tokens_file = tmp_path / "tokens.json"
        tokens_file.write_text("[]", encoding="utf-8")
        result = _prod_load_tokens(str(tokens_file))
        assert result is None

    def test_loads_valid_tokens(self, tmp_path: Path) -> None:
        tokens_file = tmp_path / "tokens.json"
        tokens_file.write_text(
            '{"access_token": "abc", "refresh_token": "def", '
            '"expires_at": 1735200000, "token_type": "Bearer"}',
            encoding="utf-8",
        )
        result = _prod_load_tokens(str(tokens_file))
        assert result  # not None
        assert result["access_token"] == "abc"
        assert result["refresh_token"] == "def"

    def test_loads_from_env_vars(self) -> None:
        """Test loading tokens from environment variables."""
        original_get_env = config_test_hooks.get_env
        config_test_hooks.get_env = _make_fake_env_get(
            {
                "GOOGLE_CALENDAR_ACCESS_TOKEN": "env_access",
                "GOOGLE_CALENDAR_REFRESH_TOKEN": "env_refresh",
                "GOOGLE_CALENDAR_TOKEN_EXPIRES_AT": "1735200000",
            }
        )
        try:
            result = _prod_load_tokens()
            assert result
            assert result["access_token"] == "env_access"
            assert result["refresh_token"] == "env_refresh"
            assert result["expires_at"] == 1735200000
            assert result["token_type"] == "Bearer"
        finally:
            config_test_hooks.get_env = original_get_env

    def test_partial_env_vars_raises_missing_access_token(self) -> None:
        """Test partial env var config raises error when access_token missing."""
        original_get_env = config_test_hooks.get_env
        config_test_hooks.get_env = _make_fake_env_get(
            {
                "GOOGLE_CALENDAR_REFRESH_TOKEN": "env_refresh",
                "GOOGLE_CALENDAR_TOKEN_EXPIRES_AT": "1735200000",
            }
        )
        try:
            with pytest.raises(AppError) as exc_info:
                _prod_load_tokens()
            error: AppError[CalendarErrorCode] = exc_info.value
            assert error.code == CalendarErrorCode.AUTH_FAILED
            assert "GOOGLE_CALENDAR_ACCESS_TOKEN" in error.message
        finally:
            config_test_hooks.get_env = original_get_env

    def test_partial_env_vars_raises_missing_refresh_token(self) -> None:
        """Test partial env var config raises error when refresh_token missing."""
        original_get_env = config_test_hooks.get_env
        config_test_hooks.get_env = _make_fake_env_get(
            {
                "GOOGLE_CALENDAR_ACCESS_TOKEN": "env_access",
                "GOOGLE_CALENDAR_TOKEN_EXPIRES_AT": "1735200000",
            }
        )
        try:
            with pytest.raises(AppError) as exc_info:
                _prod_load_tokens()
            error: AppError[CalendarErrorCode] = exc_info.value
            assert error.code == CalendarErrorCode.AUTH_FAILED
            assert "GOOGLE_CALENDAR_REFRESH_TOKEN" in error.message
        finally:
            config_test_hooks.get_env = original_get_env

    def test_partial_env_vars_raises_missing_expires_at(self) -> None:
        """Test partial env var config raises error when expires_at missing."""
        original_get_env = config_test_hooks.get_env
        config_test_hooks.get_env = _make_fake_env_get(
            {
                "GOOGLE_CALENDAR_ACCESS_TOKEN": "env_access",
                "GOOGLE_CALENDAR_REFRESH_TOKEN": "env_refresh",
            }
        )
        try:
            with pytest.raises(AppError) as exc_info:
                _prod_load_tokens()
            error: AppError[CalendarErrorCode] = exc_info.value
            assert error.code == CalendarErrorCode.AUTH_FAILED
            assert "GOOGLE_CALENDAR_TOKEN_EXPIRES_AT" in error.message
        finally:
            config_test_hooks.get_env = original_get_env


class TestProdSaveTokens:
    def test_saves_tokens_to_file(self, tmp_path: Path) -> None:
        tokens = OAuthTokens(
            access_token="test_access",
            refresh_token="test_refresh",
            expires_at=1735200000,
            token_type="Bearer",
        )
        tokens_file = tmp_path / "subdir" / "tokens.json"
        _prod_save_tokens(tokens, str(tokens_file))

        assert tokens_file.exists()
        content = tokens_file.read_text(encoding="utf-8")
        assert "test_access" in content
        assert "test_refresh" in content


class TestProdLoadCredentials:
    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(AppError) as exc_info:
            _prod_load_credentials(str(tmp_path / "nonexistent.json"))
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CREDENTIALS_NOT_FOUND
        assert "not found" in error.message

    def test_raises_for_invalid_json(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("not valid json", encoding="utf-8")
        with pytest.raises(AppError) as exc_info:
            _prod_load_credentials(str(creds_file))
        error: AppError[CalendarErrorCode] = exc_info.value
        assert error.code == CalendarErrorCode.CREDENTIALS_NOT_FOUND
        assert "not valid JSON" in error.message

    def test_loads_valid_credentials(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "creds.json"
        creds_file.write_text(
            '{"installed": {"client_id": "my_id", "client_secret": "my_secret", '
            '"redirect_uris": ["http://localhost:8080"]}}',
            encoding="utf-8",
        )
        result = _prod_load_credentials(str(creds_file))
        assert result["client_id"] == "my_id"
        assert result["client_secret"] == "my_secret"
        assert result["redirect_uri"] == "http://localhost:8080"

    def test_handles_empty_redirect_uris(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "creds.json"
        creds_file.write_text(
            '{"installed": {"client_id": "my_id", "client_secret": "my_secret", '
            '"redirect_uris": []}}',
            encoding="utf-8",
        )
        result = _prod_load_credentials(str(creds_file))
        assert result["redirect_uri"] == ""

    def test_loads_from_env_vars(self) -> None:
        """Test loading credentials from environment variables."""
        original_get_env = config_test_hooks.get_env
        config_test_hooks.get_env = _make_fake_env_get(
            {
                "GOOGLE_CALENDAR_CLIENT_ID": "env_client_id",
                "GOOGLE_CALENDAR_CLIENT_SECRET": "env_client_secret",
                "GOOGLE_CALENDAR_REDIRECT_URI": "http://custom:9000",
            }
        )
        try:
            result = _prod_load_credentials()
            assert result["client_id"] == "env_client_id"
            assert result["client_secret"] == "env_client_secret"
            assert result["redirect_uri"] == "http://custom:9000"
        finally:
            config_test_hooks.get_env = original_get_env

    def test_env_vars_default_redirect_uri(self) -> None:
        """Test env var loading uses default redirect URI when not specified."""
        original_get_env = config_test_hooks.get_env
        config_test_hooks.get_env = _make_fake_env_get(
            {
                "GOOGLE_CALENDAR_CLIENT_ID": "env_client_id",
                "GOOGLE_CALENDAR_CLIENT_SECRET": "env_client_secret",
            }
        )
        try:
            result = _prod_load_credentials()
            assert result["client_id"] == "env_client_id"
            assert result["client_secret"] == "env_client_secret"
            assert result["redirect_uri"] == "http://localhost"
        finally:
            config_test_hooks.get_env = original_get_env

    def test_partial_env_vars_raises_missing_client_id(self) -> None:
        """Test partial env var config raises error when client_id missing."""
        original_get_env = config_test_hooks.get_env
        config_test_hooks.get_env = _make_fake_env_get(
            {
                "GOOGLE_CALENDAR_CLIENT_SECRET": "env_secret",
            }
        )
        try:
            with pytest.raises(AppError) as exc_info:
                _prod_load_credentials()
            error: AppError[CalendarErrorCode] = exc_info.value
            assert error.code == CalendarErrorCode.CREDENTIALS_NOT_FOUND
            assert "GOOGLE_CALENDAR_CLIENT_ID" in error.message
        finally:
            config_test_hooks.get_env = original_get_env

    def test_partial_env_vars_raises_missing_client_secret(self) -> None:
        """Test partial env var config raises error when client_secret missing."""
        original_get_env = config_test_hooks.get_env
        config_test_hooks.get_env = _make_fake_env_get(
            {
                "GOOGLE_CALENDAR_CLIENT_ID": "env_id",
            }
        )
        try:
            with pytest.raises(AppError) as exc_info:
                _prod_load_credentials()
            error: AppError[CalendarErrorCode] = exc_info.value
            assert error.code == CalendarErrorCode.CREDENTIALS_NOT_FOUND
            assert "GOOGLE_CALENDAR_CLIENT_SECRET" in error.message
        finally:
            config_test_hooks.get_env = original_get_env


class TestProdHttpHooks:
    def test_http_get_makes_request_with_headers(self) -> None:
        """Test _prod_http_get with a local server and headers."""
        import http.server
        import socketserver
        import threading

        from platform_calendar.testing import _prod_http_get

        received_headers: dict[str, str] = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                # Capture the custom header
                auth = self.headers.get("Authorization")
                if auth:
                    received_headers["Authorization"] = auth
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')

            def log_message(self, format: str, *args: str) -> None:
                pass  # Suppress logging

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.handle_request)
            thread.start()

            result = _prod_http_get(
                f"http://127.0.0.1:{port}/test",
                {"Authorization": "Bearer token123"},
            )
            thread.join()

        assert result == '{"status": "ok"}'
        assert received_headers["Authorization"] == "Bearer token123"

    def test_http_post_makes_request(self) -> None:
        """Test _prod_http_post with a local server."""
        import http.server
        import socketserver
        import threading

        from platform_calendar.testing import _prod_http_post

        received_body: list[bytes] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                received_body.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"received": true}')

            def log_message(self, format: str, *args: str) -> None:
                pass  # Suppress logging

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.handle_request)
            thread.start()

            result = _prod_http_post(
                f"http://127.0.0.1:{port}/test",
                {"Content-Type": "application/json"},
                '{"data": "test"}',
            )
            thread.join()

        assert result == '{"received": true}'
        assert received_body[0] == b'{"data": "test"}'

    def test_http_delete_makes_request(self) -> None:
        """Test _prod_http_delete with a local server."""
        import http.server
        import socketserver
        import threading

        delete_received: list[bool] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_DELETE(self) -> None:
                delete_received.append(True)
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: str) -> None:
                pass  # Suppress logging

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.handle_request)
            thread.start()

            _prod_http_delete(
                f"http://127.0.0.1:{port}/test/event123",
                {"Authorization": "Bearer token123"},
            )
            thread.join()

        assert delete_received == [True]

    def test_http_patch_makes_request(self) -> None:
        """Test _prod_http_patch with a local server."""
        import http.server
        import socketserver
        import threading

        received_body: list[bytes] = []
        received_method: list[str] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_PATCH(self) -> None:
                received_method.append("PATCH")
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                received_body.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"patched": true}')

            def log_message(self, format: str, *args: str) -> None:
                pass  # Suppress logging

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.handle_request)
            thread.start()

            result = _prod_http_patch(
                f"http://127.0.0.1:{port}/test/event123",
                {"Content-Type": "application/json"},
                '{"summary": "Updated Event"}',
            )
            thread.join()

        assert result == '{"patched": true}'
        assert received_method == ["PATCH"]
        assert received_body[0] == b'{"summary": "Updated Event"}'


class TestProdOpenBrowser:
    def test_calls_opener_with_url(self) -> None:
        """Test _prod_open_browser with a fake opener."""
        from platform_calendar.testing import _prod_open_browser

        opened_urls: list[str] = []

        def fake_opener(url: str) -> bool:
            opened_urls.append(url)
            return True

        _prod_open_browser("https://example.com/auth", _opener=fake_opener)

        assert opened_urls == ["https://example.com/auth"]


class TestProdConsoleInput:
    def test_returns_input_from_func(self) -> None:
        """Test _prod_console_input with a fake input function."""
        from platform_calendar.testing import _prod_console_input

        captured_prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "user_entered_code"

        result = _prod_console_input("Enter code: ", _input_func=fake_input)

        assert result == "user_entered_code"
        assert captured_prompts == ["Enter code: "]
