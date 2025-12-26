"""Tests for platform_calendar.testing module."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError, CalendarErrorCode

from platform_calendar.testing import (
    FakeCalendarClient,
    _prod_console_output,
    _prod_current_time,
    _prod_file_exists,
    _prod_load_credentials,
    _prod_load_tokens,
    _prod_read_file,
    _prod_save_tokens,
    _prod_write_file,
    hooks,
    make_fake_calendar,
    make_fake_console,
    make_fake_credentials,
    make_fake_current_time,
    make_fake_event,
    make_fake_file_system,
    make_fake_http_get,
    make_fake_http_post,
    make_fake_no_tokens,
    make_fake_tokens,
    make_raising_http_get,
    make_raising_http_post,
    reset_hooks,
)
from platform_calendar.types import (
    EventDateTime,
    OAuthCredentials,
    OAuthTokens,
)


class TestFakeCalendarClient:
    def test_implements_protocol(self) -> None:
        client = FakeCalendarClient()
        # Verify protocol methods exist by checking they are callable
        assert callable(client.list_calendars)
        assert callable(client.get_events)
        assert callable(client.create_event)
        assert callable(client.update_event)
        assert callable(client.delete_event)

    def test_add_calendar(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")
        calendars = client.list_calendars()
        assert len(calendars) == 1
        assert calendars[0]["id"] == "primary"
        assert calendars[0]["summary"] == "My Calendar"

    def test_add_calendar_with_options(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(
            calendar_id="work",
            summary="Work",
            description="Work calendar",
            primary=False,
            time_zone="America/New_York",
        )
        calendars = client.list_calendars()
        assert calendars[0]["description"] == "Work calendar"
        assert calendars[0]["primary"] is False
        assert calendars[0]["timeZone"] == "America/New_York"

    def test_create_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        event = client.create_event(
            calendar_id="primary",
            summary="Test Event",
            description="Test description",
            start=start,
            end=end,
            reminders=(60, 1440),
        )

        assert event["id"].startswith("fake_event_")
        assert event["summary"] == "Test Event"
        assert event["status"] == "confirmed"
        assert len(event["reminders"]["overrides"]) == 2

    def test_get_created_events(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        client.create_event(
            calendar_id="primary",
            summary="Event 1",
            description="",
            start=start,
            end=end,
            reminders=(),
        )
        client.create_event(
            calendar_id="primary",
            summary="Event 2",
            description="",
            start=start,
            end=end,
            reminders=(),
        )

        created = client.get_created_events()
        assert len(created) == 2
        assert created[0]["summary"] == "Event 1"
        assert created[1]["summary"] == "Event 2"

    def test_get_events(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        client.create_event(
            calendar_id="primary",
            summary="Test",
            description="",
            start=start,
            end=end,
            reminders=(),
        )

        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-26T00:00:00Z",
            time_max="2025-12-27T00:00:00Z",
        )
        assert len(events) == 1

    def test_get_events_empty_calendar(self) -> None:
        client = FakeCalendarClient()
        events = client.get_events(
            calendar_id="nonexistent",
            time_min="2025-12-26T00:00:00Z",
            time_max="2025-12-27T00:00:00Z",
        )
        assert len(events) == 0

    def test_update_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="Original",
            description="Original desc",
            start=start,
            end=end,
            reminders=(),
        )

        updated = client.update_event(
            calendar_id="primary",
            event_id=created["id"],
            summary="Updated",
            description="Updated desc",
        )

        assert updated["summary"] == "Updated"
        assert updated["description"] == "Updated desc"

    def test_update_event_partial(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="Original",
            description="Original desc",
            start=start,
            end=end,
            reminders=(),
        )

        updated = client.update_event(
            calendar_id="primary",
            event_id=created["id"],
            summary="Updated",
        )

        assert updated["summary"] == "Updated"
        assert updated["description"] == "Original desc"

    def test_update_event_not_found(self) -> None:
        client = FakeCalendarClient()
        updated = client.update_event(
            calendar_id="primary",
            event_id="nonexistent",
            summary="Test",
        )
        # Returns placeholder when not found
        assert updated["id"] == "nonexistent"

    def test_get_updated_events(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="Original",
            description="",
            start=start,
            end=end,
            reminders=(),
        )

        client.update_event(
            calendar_id="primary",
            event_id=created["id"],
            summary="Updated",
        )

        updated = client.get_updated_events()
        assert len(updated) == 1
        assert updated[0]["summary"] == "Updated"

    def test_delete_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        created = client.create_event(
            calendar_id="primary",
            summary="To Delete",
            description="",
            start=start,
            end=end,
            reminders=(),
        )

        client.delete_event(calendar_id="primary", event_id=created["id"])

        deleted = client.get_deleted_events()
        assert len(deleted) == 1
        assert deleted[0] == ("primary", created["id"])

        events = client.get_events(
            calendar_id="primary",
            time_min="2025-12-26T00:00:00Z",
            time_max="2025-12-27T00:00:00Z",
        )
        assert len(events) == 0

    def test_add_event(self) -> None:
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        event = make_fake_event(event_id="existing")
        client.add_event(calendar_id="primary", event=event)

        events = client.get_events(
            calendar_id="primary",
            time_min="2025-01-01T00:00:00Z",
            time_max="2025-12-31T00:00:00Z",
        )
        assert len(events) == 1
        assert events[0]["id"] == "existing"


class TestHooksContainer:
    def test_hooks_are_set(self) -> None:
        # After reset_hooks, all hooks should be production implementations
        reset_hooks()
        assert callable(hooks.http_get)
        assert callable(hooks.http_post)
        assert callable(hooks.load_tokens)
        assert callable(hooks.save_tokens)
        assert callable(hooks.load_credentials)
        assert callable(hooks.open_browser)
        assert callable(hooks.current_time)
        assert callable(hooks.read_file)
        assert callable(hooks.write_file)
        assert callable(hooks.file_exists)


class TestMakeFakeHttpGet:
    def test_returns_fixed_response(self) -> None:
        hook = make_fake_http_get('{"result": "ok"}')
        result = hook("https://example.com", {"Authorization": "Bearer token"})
        assert result == '{"result": "ok"}'


class TestMakeFakeHttpPost:
    def test_returns_fixed_response(self) -> None:
        hook = make_fake_http_post('{"id": "123"}')
        result = hook("https://example.com", {}, '{"data": "test"}')
        assert result == '{"id": "123"}'


class TestMakeRaisingHttpGet:
    def test_raises_exception(self) -> None:
        hook = make_raising_http_get(ConnectionError("Network error"))
        with pytest.raises(ConnectionError, match="Network error"):
            hook("https://example.com", {})


class TestMakeRaisingHttpPost:
    def test_raises_exception(self) -> None:
        hook = make_raising_http_post(TimeoutError("Timeout"))
        with pytest.raises(TimeoutError, match="Timeout"):
            hook("https://example.com", {}, "")


class TestMakeFakeTokens:
    def test_returns_tokens(self) -> None:
        tokens = OAuthTokens(
            access_token="access",
            refresh_token="refresh",
            expires_at=1735200000,
            token_type="Bearer",
        )
        hook = make_fake_tokens(tokens)
        result = hook()
        # Verify we get the tokens back with correct values
        assert result == tokens
        assert result["access_token"] == "access"


class TestMakeFakeNoTokens:
    def test_returns_none(self) -> None:
        hook = make_fake_no_tokens()
        assert hook() is None


class TestMakeFakeCredentials:
    def test_returns_credentials(self) -> None:
        creds = OAuthCredentials(
            client_id="id",
            client_secret="secret",
            redirect_uri="http://localhost",
        )
        hook = make_fake_credentials(creds)
        result = hook()
        assert result["client_id"] == "id"


class TestMakeFakeCurrentTime:
    def test_returns_timestamp(self) -> None:
        hook = make_fake_current_time(1735200000)
        assert hook() == 1735200000


class TestMakeFakeFileSystem:
    def test_read_existing_file(self) -> None:
        read_hook, _write, _exists = make_fake_file_system(
            {
                "/path/to/file.json": '{"key": "value"}',
            }
        )
        content = read_hook("/path/to/file.json")
        assert content == '{"key": "value"}'

    def test_read_nonexistent_file(self) -> None:
        read_hook, _write, _exists = make_fake_file_system({})
        with pytest.raises(FileNotFoundError):
            read_hook("/nonexistent")

    def test_write_file(self) -> None:
        read_hook, write_hook, _exists = make_fake_file_system({})
        write_hook("/new/file.txt", "content")
        assert read_hook("/new/file.txt") == "content"

    def test_file_exists(self) -> None:
        _read, _write, exists_hook = make_fake_file_system(
            {
                "/exists.txt": "content",
            }
        )
        assert exists_hook("/exists.txt") is True
        assert exists_hook("/not_exists.txt") is False


class TestMakeFakeEvent:
    def test_default_values(self) -> None:
        event = make_fake_event()
        assert event["id"] == "test_event_1"
        assert event["summary"] == "Test Event"
        assert event["status"] == "confirmed"

    def test_custom_values(self) -> None:
        event = make_fake_event(
            event_id="custom123",
            summary="Custom Event",
            description="Custom desc",
            status="tentative",
        )
        assert event["id"] == "custom123"
        assert event["summary"] == "Custom Event"
        assert event["status"] == "tentative"

    def test_cancelled_status(self) -> None:
        event = make_fake_event(status="cancelled")
        assert event["status"] == "cancelled"


class TestMakeFakeCalendar:
    def test_default_values(self) -> None:
        cal = make_fake_calendar()
        assert cal["id"] == "primary"
        assert cal["summary"] == "Primary Calendar"
        assert cal["primary"] is True

    def test_custom_values(self) -> None:
        cal = make_fake_calendar(
            calendar_id="work",
            summary="Work Calendar",
            description="Work stuff",
            primary=False,
            time_zone="America/New_York",
        )
        assert cal["id"] == "work"
        assert cal["summary"] == "Work Calendar"
        assert cal["primary"] is False
        assert cal["timeZone"] == "America/New_York"


# =============================================================================
# Production Hook Tests
# =============================================================================


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


class TestMakeFakeConsole:
    def test_captures_output(self) -> None:
        output_hook, _input_hook = make_fake_console([])
        output_hook("Hello")
        output_hook("World")
        # Just verify it doesn't raise

    def test_returns_inputs_in_order(self) -> None:
        _output_hook, input_hook = make_fake_console(["first", "second", "third"])
        assert input_hook("prompt1: ") == "first"
        assert input_hook("prompt2: ") == "second"
        assert input_hook("prompt3: ") == "third"

    def test_returns_empty_when_exhausted(self) -> None:
        _output_hook, input_hook = make_fake_console(["only one"])
        assert input_hook("prompt: ") == "only one"
        assert input_hook("prompt: ") == ""
        assert input_hook("prompt: ") == ""


class TestFakeCalendarClientBranches:
    def test_add_event_to_new_calendar(self) -> None:
        """Test add_event when calendar_id is not yet in _events dict."""
        client = FakeCalendarClient()
        # Don't add the calendar first - just add an event directly
        event = make_fake_event(event_id="test123")
        client.add_event(calendar_id="new_calendar", event=event)

        events = client.get_events(
            calendar_id="new_calendar",
            time_min="2025-01-01T00:00:00Z",
            time_max="2025-12-31T00:00:00Z",
        )
        assert len(events) == 1
        assert events[0]["id"] == "test123"

    def test_create_event_on_new_calendar(self) -> None:
        """Test create_event when calendar_id is not yet in _events dict."""
        client = FakeCalendarClient()
        # Don't add the calendar first - create an event directly
        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        event = client.create_event(
            calendar_id="new_calendar",
            summary="Test Event",
            description="Test",
            start=start,
            end=end,
            reminders=(),
        )

        assert event["id"].startswith("fake_event_")
        events = client.get_events(
            calendar_id="new_calendar",
            time_min="2025-01-01T00:00:00Z",
            time_max="2025-12-31T00:00:00Z",
        )
        assert len(events) == 1

    def test_update_event_iterates_through_multiple(self) -> None:
        """Test update_event when target is not the first event (covers loop branch)."""
        client = FakeCalendarClient()
        client.add_calendar(calendar_id="primary", summary="My Calendar")

        start = EventDateTime(dateTime="2025-12-26T14:00:00Z", timeZone="UTC")
        end = EventDateTime(dateTime="2025-12-26T15:00:00Z", timeZone="UTC")

        # Create multiple events
        client.create_event(
            calendar_id="primary",
            summary="Event 1",
            description="First",
            start=start,
            end=end,
            reminders=(),
        )
        second_event = client.create_event(
            calendar_id="primary",
            summary="Event 2",
            description="Second",
            start=start,
            end=end,
            reminders=(),
        )
        client.create_event(
            calendar_id="primary",
            summary="Event 3",
            description="Third",
            start=start,
            end=end,
            reminders=(),
        )

        # Update the second event (requires iterating past the first)
        updated = client.update_event(
            calendar_id="primary",
            event_id=second_event["id"],
            summary="Updated Event 2",
        )

        assert updated["id"] == second_event["id"]
        assert updated["summary"] == "Updated Event 2"


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
