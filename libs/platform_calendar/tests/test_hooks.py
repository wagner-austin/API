"""Tests for hooks container and fake hook helpers."""

from __future__ import annotations

import pytest

from platform_calendar.fakes import (
    make_fake_calendar,
    make_fake_console,
    make_fake_credentials,
    make_fake_current_time,
    make_fake_event,
    make_fake_file_system,
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_patch,
    make_fake_http_post,
    make_fake_no_tokens,
    make_fake_tokens,
    make_raising_http_delete,
    make_raising_http_get,
    make_raising_http_patch,
    make_raising_http_post,
)
from platform_calendar.testing import (
    hooks,
    reset_hooks,
)
from platform_calendar.types import (
    OAuthCredentials,
    OAuthTokens,
)


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


class TestMakeFakeHttpDelete:
    def test_does_nothing(self) -> None:
        hook = make_fake_http_delete()
        # Should not raise, returns None
        result = hook("https://example.com/delete", {"Authorization": "Bearer token"})
        assert result is None


class TestMakeRaisingHttpDelete:
    def test_raises_exception(self) -> None:
        hook = make_raising_http_delete(ConnectionError("Delete failed"))
        with pytest.raises(ConnectionError, match="Delete failed"):
            hook("https://example.com/delete", {})


class TestMakeFakeHttpPatch:
    def test_returns_fixed_response(self) -> None:
        hook = make_fake_http_patch('{"updated": true}')
        result = hook("https://example.com/event", {}, '{"summary": "Updated"}')
        assert result == '{"updated": true}'


class TestMakeRaisingHttpPatch:
    def test_raises_exception(self) -> None:
        hook = make_raising_http_patch(ConnectionError("Patch failed"))
        with pytest.raises(ConnectionError, match="Patch failed"):
            hook("https://example.com/event", {}, '{"summary": "Test"}')


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
