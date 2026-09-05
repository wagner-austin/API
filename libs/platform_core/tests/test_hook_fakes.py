"""The CLI hook fakes two libraries used to keep their own copies of.

These are shipped test utilities, so they are tested like production code
rather than trusted because tests use them. A fake that quietly does the wrong
thing does not fail -- it makes whatever uses it pass.
"""

from __future__ import annotations

import pytest

from platform_core.hook_fakes import (
    make_fake_console,
    make_fake_current_time,
    make_fake_file_system,
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_send,
    make_raising_http_delete,
    make_raising_http_get,
    make_raising_http_send,
)


class TestTheHttpShapes:
    """Three shapes cover the four verbs, and each keeps its own arity."""

    def test_the_get_shape_answers_the_fixed_body(self) -> None:
        hook = make_fake_http_get('{"result": "ok"}')

        assert hook("https://example.com", {"Authorization": "Bearer t"}) == '{"result": "ok"}'

    def test_the_send_shape_takes_a_body_and_answers(self) -> None:
        """POST and PATCH share this signature, which is why there is one
        maker for them rather than the two each library used to have."""
        hook = make_fake_http_send('{"id": "123"}')

        assert hook("https://example.com", {}, '{"name": "x"}') == '{"id": "123"}'

    def test_the_delete_shape_answers_nothing(self) -> None:
        hook = make_fake_http_delete()

        assert hook("https://example.com/1", {"Authorization": "Bearer t"}) is None

    def test_the_body_is_fixed_across_calls(self) -> None:
        """A hook that answered only once would let a test pass on its first
        request and fail on a retry for reasons nothing in the test says."""
        hook = make_fake_http_get("same")

        assert [hook("u", {}), hook("u", {})] == ["same", "same"]


class TestTheRaisingShapes:
    """Putting a transport failure in front of the code under test."""

    def test_the_get_shape_raises(self) -> None:
        hook = make_raising_http_get(ConnectionError("network down"))

        with pytest.raises(ConnectionError, match="network down"):
            hook("https://example.com", {})

    def test_the_send_shape_raises(self) -> None:
        hook = make_raising_http_send(TimeoutError("timed out"))

        with pytest.raises(TimeoutError, match="timed out"):
            hook("https://example.com", {}, "{}")

    def test_the_delete_shape_raises(self) -> None:
        hook = make_raising_http_delete(PermissionError("forbidden"))

        with pytest.raises(PermissionError, match="forbidden"):
            hook("https://example.com/1", {})


class TestTheClock:
    def test_it_reports_the_same_instant(self) -> None:
        hook = make_fake_current_time(1704067200)

        assert [hook(), hook()] == [1704067200, 1704067200]


class TestTheFileSystem:
    def test_it_reads_what_it_was_seeded_with(self) -> None:
        read, _write, _exists = make_fake_file_system({"/a.json": '{"k": 1}'})

        assert read("/a.json") == '{"k": 1}'

    def test_a_missing_path_raises_rather_than_answering_empty(self) -> None:
        read, _write, _exists = make_fake_file_system({})

        with pytest.raises(FileNotFoundError, match="/gone"):
            read("/gone")

    def test_a_write_is_visible_to_a_later_read(self) -> None:
        """The three hooks share one store, which is the only reason to get
        them from a single call rather than separately."""
        read, write, exists = make_fake_file_system({})
        write("/new.txt", "content")

        assert read("/new.txt") == "content"
        assert exists("/new.txt") is True

    def test_exists_answers_false_for_an_unwritten_path(self) -> None:
        _read, _write, exists = make_fake_file_system({"/a": "x"})

        assert exists("/b") is False

    def test_the_seed_mapping_is_copied_not_captured(self) -> None:
        """A caller that reuses its seed dict between tests would otherwise
        see writes from an earlier one."""
        seed = {"/a": "x"}
        _read, write, _exists = make_fake_file_system(seed)
        write("/b", "y")

        assert seed == {"/a": "x"}


class TestTheConsole:
    def test_it_answers_the_scripted_inputs_in_order(self) -> None:
        _output, prompt = make_fake_console(["first", "second"])

        assert [prompt("a: "), prompt("b: ")] == ["first", "second"]

    def test_reading_past_the_script_is_refused(self) -> None:
        """Both copies returned "" forever once exhausted, and both had a test
        pinning that: a test consuming more prompts than it scripted passed
        while the code under test read an answer no person would have given."""
        _output, prompt = make_fake_console(["only"])
        assert prompt("a: ") == "only"

        with pytest.raises(AssertionError, match="scripted with only 1 answer"):
            prompt("b: ")

    def test_the_refusal_names_the_prompt_that_went_unanswered(self) -> None:
        _output, prompt = make_fake_console([])

        with pytest.raises(AssertionError, match="Continue"):
            prompt("Continue")

    def test_output_accepts_what_it_is_given(self) -> None:
        output, _prompt = make_fake_console([])

        assert output("hello") is None


__all__ = [
    "TestTheClock",
    "TestTheConsole",
    "TestTheFileSystem",
    "TestTheHttpShapes",
    "TestTheRaisingShapes",
]
