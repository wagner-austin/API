"""Email CLI: arg extraction, decoding, parser, dispatch, main."""

from __future__ import annotations

import argparse
from collections.abc import Generator

import pytest

from platform_email import cli
from platform_email.testing import reset_hooks


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


# =============================================================================
# Token Types
# =============================================================================


class TestExtractStr:
    """Tests for _extract_str function."""

    def test_extracts_string(self) -> None:
        """Test extracts string value."""
        ns = argparse.Namespace(key="value")
        result = cli._extract_str(ns, "key", "default")
        assert result == "value"

    def test_returns_default_for_missing(self) -> None:
        """Test returns default for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_str(ns, "missing", "default")
        assert result == "default"

    def test_returns_default_for_non_string(self) -> None:
        """Test returns default for non-string value."""
        ns = argparse.Namespace(key=123)
        result = cli._extract_str(ns, "key", "default")
        assert result == "default"


class TestExtractInt:
    """Tests for _extract_int function."""

    def test_extracts_int(self) -> None:
        """Test extracts int value."""
        ns = argparse.Namespace(key=42)
        result = cli._extract_int(ns, "key", 0)
        assert result == 42

    def test_returns_default_for_missing(self) -> None:
        """Test returns default for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_int(ns, "missing", 10)
        assert result == 10


class TestDecodeListArgs:
    """Tests for decode_list_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes list arguments."""
        ns = argparse.Namespace(folder="sent", count=20)
        result = cli.decode_list_args(ns)
        assert result["folder"] == "sent"
        assert result["count"] == 20


class TestDecodeReadArgs:
    """Tests for decode_read_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes read arguments."""
        ns = argparse.Namespace(index=5)
        result = cli.decode_read_args(ns)
        assert result["index"] == 5


class TestExtractOptionalStr:
    """Tests for _extract_optional_str function."""

    def test_extracts_string(self) -> None:
        """Test extracts string value when present."""
        ns = argparse.Namespace(key="value")
        result = cli._extract_optional_str(ns, "key")
        assert result == "value"

    def test_returns_none_for_missing(self) -> None:
        """Test returns None for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_optional_str(ns, "missing")
        assert result is None

    def test_returns_none_for_none_value(self) -> None:
        """Test returns None when value is None."""
        ns = argparse.Namespace(key=None)
        result = cli._extract_optional_str(ns, "key")
        assert result is None

    def test_returns_none_for_non_string(self) -> None:
        """Test returns None for non-string value."""
        ns = argparse.Namespace(key=123)
        result = cli._extract_optional_str(ns, "key")
        assert result is None


class TestExtractBool:
    """Tests for _extract_bool function."""

    def test_extracts_true(self) -> None:
        """Test extracts True value."""
        ns = argparse.Namespace(key=True)
        result = cli._extract_bool(ns, "key", False)
        assert result is True

    def test_extracts_false(self) -> None:
        """Test extracts False value."""
        ns = argparse.Namespace(key=False)
        result = cli._extract_bool(ns, "key", True)
        assert result is False

    def test_returns_default_for_missing(self) -> None:
        """Test returns default for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_bool(ns, "missing", True)
        assert result is True

    def test_returns_default_for_non_bool(self) -> None:
        """Test returns default for non-bool value."""
        ns = argparse.Namespace(key="not_a_bool")
        result = cli._extract_bool(ns, "key", False)
        assert result is False


class TestExtractStrTuple:
    """Tests for _extract_str_tuple function."""

    def test_extracts_list_of_strings(self) -> None:
        """Test extracts list of strings as tuple."""
        attach_list: list[str] = ["file1.pdf", "file2.zip"]
        ns = argparse.Namespace(attach=attach_list)
        result = cli._extract_str_tuple(ns, "attach")
        assert result == ("file1.pdf", "file2.zip")

    def test_returns_empty_tuple_for_none(self) -> None:
        """Test returns empty tuple when value is None."""
        ns = argparse.Namespace(attach=None)
        result = cli._extract_str_tuple(ns, "attach")
        assert result == ()

    def test_returns_empty_tuple_for_missing(self) -> None:
        """Test returns empty tuple for missing key."""
        ns = argparse.Namespace()
        result = cli._extract_str_tuple(ns, "attach")
        assert result == ()

    def test_filters_non_string_entries(self) -> None:
        """Test filters out non-string entries from list."""
        mixed_list: list[str | int] = ["file.pdf", 123, "other.txt"]
        ns = argparse.Namespace(attach=mixed_list)
        result = cli._extract_str_tuple(ns, "attach")
        assert result == ("file.pdf", "other.txt")


class TestDecodeSendArgs:
    """Tests for decode_send_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes send arguments."""
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        result = cli.decode_send_args(ns)
        assert result["to"] == "to@test.com"
        assert result["subject"] == "Subject"
        assert result["body_file"] == "/body.txt"
        assert result["cc"] == ""
        assert result["bcc"] == ""
        assert result["html"] is False
        assert result["attachments"] == ()

    def test_decodes_args_with_cc_and_bcc(self) -> None:
        """Test decodes send arguments with cc and bcc."""
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="cc@test.com",
            bcc="bcc@test.com",
            html=False,
            attach=None,
        )
        result = cli.decode_send_args(ns)
        assert result["cc"] == "cc@test.com"
        assert result["bcc"] == "bcc@test.com"

    def test_decodes_args_with_html(self) -> None:
        """Test decodes send arguments with html flag."""
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=True,
            attach=None,
        )
        result = cli.decode_send_args(ns)
        assert result["html"] is True

    def test_decodes_args_with_attachments(self) -> None:
        """Test decodes send arguments with attachment list."""
        attach_list: list[str] = ["/path/doc.pdf", "/path/img.png"]
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=attach_list,
        )
        result = cli.decode_send_args(ns)
        assert result["attachments"] == ("/path/doc.pdf", "/path/img.png")


class TestDecodeSearchArgs:
    """Tests for decode_search_args function."""

    def test_decodes_args(self) -> None:
        """Test decodes search arguments."""
        ns = argparse.Namespace(query="turkic", count=20)
        result = cli.decode_search_args(ns)
        assert result["query"] == "turkic"
        assert result["count"] == 20

    def test_defaults(self) -> None:
        """Test decode uses defaults for missing fields."""
        ns = argparse.Namespace()
        result = cli.decode_search_args(ns)
        assert result["query"] == ""
        assert result["count"] == 10


# =============================================================================
# Main Entry Point
# =============================================================================
