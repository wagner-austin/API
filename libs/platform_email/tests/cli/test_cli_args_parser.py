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
