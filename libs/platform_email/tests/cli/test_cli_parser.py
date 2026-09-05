"""Email CLI: parser construction, dispatch, main."""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Generator
from datetime import datetime

import pytest
from platform_core.cli_args import namespace_int, namespace_str

from platform_email import cli
from platform_email.testing import hooks, reset_hooks


@pytest.fixture(autouse=True)
def _reset_hooks_after_test() -> Generator[None, None, None]:
    """Reset hooks after each test."""
    yield
    reset_hooks()


# =============================================================================
# Token Types
# =============================================================================


class TestBuildParser:
    """Tests for _build_parser function."""

    def test_returns_parser_with_subparsers(self) -> None:
        """Test returns a parser with subparsers."""
        parser = cli._build_parser()
        # Verify it can parse known commands
        args = parser.parse_args(["auth"])
        command = namespace_str(args, "command", "")
        assert command == "auth"

    def test_parses_auth_command(self) -> None:
        """Test parses auth command."""
        parser = cli._build_parser()
        args = parser.parse_args(["auth"])
        command = namespace_str(args, "command", "")
        assert command == "auth"

    def test_parses_list_command_with_options(self) -> None:
        """Test parses list command with options."""
        parser = cli._build_parser()
        args = parser.parse_args(["list", "-f", "sent", "-n", "20"])
        command = namespace_str(args, "command", "")
        folder = namespace_str(args, "folder", "")
        count = namespace_int(args, "count", 0)
        assert command == "list"
        assert folder == "sent"
        assert count == 20

    def test_parses_send_command_with_body_file(self) -> None:
        """Test parses send command with body_file positional arg."""
        parser = cli._build_parser()
        args = parser.parse_args(["send", "to@test.com", "Subject", "/path/to/body.txt"])
        command = namespace_str(args, "command", "")
        send_args = cli.decode_send_args(args)
        assert command == "send"
        assert send_args["to"] == "to@test.com"
        assert send_args["subject"] == "Subject"
        assert send_args["body_file"] == "/path/to/body.txt"
        assert send_args["cc"] == ""
        assert send_args["bcc"] == ""

    def test_parses_send_command_with_cc_and_bcc(self) -> None:
        """Test parses send command with --cc and --bcc flags."""
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "send",
                "to@test.com",
                "Subject",
                "/body.txt",
                "--cc",
                "a@b.com,c@d.com",
                "--bcc",
                "secret@x.com",
            ]
        )
        send_args = cli.decode_send_args(args)
        assert send_args["cc"] == "a@b.com,c@d.com"
        assert send_args["bcc"] == "secret@x.com"

    def test_parses_send_command_with_html_flag(self) -> None:
        """Test parses send command with --html flag."""
        parser = cli._build_parser()
        args = parser.parse_args(["send", "to@test.com", "Subject", "/path.txt", "--html"])
        send_args = cli.decode_send_args(args)
        assert send_args["to"] == "to@test.com"
        assert send_args["body_file"] == "/path.txt"
        assert send_args["html"] is True

    def test_parses_send_command_with_attachments(self) -> None:
        """Test parses send command with --attach flags."""
        parser = cli._build_parser()
        args = parser.parse_args(
            [
                "send",
                "to@test.com",
                "Subject",
                "/body.txt",
                "--attach",
                "/path/doc.pdf",
                "--attach",
                "/path/img.png",
            ]
        )
        send_args = cli.decode_send_args(args)
        assert send_args["attachments"] == ("/path/doc.pdf", "/path/img.png")

    def test_parses_send_command_no_attachments_default(self) -> None:
        """Test send command defaults to no attachments."""
        parser = cli._build_parser()
        args = parser.parse_args(["send", "to@test.com", "Subject", "/body.txt"])
        send_args = cli.decode_send_args(args)
        assert send_args["attachments"] == ()

    def test_parses_search_command(self) -> None:
        """Test parses search command with query and count."""
        parser = cli._build_parser()
        args = parser.parse_args(["search", "turkic workshop", "-n", "20"])
        command = namespace_str(args, "command", "")
        search_args = cli.decode_search_args(args)
        assert command == "search"
        assert search_args["query"] == "turkic workshop"
        assert search_args["count"] == 20

    def test_parses_search_command_defaults(self) -> None:
        """Test parses search command with default count."""
        parser = cli._build_parser()
        args = parser.parse_args(["search", "TU+11"])
        search_args = cli.decode_search_args(args)
        assert search_args["query"] == "TU+11"
        assert search_args["count"] == 10


class TestDispatchCommand:
    """Tests for _dispatch_command function."""

    def test_dispatches_auth(self) -> None:
        """Test dispatches auth command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace()
        cli._dispatch_command("auth", ns)
        output = " ".join(messages)
        assert "Missing credentials" in output

    def test_dispatches_folders(self) -> None:
        """Test dispatches folders command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace()
        cli._dispatch_command("folders", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_list(self) -> None:
        """Test dispatches list command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(folder="inbox", count=10)
        cli._dispatch_command("list", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_ls_alias(self) -> None:
        """Test dispatches ls alias for list."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(folder="inbox", count=10)
        cli._dispatch_command("ls", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_read(self) -> None:
        """Test dispatches read command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(index=1)
        cli._dispatch_command("read", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_send(self) -> None:
        """Test dispatches send command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_send_missing_to(self) -> None:
        """Test dispatches send with missing to field."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(
            to="",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Missing required arguments" in output

    def test_dispatches_send_missing_body_file(self) -> None:
        """Test dispatches send with empty body_file."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Missing required argument: body_file" in output

    def test_dispatches_send_with_body_file(self) -> None:
        """Test dispatches send reading body from file."""
        messages: list[str] = []
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Multi-line\nEmail body\nFrom file"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/path/to/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Email sent" in output
        assert len(post_calls) == 1
        assert "Multi-line" in post_calls[0][2]
        assert "From file" in post_calls[0][2]

    def test_dispatches_send_with_cc_and_bcc(self) -> None:
        """Test dispatches send passes cc and bcc through to cmd_send."""
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: None
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="cc@test.com",
            bcc="bcc@test.com",
            html=False,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        assert len(post_calls) == 1
        assert "cc@test.com" in post_calls[0][2]
        assert "bcc@test.com" in post_calls[0][2]

    def test_dispatches_send_with_html_flag(self) -> None:
        """Test dispatches send with --html wraps body in pre tags."""
        messages: list[str] = []
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }
        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: messages.append(m)
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Line1\nLine2"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=True,
            attach=None,
        )
        cli._dispatch_command("send", ns)
        output = " ".join(messages)
        assert "Email sent" in output
        assert len(post_calls) == 1
        assert '"contentType":"HTML"' in post_calls[0][2]
        assert "<pre" in post_calls[0][2]
        assert "Line1" in post_calls[0][2]

    def test_dispatches_send_with_attachments(self) -> None:
        """Test dispatches send passes attachments through to cmd_send."""
        post_calls: list[tuple[str, dict[str, str], str]] = []
        env = {
            "OUTLOOK_ACCESS_TOKEN": "token",
            "OUTLOOK_TOKEN_EXPIRES_AT": "9999999999",
        }

        hooks.cli_get_env = lambda k: env.get(k)
        hooks.cli_get_now = lambda: datetime.fromtimestamp(1735689600)
        hooks.console_output = lambda m: None
        hooks.file_exists = lambda p: True
        hooks.read_file = lambda p: "Body"
        hooks.read_file_bytes = lambda p: b"binary content"

        def fake_post(url: str, headers: dict[str, str], body: str) -> str:
            post_calls.append((url, headers, body))
            return ""

        hooks.http_post = fake_post

        attach_list: list[str] = ["/path/doc.pdf"]
        ns = argparse.Namespace(
            to="to@test.com",
            subject="Subject",
            body_file="/body.txt",
            cc="",
            bcc="",
            html=False,
            attach=attach_list,
        )
        cli._dispatch_command("send", ns)
        assert len(post_calls) == 1
        assert "doc.pdf" in post_calls[0][2]
        assert "#microsoft.graph.fileAttachment" in post_calls[0][2]

    def test_dispatches_search(self) -> None:
        """Test dispatches search command."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(query="turkic", count=10)
        cli._dispatch_command("search", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output

    def test_dispatches_search_missing_query(self) -> None:
        """Test dispatches search with empty query shows error."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace(query="", count=10)
        cli._dispatch_command("search", ns)
        output = " ".join(messages)
        assert "Missing required argument: query" in output

    def test_dispatches_default(self) -> None:
        """Test dispatches default (list inbox)."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        ns = argparse.Namespace()
        cli._dispatch_command("", ns)
        output = " ".join(messages)
        assert "Not authenticated" in output


class TestMain:
    """The console-script entry point, called for real.

    What stood here reimplemented main's body in the test and asserted on the
    reimplementation -- its own docstring said so -- while a coverage
    exclusion for ``def main() -> None:`` kept the package at 100% without
    the entry point ever running. Both are gone.
    """

    @staticmethod
    def _run(*tokens: str) -> list[str]:
        """Run main with the given command line, returning what it printed.

        Args:
            *tokens: Arguments as a shell would pass them, without the
                program name.

        Returns:
            Every line the CLI wrote to the console.
        """
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        original = sys.argv
        sys.argv = ["email", *tokens]
        try:
            cli.main()
        finally:
            sys.argv = original
        return messages

    def test_a_named_subcommand_runs(self) -> None:
        assert "Not authenticated" in " ".join(self._run("folders"))

    def test_no_subcommand_runs_the_default_view(self) -> None:
        """argparse leaves `command` None here, and the dispatcher's final
        branch turns that into the inbox listing."""
        assert "Not authenticated" in " ".join(self._run())

    def test_a_subcommand_flag_reaches_the_command(self) -> None:
        """`-n` is parsed by the subparser, so this proves main hands the
        whole namespace on rather than only the command name."""
        assert "Not authenticated" in " ".join(self._run("list", "-n", "3"))

    def test_a_mistyped_subcommand_exits_rather_than_running_the_default(self) -> None:
        """Falling through to the dispatcher's final branch would list the
        inbox for someone who asked for something else entirely."""
        with pytest.raises(SystemExit):
            self._run("foldres")

    def test_running_the_module_directly_reaches_main(self) -> None:
        """`python -m platform_email.cli` is a supported invocation and the
        guard requires the __main__ block, so the block itself is exercised
        rather than excluded from coverage."""
        messages: list[str] = []
        hooks.cli_get_env = lambda k: None
        hooks.console_output = lambda m: messages.append(m)

        original = list(sys.argv)
        sys.argv[:] = ["email", "folders"]
        sys.modules.pop("platform_email.cli", None)
        try:
            runpy.run_module("platform_email.cli", run_name="__main__")
        finally:
            sys.argv[:] = original

        assert "Not authenticated" in " ".join(messages)
