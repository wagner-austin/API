#!/usr/bin/env python
"""Email CLI - Check and send emails via Outlook."""

from __future__ import annotations

import argparse
import sys
from typing import TypedDict

# =============================================================================
# Styles (for rich console output)
# =============================================================================
# =============================================================================
# Token Types
# =============================================================================
# Use "common" tenant for multi-tenant support
from platform_core.cli_args import (
    namespace_bool,
    namespace_int,
    namespace_str,
    namespace_str_tuple,
    run_subcommand_cli,
)

from platform_email.cli_auth import STYLE_ERROR, _print
from platform_email.cli_commands import (
    cmd_auth,
    cmd_folders,
    cmd_list,
    cmd_read,
    cmd_search,
    cmd_send,
)


class ListArgs(TypedDict):
    """Arguments for list command."""

    folder: str
    count: int


class ReadArgs(TypedDict):
    """Arguments for read command."""

    index: int


class SendArgs(TypedDict):
    """Arguments for send command."""

    to: str
    subject: str
    body_file: str
    cc: str
    bcc: str
    html: bool
    attachments: tuple[str, ...]


class SearchArgs(TypedDict):
    """Arguments for search command."""

    query: str
    count: int


def decode_list_args(args: argparse.Namespace) -> ListArgs:
    """Decode list arguments."""
    return ListArgs(
        folder=namespace_str(args, "folder", "inbox"),
        count=namespace_int(args, "count", 10),
    )


def decode_read_args(args: argparse.Namespace) -> ReadArgs:
    """Decode read arguments."""
    return ReadArgs(index=namespace_int(args, "index", 1))


def decode_send_args(args: argparse.Namespace) -> SendArgs:
    """Decode send arguments.

    Args:
        args: Parsed argparse namespace.

    Returns:
        SendArgs with to, subject, body_file, cc, bcc, and html fields.
    """
    return SendArgs(
        to=namespace_str(args, "to", ""),
        subject=namespace_str(args, "subject", ""),
        body_file=namespace_str(args, "body_file", ""),
        cc=namespace_str(args, "cc", ""),
        bcc=namespace_str(args, "bcc", ""),
        html=namespace_bool(args, "html", False),
        attachments=namespace_str_tuple(args, "attach"),
    )


def decode_search_args(args: argparse.Namespace) -> SearchArgs:
    """Decode search arguments.

    Args:
        args: Parsed argparse namespace.

    Returns:
        SearchArgs with query and count fields.
    """
    return SearchArgs(
        query=namespace_str(args, "query", ""),
        count=namespace_int(args, "count", 10),
    )


# =============================================================================
# Main
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="Email CLI for Outlook")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # auth
    subparsers.add_parser("auth", help="Authorize with Microsoft")

    # folders
    subparsers.add_parser("folders", help="List email folders")

    # list
    list_parser = subparsers.add_parser("list", aliases=["ls"], help="List recent emails")
    list_parser.add_argument("-f", "--folder", default="inbox", help="Folder (inbox, sent, drafts)")
    list_parser.add_argument("-n", "--count", type=int, default=10, help="Number of emails")

    # read
    read_parser = subparsers.add_parser("read", help="Read an email by index")
    read_parser.add_argument("index", type=int, help="Email index from list")

    # send
    send_parser = subparsers.add_parser("send", help="Send an email")
    send_parser.add_argument("to", help="Recipient email")
    send_parser.add_argument("subject", help="Email subject")
    send_parser.add_argument("body_file", help="Path to file containing email body")
    send_parser.add_argument("--cc", default="", help="Comma-separated CC recipients")
    send_parser.add_argument("--bcc", default="", help="Comma-separated BCC recipients")
    send_parser.add_argument(
        "--html",
        action="store_true",
        default=False,
        help="Send as HTML with <pre> formatting to preserve whitespace",
    )
    send_parser.add_argument(
        "--attach",
        action="append",
        default=None,
        help="File to attach (can be repeated for multiple files)",
    )

    # search
    search_parser = subparsers.add_parser("search", help="Search emails")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--count", type=int, default=10, help="Max results")

    return parser


def _dispatch_command(command_str: str, args: argparse.Namespace) -> None:
    """Dispatch command to handler.

    Args:
        command_str: Command name.
        args: Parsed arguments.
    """
    if command_str == "auth":
        cmd_auth()
    elif command_str == "folders":
        cmd_folders()
    elif command_str in ("list", "ls"):
        list_args = decode_list_args(args)
        cmd_list(list_args["folder"], list_args["count"])
    elif command_str == "read":
        read_args = decode_read_args(args)
        cmd_read(read_args["index"])
    elif command_str == "send":
        send_args = decode_send_args(args)
        if not send_args["to"] or not send_args["subject"]:
            _print(f"[{STYLE_ERROR}]Missing required arguments: to and subject are required[/]")
            return
        if not send_args["body_file"]:
            _print(f"[{STYLE_ERROR}]Missing required argument: body_file[/]")
            return
        cmd_send(
            send_args["to"],
            send_args["subject"],
            send_args["body_file"],
            cc=send_args["cc"],
            bcc=send_args["bcc"],
            html=send_args["html"],
            attachments=send_args["attachments"],
        )
    elif command_str == "search":
        search_args = decode_search_args(args)
        if not search_args["query"]:
            _print(f"[{STYLE_ERROR}]Missing required argument: query[/]")
            return
        cmd_search(search_args["query"], search_args["count"])
    else:
        # Default: show inbox
        cmd_list("inbox", 10)


def main() -> None:
    """Main entry point."""
    run_subcommand_cli(sys.argv[1:], build_parser=_build_parser, dispatch=_dispatch_command)


if __name__ == "__main__":
    main()
