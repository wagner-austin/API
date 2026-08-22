#!/usr/bin/env python
"""Email CLI - Check and send emails via Outlook."""

from __future__ import annotations

import argparse
from typing import TypedDict

# =============================================================================
# Styles (for rich console output)
# =============================================================================
# =============================================================================
# Token Types
# =============================================================================
# Use "common" tenant for multi-tenant support
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


def _extract_str(ns: argparse.Namespace, key: str, default: str) -> str:
    """Extract string attribute from namespace.

    Args:
        ns: Namespace to extract from.
        key: Attribute name.
        default: Default value if not found or wrong type.

    Returns:
        String value or default.
    """
    val: str | int | bool | None = getattr(ns, key, default)
    return val if isinstance(val, str) else default


def _extract_int(ns: argparse.Namespace, key: str, default: int) -> int:
    """Extract int attribute from namespace.

    Args:
        ns: Namespace to extract from.
        key: Attribute name.
        default: Default value if not found or wrong type.

    Returns:
        Int value or default.
    """
    val: str | int | bool | None = getattr(ns, key, default)
    return val if isinstance(val, int) else default


def decode_list_args(args: argparse.Namespace) -> ListArgs:
    """Decode list arguments."""
    return ListArgs(
        folder=_extract_str(args, "folder", "inbox"),
        count=_extract_int(args, "count", 10),
    )


def decode_read_args(args: argparse.Namespace) -> ReadArgs:
    """Decode read arguments."""
    return ReadArgs(index=_extract_int(args, "index", 1))


def _extract_str_tuple(ns: argparse.Namespace, key: str) -> tuple[str, ...]:
    """Extract a tuple of strings from namespace (for argparse append actions).

    Args:
        ns: Namespace to extract from.
        key: Attribute name.

    Returns:
        Tuple of strings, empty if not found or wrong type.
    """
    val: str | int | bool | list[str] | None = getattr(ns, key, None)
    if isinstance(val, list):
        return tuple(v for v in val if isinstance(v, str))
    return ()


def _extract_optional_str(ns: argparse.Namespace, key: str) -> str | None:
    """Extract optional string attribute from namespace.

    Args:
        ns: Namespace to extract from.
        key: Attribute name.

    Returns:
        String value if present and is a string, None otherwise.
    """
    val: str | int | bool | None = getattr(ns, key, None)
    return val if isinstance(val, str) else None


def _extract_bool(ns: argparse.Namespace, key: str, default: bool) -> bool:
    """Extract bool attribute from namespace.

    Args:
        ns: Namespace to extract from.
        key: Attribute name.
        default: Default value if not found or wrong type.

    Returns:
        Bool value or default.
    """
    val: str | int | bool | None = getattr(ns, key, default)
    return val if isinstance(val, bool) else default


def decode_send_args(args: argparse.Namespace) -> SendArgs:
    """Decode send arguments.

    Args:
        args: Parsed argparse namespace.

    Returns:
        SendArgs with to, subject, body_file, cc, bcc, and html fields.
    """
    return SendArgs(
        to=_extract_str(args, "to", ""),
        subject=_extract_str(args, "subject", ""),
        body_file=_extract_str(args, "body_file", ""),
        cc=_extract_str(args, "cc", ""),
        bcc=_extract_str(args, "bcc", ""),
        html=_extract_bool(args, "html", False),
        attachments=_extract_str_tuple(args, "attach"),
    )


def decode_search_args(args: argparse.Namespace) -> SearchArgs:
    """Decode search arguments.

    Args:
        args: Parsed argparse namespace.

    Returns:
        SearchArgs with query and count fields.
    """
    return SearchArgs(
        query=_extract_str(args, "query", ""),
        count=_extract_int(args, "count", 10),
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
    parser = _build_parser()
    args = parser.parse_args()
    command_str = _extract_str(args, "command", "")
    _dispatch_command(command_str, args)


if __name__ == "__main__":
    main()
