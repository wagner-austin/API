"""Test utilities, fakes, and hooks for platform_email.

Provides:
- EmailClientProtocol: Protocol for email client implementations
- HooksContainer: Dependency injection container for testing
- FakeEmailClient: In-memory fake for testing
- Factory functions for creating test data
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol, runtime_checkable

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
    _prod_load_gmail_credentials,
    _prod_load_gmail_tokens,
    _prod_load_outlook_config,
    _prod_load_outlook_tokens,
    _prod_open_browser,
    _prod_outlook_credentials_path,
    _prod_outlook_tokens_path,
    _prod_read_file,
    _prod_read_file_bytes,
    _prod_save_gmail_tokens,
    _prod_save_outlook_tokens,
    _prod_write_file,
)
from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailListResult,
    Folder,
    OAuthCredentials,
    OAuthTokens,
    OutlookOAuthConfig,
)

# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class EmailClientProtocol(Protocol):
    """Protocol for email client.

    Defines the interface for interacting with email APIs (Outlook, Gmail).
    """

    def send_email(
        self,
        *,
        to: tuple[str, ...],
        subject: str,
        body: str,
        body_type: BodyType = "text",
        cc: tuple[str, ...] = (),
        bcc: tuple[str, ...] = (),
        attachments: tuple[Attachment, ...] = (),
    ) -> Email:
        """Send an email.

        Args:
            to: Recipient email addresses.
            subject: Email subject.
            body: Email body content.
            body_type: Body content type (text or html).
            cc: CC recipient addresses.
            bcc: BCC recipient addresses.
            attachments: Email attachments.

        Returns:
            The sent Email.

        Raises:
            AppError[EmailErrorCode]: On send failure.
        """
        ...

    def get_email(self, *, email_id: str) -> Email:
        """Get a single email by ID.

        Args:
            email_id: ID of the email to retrieve.

        Returns:
            The Email.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        ...

    def list_emails(
        self,
        *,
        folder_id: str | None = None,
        query: str | None = None,
        max_results: int = 50,
        page_token: str | None = None,
    ) -> EmailListResult:
        """List emails in a folder.

        Args:
            folder_id: Folder to list from. None for all mail.
            query: Search query string.
            max_results: Maximum number of results.
            page_token: Token for pagination.

        Returns:
            EmailListResult with emails and next page token.
        """
        ...

    def search_emails(self, *, query: str, max_results: int = 50) -> tuple[Email, ...]:
        """Search emails.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            Tuple of matching emails.
        """
        ...

    def create_draft(
        self,
        *,
        to: tuple[str, ...],
        subject: str,
        body: str,
        body_type: BodyType = "text",
        cc: tuple[str, ...] = (),
        bcc: tuple[str, ...] = (),
    ) -> Draft:
        """Create a draft email.

        Args:
            to: Recipient email addresses.
            subject: Draft subject.
            body: Draft body content.
            body_type: Body content type.
            cc: CC recipient addresses.
            bcc: BCC recipient addresses.

        Returns:
            The created Draft.
        """
        ...

    def send_draft(self, *, draft_id: str) -> Email:
        """Send a draft email.

        Args:
            draft_id: ID of the draft to send.

        Returns:
            The sent Email.

        Raises:
            AppError[EmailErrorCode]: If draft not found.
        """
        ...

    def reply_to_email(
        self,
        *,
        email_id: str,
        body: str,
        body_type: BodyType = "text",
        reply_all: bool = False,
    ) -> Email:
        """Reply to an email.

        Args:
            email_id: ID of the email to reply to.
            body: Reply body content.
            body_type: Body content type.
            reply_all: Whether to reply to all recipients.

        Returns:
            The sent reply Email.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        ...

    def delete_email(self, *, email_id: str, permanent: bool = False) -> None:
        """Delete an email.

        Args:
            email_id: ID of the email to delete.
            permanent: If True, permanently delete. If False, move to trash.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        ...

    def move_email(self, *, email_id: str, destination_folder_id: str) -> Email:
        """Move an email to a folder.

        Args:
            email_id: ID of the email to move.
            destination_folder_id: ID of the destination folder.

        Returns:
            The moved Email.

        Raises:
            AppError[EmailErrorCode]: If email or folder not found.
        """
        ...

    def list_folders(self) -> tuple[Folder, ...]:
        """List all email folders.

        Returns:
            Tuple of all folders.
        """
        ...

    def get_attachment(self, *, email_id: str, attachment_id: str) -> Attachment:
        """Get an email attachment with content.

        Args:
            email_id: ID of the email containing the attachment.
            attachment_id: ID of the attachment.

        Returns:
            Attachment with content_bytes populated.

        Raises:
            AppError[EmailErrorCode]: If email or attachment not found.
        """
        ...


# =============================================================================
# HTTP Error Protocol
# =============================================================================


@runtime_checkable
class HTTPErrorProtocol(Protocol):
    """Protocol for HTTP error with code and response body."""

    @property
    def code(self) -> int:
        """HTTP status code."""
        ...

    def read(self) -> bytes:
        """Read response body."""
        ...


class FakeHTTPError(OSError):
    """Fake HTTP error for testing.

    Implements HTTPErrorProtocol for testing HTTP error handling.
    """

    def __init__(self, status_code: int, body: str) -> None:
        """Initialize fake HTTP error.

        Args:
            status_code: HTTP status code.
            body: Response body as string.
        """
        super().__init__(f"HTTP Error {status_code}")
        self._code = status_code
        self._body = body.encode("utf-8")

    @property
    def code(self) -> int:
        """HTTP status code."""
        return self._code

    def read(self) -> bytes:
        """Read response body."""
        return self._body


# =============================================================================
# Hook Type Definitions
# =============================================================================

HttpGetHook = Callable[[str, dict[str, str]], str]
HttpPostHook = Callable[[str, dict[str, str], str], str]
HttpPatchHook = Callable[[str, dict[str, str], str], str]
HttpDeleteHook = Callable[[str, dict[str, str]], None]
LoadOutlookTokensHook = Callable[[], OAuthTokens | None]
SaveOutlookTokensHook = Callable[[OAuthTokens], None]
LoadOutlookConfigHook = Callable[[], OutlookOAuthConfig]
LoadGmailTokensHook = Callable[[], OAuthTokens | None]
SaveGmailTokensHook = Callable[[OAuthTokens], None]
LoadGmailCredentialsHook = Callable[[], OAuthCredentials]
OpenBrowserHook = Callable[[str], None]
CurrentTimeHook = Callable[[], int]
ReadFileHook = Callable[[str], str]
ReadFileBytesHook = Callable[[str], bytes]
WriteFileHook = Callable[[str, str], None]
FileExistsHook = Callable[[str], bool]
ConsoleOutputHook = Callable[[str], None]
ConsoleInputHook = Callable[[str], str]
GetPathHook = Callable[[], str]

# CLI-specific hooks
CliGetEnvHook = Callable[[str], str | None]
CliSetEnvHook = Callable[[str, str], None]
CliGetNowHook = Callable[[], datetime]


# =============================================================================
# Hooks Container
# =============================================================================


class HooksContainer:
    """Container for dependency injection hooks."""

    http_get: HttpGetHook
    http_post: HttpPostHook
    http_patch: HttpPatchHook
    http_delete: HttpDeleteHook
    load_outlook_tokens: LoadOutlookTokensHook
    save_outlook_tokens: SaveOutlookTokensHook
    load_outlook_config: LoadOutlookConfigHook
    load_gmail_tokens: LoadGmailTokensHook
    save_gmail_tokens: SaveGmailTokensHook
    load_gmail_credentials: LoadGmailCredentialsHook
    open_browser: OpenBrowserHook
    current_time: CurrentTimeHook
    read_file: ReadFileHook
    read_file_bytes: ReadFileBytesHook
    write_file: WriteFileHook
    file_exists: FileExistsHook
    console_output: ConsoleOutputHook
    console_input: ConsoleInputHook
    outlook_tokens_path: GetPathHook
    outlook_credentials_path: GetPathHook
    gmail_tokens_path: GetPathHook
    gmail_credentials_path: GetPathHook

    # CLI-specific hooks
    cli_get_env: CliGetEnvHook
    cli_set_env: CliSetEnvHook
    cli_get_now: CliGetNowHook

    def reset(self) -> None:
        """Restore every hook to its production implementation.

        The same restoration `reset_hooks()` performs, exposed as a method so
        an autouse fixture can express the per-test isolation as
        `hooks.reset()`. That form states which container is protected, which
        a bare module-level call cannot.
        """
        reset_hooks()


hooks = HooksContainer()


# =============================================================================
# Production Implementations
# =============================================================================


def _init_production_hooks() -> None:
    """Initialize hooks with production implementations."""
    hooks.http_get = _prod_http_get
    hooks.http_post = _prod_http_post
    hooks.http_patch = _prod_http_patch
    hooks.http_delete = _prod_http_delete
    hooks.load_outlook_tokens = _prod_load_outlook_tokens
    hooks.save_outlook_tokens = _prod_save_outlook_tokens
    hooks.load_outlook_config = _prod_load_outlook_config
    hooks.load_gmail_tokens = _prod_load_gmail_tokens
    hooks.save_gmail_tokens = _prod_save_gmail_tokens
    hooks.load_gmail_credentials = _prod_load_gmail_credentials
    hooks.open_browser = _prod_open_browser
    hooks.current_time = _prod_current_time
    hooks.read_file = _prod_read_file
    hooks.read_file_bytes = _prod_read_file_bytes
    hooks.write_file = _prod_write_file
    hooks.file_exists = _prod_file_exists
    hooks.console_output = _prod_console_output
    hooks.console_input = _prod_console_input
    hooks.outlook_tokens_path = _prod_outlook_tokens_path
    hooks.outlook_credentials_path = _prod_outlook_credentials_path
    hooks.gmail_tokens_path = _prod_gmail_tokens_path
    hooks.gmail_credentials_path = _prod_gmail_credentials_path
    # CLI hooks
    hooks.cli_get_env = _prod_cli_get_env
    hooks.cli_set_env = _prod_cli_set_env
    hooks.cli_get_now = _prod_cli_get_now


# Initialize on module load
_init_production_hooks()


def reset_hooks() -> None:
    """Reset all hooks to production implementations (for test teardown)."""
    from platform_email import _prod_hooks

    _prod_hooks._cli_env_loaded = False
    _prod_hooks._cli_env_cache = {}
    _init_production_hooks()


# =============================================================================
# Fake Email Client
# =============================================================================


__all__ = [
    "CliGetEnvHook",
    "CliGetNowHook",
    "CliSetEnvHook",
    "ConsoleInputHook",
    "ConsoleOutputHook",
    "CurrentTimeHook",
    "EmailClientProtocol",
    "FakeHTTPError",
    "FileExistsHook",
    "GetPathHook",
    "HTTPErrorProtocol",
    "HooksContainer",
    "HttpDeleteHook",
    "HttpGetHook",
    "HttpPatchHook",
    "HttpPostHook",
    "LoadGmailCredentialsHook",
    "LoadGmailTokensHook",
    "LoadOutlookConfigHook",
    "LoadOutlookTokensHook",
    "OpenBrowserHook",
    "ReadFileBytesHook",
    "ReadFileHook",
    "SaveGmailTokensHook",
    "SaveOutlookTokensHook",
    "WriteFileHook",
    "hooks",
    "reset_hooks",
]
