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
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from platform_core.errors import AppError, EmailErrorCode

from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailAddress,
    EmailListResult,
    Folder,
    FolderType,
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


hooks = HooksContainer()


# =============================================================================
# Production Implementations
# =============================================================================


def _prod_http_get(url: str, headers: dict[str, str]) -> str:
    """Production HTTP GET using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url)
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    body = response.read()
    response.close()
    return body.decode("utf-8")


def _prod_http_post(url: str, headers: dict[str, str], body: str) -> str:
    """Production HTTP POST using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    response_body = response.read()
    response.close()
    return response_body.decode("utf-8")


def _prod_http_patch(url: str, headers: dict[str, str], body: str) -> str:
    """Production HTTP PATCH using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, data=body.encode("utf-8"), method="PATCH")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    response_body = response.read()
    response.close()
    return response_body.decode("utf-8")


def _prod_http_delete(url: str, headers: dict[str, str]) -> None:
    """Production HTTP DELETE using urllib."""
    import urllib.request
    from http.client import HTTPResponse

    req = urllib.request.Request(url, method="DELETE")
    for key, value in headers.items():
        req.add_header(key, value)
    response = urllib.request.urlopen(req, timeout=30)
    assert isinstance(response, HTTPResponse)
    response.close()


def _prod_load_outlook_tokens() -> OAuthTokens | None:
    """Production Outlook token loader."""
    from pathlib import Path

    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_email.types import decode_oauth_tokens

    tokens_path = Path(hooks.outlook_tokens_path())
    if not tokens_path.exists():
        return None
    content = tokens_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError):
        return None
    return decode_oauth_tokens(data)


def _prod_save_outlook_tokens(tokens: OAuthTokens) -> None:
    """Production Outlook token saver."""
    from pathlib import Path

    from platform_core.json_utils import dump_json_str

    from platform_email.types import encode_oauth_tokens

    tokens_path = Path(hooks.outlook_tokens_path())
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    content = dump_json_str(encode_oauth_tokens(tokens), indent=2)
    tokens_path.write_text(content, encoding="utf-8")


def _prod_load_outlook_config() -> OutlookOAuthConfig:
    """Production Outlook config loader."""
    from pathlib import Path

    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_email.types import decode_outlook_oauth_config

    creds_path = Path(hooks.outlook_credentials_path())
    if not creds_path.exists():
        msg = f"Outlook credentials file not found at {creds_path}"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
    content = creds_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Outlook credentials file is not valid JSON: {e}"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401) from e
    return decode_outlook_oauth_config(data)


def _prod_load_gmail_tokens() -> OAuthTokens | None:
    """Production Gmail token loader."""
    from pathlib import Path

    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
    )

    from platform_email.types import decode_oauth_tokens

    tokens_path = Path(hooks.gmail_tokens_path())
    if not tokens_path.exists():
        return None
    content = tokens_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError):
        return None
    return decode_oauth_tokens(data)


def _prod_save_gmail_tokens(tokens: OAuthTokens) -> None:
    """Production Gmail token saver."""
    from pathlib import Path

    from platform_core.json_utils import dump_json_str

    from platform_email.types import encode_oauth_tokens

    tokens_path = Path(hooks.gmail_tokens_path())
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    content = dump_json_str(encode_oauth_tokens(tokens), indent=2)
    tokens_path.write_text(content, encoding="utf-8")


def _prod_load_gmail_credentials() -> OAuthCredentials:
    """Production Gmail credentials loader."""
    from pathlib import Path

    from platform_core.json_utils import (
        InvalidJsonError,
        JSONTypeError,
        load_json_str,
        narrow_json_to_dict,
        require_list,
        require_str,
    )

    creds_path = Path(hooks.gmail_credentials_path())
    if not creds_path.exists():
        msg = f"Gmail credentials file not found at {creds_path}"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
    content = creds_path.read_text(encoding="utf-8")
    try:
        raw_value = load_json_str(content)
        data = narrow_json_to_dict(raw_value)
    except (InvalidJsonError, JSONTypeError) as e:
        msg = f"Gmail credentials file is not valid JSON: {e}"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401) from e
    # Google credentials file has "installed" wrapper
    installed_raw = data.get("installed")
    if not isinstance(installed_raw, dict):
        msg = "Gmail credentials file missing 'installed' section"
        raise AppError(EmailErrorCode.CREDENTIALS_NOT_FOUND, msg, http_status=401)
    installed = installed_raw
    redirect_uris_raw = require_list(installed, "redirect_uris")
    redirect_uri = redirect_uris_raw[0] if redirect_uris_raw else "http://localhost"
    if not isinstance(redirect_uri, str):
        redirect_uri = "http://localhost"
    return OAuthCredentials(
        client_id=require_str(installed, "client_id"),
        client_secret=require_str(installed, "client_secret"),
        redirect_uri=redirect_uri,
    )


def _prod_open_browser(url: str) -> None:
    """Production browser opener."""
    import webbrowser

    webbrowser.open(url)


def _prod_current_time() -> int:
    """Production current time in seconds since epoch."""
    import time

    return int(time.time())


def _prod_read_file(path: str) -> str:
    """Production file reader."""
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8")


def _prod_read_file_bytes(path: str) -> bytes:
    """Production binary file reader.

    Args:
        path: Path to the file.

    Returns:
        Raw bytes of the file.
    """
    from pathlib import Path

    return Path(path).read_bytes()


def _prod_write_file(path: str, content: str) -> None:
    """Production file writer."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _prod_file_exists(path: str) -> bool:
    """Production file exists check."""
    from pathlib import Path

    return Path(path).exists()


def _prod_console_output(message: str) -> None:
    """Production console output."""
    import sys

    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _prod_console_input(prompt: str) -> str:
    """Production console input."""
    return input(prompt)


def _prod_outlook_tokens_path() -> str:
    """Production Outlook tokens path."""
    from platform_email.config import DEFAULT_OUTLOOK_TOKENS_PATH

    return str(DEFAULT_OUTLOOK_TOKENS_PATH)


def _prod_outlook_credentials_path() -> str:
    """Production Outlook credentials path."""
    from platform_email.config import DEFAULT_OUTLOOK_CREDENTIALS_PATH

    return str(DEFAULT_OUTLOOK_CREDENTIALS_PATH)


def _prod_gmail_tokens_path() -> str:
    """Production Gmail tokens path."""
    from platform_email.config import DEFAULT_GMAIL_TOKENS_PATH

    return str(DEFAULT_GMAIL_TOKENS_PATH)


def _prod_gmail_credentials_path() -> str:
    """Production Gmail credentials path."""
    from platform_email.config import DEFAULT_GMAIL_CREDENTIALS_PATH

    return str(DEFAULT_GMAIL_CREDENTIALS_PATH)


# =============================================================================
# CLI Production Implementations
# =============================================================================

# Module-level cache for CLI environment
_cli_env_loaded: bool = False
_cli_env_cache: dict[str, str] = {}


def _prod_cli_get_env(key: str) -> str | None:
    """Production CLI environment variable getter.

    Loads from .env file in the platform_email package directory.

    Args:
        key: Environment variable name.

    Returns:
        Value if found, None otherwise.
    """
    import os

    global _cli_env_loaded, _cli_env_cache

    if not _cli_env_loaded:
        # Load from .env file relative to this module
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        k, v = line.strip().split("=", 1)
                        _cli_env_cache[k] = v
        _cli_env_loaded = True

    return _cli_env_cache.get(key)


def _prod_cli_set_env(key: str, value: str) -> None:
    """Production CLI environment variable setter.

    Updates the in-memory cache with the new value.

    Args:
        key: Environment variable name.
        value: Value to set.
    """
    global _cli_env_cache
    _cli_env_cache[key] = value


def _prod_cli_get_now() -> datetime:
    """Production CLI current datetime.

    Returns:
        Current datetime.
    """
    return datetime.now()


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
    global _cli_env_loaded, _cli_env_cache, guard_find_monorepo_root, guard_load_orchestrator
    _cli_env_loaded = False
    _cli_env_cache = {}
    guard_find_monorepo_root = None
    guard_load_orchestrator = None
    _init_production_hooks()


# =============================================================================
# Guard Script Hooks
# =============================================================================


class RunForProjectProto(Protocol):
    """Protocol for run_for_project function from monorepo_guards."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guard checks for a project.

        Args:
            monorepo_root: Root directory of the monorepo.
            project_root: Root directory of the project to check.

        Returns:
            Exit code (0 for success).
        """
        ...


class FindMonorepoRootProto(Protocol):
    """Protocol for finding the monorepo root directory."""

    def __call__(self, start: Path) -> Path:
        """Find monorepo root by searching upward for libs directory.

        Args:
            start: Starting directory for search.

        Returns:
            Path to monorepo root.
        """
        ...


class LoadOrchestratorProto(Protocol):
    """Protocol for loading the guard orchestrator."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProto:
        """Load the orchestrator's run_for_project function.

        Args:
            monorepo_root: Root directory of the monorepo.

        Returns:
            The run_for_project function.
        """
        ...


# Guard hooks - None means use default behavior (production implementation)
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None


# =============================================================================
# Fake Email Client
# =============================================================================


class FakeEmailClient(EmailClientProtocol):
    """In-memory fake email client for testing."""

    def __init__(self) -> None:
        """Initialize the fake client with empty state."""
        self._emails: dict[str, Email] = {}
        self._drafts: dict[str, Draft] = {}
        self._folders: list[Folder] = []
        self._attachments: dict[str, dict[str, Attachment]] = {}
        self._next_id: int = 1
        self._sent_emails: list[Email] = []
        self._deleted_emails: list[tuple[str, bool]] = []
        self._moved_emails: list[tuple[str, str]] = []

    # -------------------------------------------------------------------------
    # Test Helpers
    # -------------------------------------------------------------------------

    def add_email(self, email: Email) -> None:
        """Add a fake email for testing.

        Args:
            email: Email to add.
        """
        self._emails[email["id"]] = email

    def add_draft(self, draft: Draft) -> None:
        """Add a fake draft for testing.

        Args:
            draft: Draft to add.
        """
        self._drafts[draft["id"]] = draft

    def add_folder(self, folder: Folder) -> None:
        """Add a fake folder for testing.

        Args:
            folder: Folder to add.
        """
        self._folders.append(folder)

    def add_attachment(self, email_id: str, attachment: Attachment) -> None:
        """Add a fake attachment for testing.

        Args:
            email_id: Email ID containing the attachment.
            attachment: Attachment to add.
        """
        if email_id not in self._attachments:
            self._attachments[email_id] = {}
        self._attachments[email_id][attachment["id"]] = attachment

    def get_sent_emails(self) -> list[Email]:
        """Get all emails sent via send_email().

        Returns:
            List of sent emails.
        """
        return list(self._sent_emails)

    def get_deleted_emails(self) -> list[tuple[str, bool]]:
        """Get all (email_id, permanent) pairs deleted via delete_email().

        Returns:
            List of (email_id, permanent) tuples.
        """
        return list(self._deleted_emails)

    def get_moved_emails(self) -> list[tuple[str, str]]:
        """Get all (email_id, folder_id) pairs moved via move_email().

        Returns:
            List of (email_id, folder_id) tuples.
        """
        return list(self._moved_emails)

    # -------------------------------------------------------------------------
    # Protocol Implementation
    # -------------------------------------------------------------------------

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
        """Send an email."""
        email_id = f"fake_email_{self._next_id}"
        self._next_id += 1

        to_addrs: list[EmailAddress] = []
        for addr in to:
            to_addrs.append(EmailAddress(address=addr, name=""))
        cc_addrs: list[EmailAddress] = []
        for addr in cc:
            cc_addrs.append(EmailAddress(address=addr, name=""))
        bcc_addrs: list[EmailAddress] = []
        for addr in bcc:
            bcc_addrs.append(EmailAddress(address=addr, name=""))

        email = Email(
            id=email_id,
            thread_id=f"thread_{email_id}",
            folder_id="sent",
            subject=subject,
            body=body,
            body_type=body_type,
            from_address=EmailAddress(address="test@example.com", name="Test User"),
            to=tuple(to_addrs),
            cc=tuple(cc_addrs),
            bcc=tuple(bcc_addrs),
            sent_at="2025-01-01T00:00:00Z",
            received_at="2025-01-01T00:00:00Z",
            is_read=True,
            is_draft=False,
            has_attachments=len(attachments) > 0,
            importance="normal",
        )

        self._emails[email_id] = email
        self._sent_emails.append(email)

        if attachments:
            self._attachments[email_id] = {}
            for att in attachments:
                self._attachments[email_id][att["id"]] = att

        return email

    def get_email(self, *, email_id: str) -> Email:
        """Get a single email by ID."""
        if email_id not in self._emails:
            msg = f"Email '{email_id}' not found"
            raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)
        return self._emails[email_id]

    def list_emails(
        self,
        *,
        folder_id: str | None = None,
        query: str | None = None,
        max_results: int = 50,
        page_token: str | None = None,
    ) -> EmailListResult:
        """List emails in a folder."""
        emails = list(self._emails.values())

        if folder_id is not None:
            emails = [e for e in emails if e["folder_id"] == folder_id]

        if query is not None:
            lower_query = query.lower()
            emails = [
                e
                for e in emails
                if lower_query in e["subject"].lower() or lower_query in e["body"].lower()
            ]

        emails = emails[:max_results]

        return EmailListResult(
            emails=tuple(emails),
            next_page_token=None,
        )

    def search_emails(self, *, query: str, max_results: int = 50) -> tuple[Email, ...]:
        """Search emails."""
        result = self.list_emails(query=query, max_results=max_results)
        return result["emails"]

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
        """Create a draft email."""
        draft_id = f"fake_draft_{self._next_id}"
        self._next_id += 1

        to_addrs: list[EmailAddress] = []
        for addr in to:
            to_addrs.append(EmailAddress(address=addr, name=""))
        cc_addrs: list[EmailAddress] = []
        for addr in cc:
            cc_addrs.append(EmailAddress(address=addr, name=""))
        bcc_addrs: list[EmailAddress] = []
        for addr in bcc:
            bcc_addrs.append(EmailAddress(address=addr, name=""))

        draft = Draft(
            id=draft_id,
            subject=subject,
            body=body,
            body_type=body_type,
            to=tuple(to_addrs),
            cc=tuple(cc_addrs),
            bcc=tuple(bcc_addrs),
        )

        self._drafts[draft_id] = draft
        return draft

    def send_draft(self, *, draft_id: str) -> Email:
        """Send a draft email."""
        if draft_id not in self._drafts:
            msg = f"Draft '{draft_id}' not found"
            raise AppError(EmailErrorCode.DRAFT_NOT_FOUND, msg, http_status=404)

        draft = self._drafts[draft_id]
        to_tuple: tuple[str, ...] = tuple(addr["address"] for addr in draft["to"])
        cc_tuple: tuple[str, ...] = tuple(addr["address"] for addr in draft["cc"])
        bcc_tuple: tuple[str, ...] = tuple(addr["address"] for addr in draft["bcc"])

        email = self.send_email(
            to=to_tuple,
            subject=draft["subject"],
            body=draft["body"],
            body_type=draft["body_type"],
            cc=cc_tuple,
            bcc=bcc_tuple,
        )

        del self._drafts[draft_id]
        return email

    def reply_to_email(
        self,
        *,
        email_id: str,
        body: str,
        body_type: BodyType = "text",
        reply_all: bool = False,
    ) -> Email:
        """Reply to an email."""
        if email_id not in self._emails:
            msg = f"Email '{email_id}' not found"
            raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)

        original = self._emails[email_id]
        to_tuple = (original["from_address"]["address"],)
        cc_tuple: tuple[str, ...] = ()

        if reply_all:
            cc_list: list[str] = []
            for addr in original["to"]:
                if addr["address"] != "test@example.com":
                    cc_list.append(addr["address"])
            for addr in original["cc"]:
                cc_list.append(addr["address"])
            cc_tuple = tuple(cc_list)

        return self.send_email(
            to=to_tuple,
            subject=f"Re: {original['subject']}",
            body=body,
            body_type=body_type,
            cc=cc_tuple,
        )

    def delete_email(self, *, email_id: str, permanent: bool = False) -> None:
        """Delete an email."""
        if email_id not in self._emails:
            msg = f"Email '{email_id}' not found"
            raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)

        self._deleted_emails.append((email_id, permanent))

        if permanent:
            del self._emails[email_id]
        else:
            # Move to trash
            email = self._emails[email_id]
            self._emails[email_id] = Email(
                id=email["id"],
                thread_id=email["thread_id"],
                folder_id="trash",
                subject=email["subject"],
                body=email["body"],
                body_type=email["body_type"],
                from_address=email["from_address"],
                to=email["to"],
                cc=email["cc"],
                bcc=email["bcc"],
                sent_at=email["sent_at"],
                received_at=email["received_at"],
                is_read=email["is_read"],
                is_draft=email["is_draft"],
                has_attachments=email["has_attachments"],
                importance=email["importance"],
            )

    def move_email(self, *, email_id: str, destination_folder_id: str) -> Email:
        """Move an email to a folder."""
        if email_id not in self._emails:
            msg = f"Email '{email_id}' not found"
            raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)

        self._moved_emails.append((email_id, destination_folder_id))

        email = self._emails[email_id]
        moved = Email(
            id=email["id"],
            thread_id=email["thread_id"],
            folder_id=destination_folder_id,
            subject=email["subject"],
            body=email["body"],
            body_type=email["body_type"],
            from_address=email["from_address"],
            to=email["to"],
            cc=email["cc"],
            bcc=email["bcc"],
            sent_at=email["sent_at"],
            received_at=email["received_at"],
            is_read=email["is_read"],
            is_draft=email["is_draft"],
            has_attachments=email["has_attachments"],
            importance=email["importance"],
        )
        self._emails[email_id] = moved
        return moved

    def list_folders(self) -> tuple[Folder, ...]:
        """List all email folders."""
        return tuple(self._folders)

    def get_attachment(self, *, email_id: str, attachment_id: str) -> Attachment:
        """Get an email attachment with content."""
        if email_id not in self._attachments:
            msg = f"Email '{email_id}' has no attachments"
            raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)
        email_attachments = self._attachments[email_id]
        if attachment_id not in email_attachments:
            msg = f"Attachment '{attachment_id}' not found"
            raise AppError(EmailErrorCode.EMAIL_NOT_FOUND, msg, http_status=404)
        return email_attachments[attachment_id]


# =============================================================================
# Factory Helpers for Tests
# =============================================================================


def make_fake_http_get(response: str) -> HttpGetHook:
    """Create a hook that returns a fixed response.

    Args:
        response: Response body to return.

    Returns:
        HttpGetHook that returns the fixed response.
    """

    def _hook(url: str, headers: dict[str, str]) -> str:
        return response

    return _hook


def make_fake_http_post(response: str) -> HttpPostHook:
    """Create a hook that returns a fixed response.

    Args:
        response: Response body to return.

    Returns:
        HttpPostHook that returns the fixed response.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        return response

    return _hook


def make_fake_http_patch(response: str) -> HttpPatchHook:
    """Create a hook that returns a fixed response.

    Args:
        response: Response body to return.

    Returns:
        HttpPatchHook that returns the fixed response.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        return response

    return _hook


def make_fake_http_delete() -> HttpDeleteHook:
    """Create a hook that does nothing (successful delete).

    Returns:
        HttpDeleteHook that does nothing.
    """

    def _hook(url: str, headers: dict[str, str]) -> None:
        pass

    return _hook


def make_raising_http_get(exc: BaseException) -> HttpGetHook:
    """Create a hook that raises an exception.

    Args:
        exc: Exception to raise.

    Returns:
        HttpGetHook that raises the exception.
    """

    def _hook(url: str, headers: dict[str, str]) -> str:
        raise exc

    return _hook


def make_raising_http_post(exc: BaseException) -> HttpPostHook:
    """Create a hook that raises an exception.

    Args:
        exc: Exception to raise.

    Returns:
        HttpPostHook that raises the exception.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        raise exc

    return _hook


def make_raising_http_patch(exc: BaseException) -> HttpPatchHook:
    """Create a hook that raises an exception.

    Args:
        exc: Exception to raise.

    Returns:
        HttpPatchHook that raises the exception.
    """

    def _hook(url: str, headers: dict[str, str], body: str) -> str:
        raise exc

    return _hook


def make_raising_http_delete(exc: BaseException) -> HttpDeleteHook:
    """Create a hook that raises an exception.

    Args:
        exc: Exception to raise.

    Returns:
        HttpDeleteHook that raises the exception.
    """

    def _hook(url: str, headers: dict[str, str]) -> None:
        raise exc

    return _hook


def make_fake_tokens(tokens: OAuthTokens) -> LoadOutlookTokensHook:
    """Create a hook that returns fixed tokens.

    Args:
        tokens: Tokens to return.

    Returns:
        LoadTokensHook that returns the tokens.
    """

    def _hook() -> OAuthTokens | None:
        return tokens

    return _hook


def make_fake_no_tokens() -> LoadOutlookTokensHook:
    """Create a hook that returns None (no cached tokens).

    Returns:
        LoadTokensHook that returns None.
    """

    def _hook() -> OAuthTokens | None:
        return None

    return _hook


def make_fake_outlook_config(config: OutlookOAuthConfig) -> LoadOutlookConfigHook:
    """Create a hook that returns fixed Outlook config.

    Args:
        config: Config to return.

    Returns:
        LoadOutlookConfigHook that returns the config.
    """

    def _hook() -> OutlookOAuthConfig:
        return config

    return _hook


def make_fake_gmail_credentials(creds: OAuthCredentials) -> LoadGmailCredentialsHook:
    """Create a hook that returns fixed Gmail credentials.

    Args:
        creds: Credentials to return.

    Returns:
        LoadGmailCredentialsHook that returns the credentials.
    """

    def _hook() -> OAuthCredentials:
        return creds

    return _hook


def make_fake_current_time(timestamp: int) -> CurrentTimeHook:
    """Create a hook that returns a fixed timestamp.

    Args:
        timestamp: Unix timestamp to return.

    Returns:
        CurrentTimeHook that returns the timestamp.
    """

    def _hook() -> int:
        return timestamp

    return _hook


def make_fake_file_system(
    files: dict[str, str],
) -> tuple[ReadFileHook, WriteFileHook, FileExistsHook]:
    """Create hooks that use an in-memory file system.

    Args:
        files: Initial file contents keyed by path.

    Returns:
        Tuple of (read_hook, write_hook, exists_hook).
    """
    storage = dict(files)

    def _read(path: str) -> str:
        if path not in storage:
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        return storage[path]

    def _write(path: str, content: str) -> None:
        storage[path] = content

    def _exists(path: str) -> bool:
        return path in storage

    return _read, _write, _exists


def make_fake_path(path: str) -> GetPathHook:
    """Create a hook that returns a fixed path.

    Args:
        path: Path string to return.

    Returns:
        GetPathHook that returns the path.
    """

    def _hook() -> str:
        return path

    return _hook


def make_fake_console(inputs: list[str]) -> tuple[ConsoleOutputHook, ConsoleInputHook]:
    """Create hooks for fake console I/O.

    Args:
        inputs: List of strings to return from console_input in order.

    Returns:
        Tuple of (output_hook, input_hook).
    """
    outputs: list[str] = []
    input_index = [0]

    def _output(message: str) -> None:
        outputs.append(message)

    def _input(prompt: str) -> str:
        if input_index[0] >= len(inputs):
            return ""
        result = inputs[input_index[0]]
        input_index[0] += 1
        return result

    return _output, _input


def make_fake_email(
    *,
    email_id: str = "test_email_1",
    thread_id: str = "test_thread_1",
    folder_id: str = "inbox",
    subject: str = "Test Subject",
    body: str = "Test body content",
    body_type: BodyType = "text",
    from_address: str = "sender@example.com",
    from_name: str = "Sender Name",
    to: tuple[str, ...] = ("recipient@example.com",),
    cc: tuple[str, ...] = (),
    bcc: tuple[str, ...] = (),
    sent_at: str = "2025-01-01T10:00:00Z",
    received_at: str = "2025-01-01T10:00:01Z",
    is_read: bool = False,
    is_draft: bool = False,
    has_attachments: bool = False,
    importance: EmailImportance = "normal",
) -> Email:
    """Create a fake Email for testing.

    Args:
        email_id: Email ID.
        thread_id: Thread ID.
        folder_id: Folder ID.
        subject: Email subject.
        body: Email body.
        body_type: Body type (text or html).
        from_address: Sender address.
        from_name: Sender name.
        to: Recipient addresses.
        cc: CC addresses.
        bcc: BCC addresses.
        sent_at: Sent timestamp.
        received_at: Received timestamp.
        is_read: Read status.
        is_draft: Draft status.
        has_attachments: Has attachments.
        importance: Email importance.

    Returns:
        Email with the specified values.
    """
    to_addrs: list[EmailAddress] = []
    for addr in to:
        to_addrs.append(EmailAddress(address=addr, name=""))
    cc_addrs: list[EmailAddress] = []
    for addr in cc:
        cc_addrs.append(EmailAddress(address=addr, name=""))
    bcc_addrs: list[EmailAddress] = []
    for addr in bcc:
        bcc_addrs.append(EmailAddress(address=addr, name=""))

    importance_val: EmailImportance
    if importance == "low":
        importance_val = "low"
    elif importance == "high":
        importance_val = "high"
    else:
        importance_val = "normal"

    return Email(
        id=email_id,
        thread_id=thread_id,
        folder_id=folder_id,
        subject=subject,
        body=body,
        body_type=body_type,
        from_address=EmailAddress(address=from_address, name=from_name),
        to=tuple(to_addrs),
        cc=tuple(cc_addrs),
        bcc=tuple(bcc_addrs),
        sent_at=sent_at,
        received_at=received_at,
        is_read=is_read,
        is_draft=is_draft,
        has_attachments=has_attachments,
        importance=importance_val,
    )


def make_fake_folder(
    *,
    folder_id: str = "inbox",
    name: str = "Inbox",
    folder_type: FolderType = "inbox",
    unread_count: int = 0,
    total_count: int = 0,
) -> Folder:
    """Create a fake Folder for testing.

    Args:
        folder_id: Folder ID.
        name: Folder name.
        folder_type: Type of folder.
        unread_count: Number of unread emails.
        total_count: Total number of emails.

    Returns:
        Folder with the specified values.
    """
    return Folder(
        id=folder_id,
        name=name,
        folder_type=folder_type,
        unread_count=unread_count,
        total_count=total_count,
    )


def make_fake_attachment(
    *,
    attachment_id: str = "attachment_1",
    name: str = "document.pdf",
    content_type: str = "application/pdf",
    size: int = 1024,
    content_bytes: str | None = None,
) -> Attachment:
    """Create a fake Attachment for testing.

    Args:
        attachment_id: Attachment ID.
        name: Filename.
        content_type: MIME type.
        size: Size in bytes.
        content_bytes: Base64-encoded content.

    Returns:
        Attachment with the specified values.
    """
    return Attachment(
        id=attachment_id,
        name=name,
        content_type=content_type,
        size=size,
        content_bytes=content_bytes,
    )


def make_fake_draft(
    *,
    draft_id: str = "draft_1",
    subject: str = "Draft Subject",
    body: str = "Draft body content",
    body_type: BodyType = "text",
    to: tuple[str, ...] = ("recipient@example.com",),
    cc: tuple[str, ...] = (),
    bcc: tuple[str, ...] = (),
) -> Draft:
    """Create a fake Draft for testing.

    Args:
        draft_id: Draft ID.
        subject: Draft subject.
        body: Draft body.
        body_type: Body type.
        to: Recipient addresses.
        cc: CC addresses.
        bcc: BCC addresses.

    Returns:
        Draft with the specified values.
    """
    to_addrs: list[EmailAddress] = []
    for addr in to:
        to_addrs.append(EmailAddress(address=addr, name=""))
    cc_addrs: list[EmailAddress] = []
    for addr in cc:
        cc_addrs.append(EmailAddress(address=addr, name=""))
    bcc_addrs: list[EmailAddress] = []
    for addr in bcc:
        bcc_addrs.append(EmailAddress(address=addr, name=""))

    return Draft(
        id=draft_id,
        subject=subject,
        body=body,
        body_type=body_type,
        to=tuple(to_addrs),
        cc=tuple(cc_addrs),
        bcc=tuple(bcc_addrs),
    )


# Type alias for EmailImportance parameter
EmailImportance = Literal["low", "normal", "high"]

__all__ = [
    "CliGetEnvHook",
    "CliGetNowHook",
    "CliSetEnvHook",
    "ConsoleInputHook",
    "ConsoleOutputHook",
    "CurrentTimeHook",
    "EmailClientProtocol",
    "FakeEmailClient",
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
    "make_fake_attachment",
    "make_fake_console",
    "make_fake_current_time",
    "make_fake_draft",
    "make_fake_email",
    "make_fake_file_system",
    "make_fake_folder",
    "make_fake_gmail_credentials",
    "make_fake_http_delete",
    "make_fake_http_get",
    "make_fake_http_patch",
    "make_fake_http_post",
    "make_fake_no_tokens",
    "make_fake_outlook_config",
    "make_fake_path",
    "make_fake_tokens",
    "make_raising_http_delete",
    "make_raising_http_get",
    "make_raising_http_patch",
    "make_raising_http_post",
    "reset_hooks",
]
