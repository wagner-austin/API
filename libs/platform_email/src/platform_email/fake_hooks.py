"""Hook factory helpers for platform_email tests."""

from __future__ import annotations

from typing import Literal

from platform_email.testing import (
    ConsoleInputHook,
    ConsoleOutputHook,
    CurrentTimeHook,
    FileExistsHook,
    GetPathHook,
    HttpDeleteHook,
    HttpGetHook,
    HttpPatchHook,
    HttpPostHook,
    LoadGmailCredentialsHook,
    LoadOutlookConfigHook,
    LoadOutlookTokensHook,
    ReadFileHook,
    WriteFileHook,
)
from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailAddress,
    Folder,
    FolderType,
    OAuthCredentials,
    OAuthTokens,
    OutlookOAuthConfig,
)


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
]
