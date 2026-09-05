"""Hook factory helpers for platform_email tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

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


def make_fake_tokens(tokens: OAuthTokens) -> Callable[[], OAuthTokens | None]:
    """Create a hook that returns fixed tokens.

    Args:
        tokens: Tokens to return.

    Returns:
        A hook that answers with the tokens.
    """

    def _hook() -> OAuthTokens | None:
        return tokens

    return _hook


def make_fake_no_tokens() -> Callable[[], OAuthTokens | None]:
    """Create a hook that returns None (no cached tokens).

    Returns:
        A hook that answers with no tokens.
    """

    def _hook() -> OAuthTokens | None:
        return None

    return _hook


def make_fake_outlook_config(config: OutlookOAuthConfig) -> Callable[[], OutlookOAuthConfig]:
    """Create a hook that returns fixed Outlook config.

    Args:
        config: Config to return.

    Returns:
        Callable[[], OutlookOAuthConfig] that returns the config.
    """

    def _hook() -> OutlookOAuthConfig:
        return config

    return _hook


def make_fake_gmail_credentials(creds: OAuthCredentials) -> Callable[[], OAuthCredentials]:
    """Create a hook that returns fixed Gmail credentials.

    Args:
        creds: Credentials to return.

    Returns:
        Callable[[], OAuthCredentials] that returns the credentials.
    """

    def _hook() -> OAuthCredentials:
        return creds

    return _hook


def make_fake_path(path: str) -> Callable[[], str]:
    """Create a hook that returns a fixed path.

    Args:
        path: Path string to return.

    Returns:
        Callable[[], str] that returns the path.
    """

    def _hook() -> str:
        return path

    return _hook


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
    "make_fake_http_send",
    "make_fake_no_tokens",
    "make_fake_outlook_config",
    "make_fake_path",
    "make_fake_tokens",
    "make_raising_http_delete",
    "make_raising_http_get",
    "make_raising_http_send",
]
