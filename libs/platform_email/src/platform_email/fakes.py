"""FakeEmailClient: the shared in-memory email client double."""

from __future__ import annotations

from platform_core.errors import AppError, EmailErrorCode

from platform_email.testing import EmailClientProtocol
from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailAddress,
    EmailListResult,
    Folder,
)


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


__all__ = [
    "FakeEmailClient",
]
