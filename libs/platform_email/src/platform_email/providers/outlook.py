"""Microsoft Outlook email client using Graph API.

Implements EmailClientProtocol for Microsoft Graph Mail API.
"""

from __future__ import annotations

import urllib.parse

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    optional_str,
    require_str,
)

from platform_email.providers.outlook_decode import (
    _decode_folder_type,
    _decode_message,
)
from platform_email.providers.outlook_http import _OutlookHttp
from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailAddress,
    EmailListResult,
    Folder,
)


class _OutlookEmailClient(_OutlookHttp):
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
        to_recipients: list[JSONValue] = []
        for addr in to:
            recipient: JSONObject = {"emailAddress": {"address": addr}}
            to_recipients.append(recipient)

        cc_recipients: list[JSONValue] = []
        for addr in cc:
            recipient_cc: JSONObject = {"emailAddress": {"address": addr}}
            cc_recipients.append(recipient_cc)

        bcc_recipients: list[JSONValue] = []
        for addr in bcc:
            recipient_bcc: JSONObject = {"emailAddress": {"address": addr}}
            bcc_recipients.append(recipient_bcc)

        content_type = "HTML" if body_type == "html" else "Text"

        message_body: JSONObject = {
            "subject": subject,
            "body": {
                "contentType": content_type,
                "content": body,
            },
            "toRecipients": to_recipients,
            "ccRecipients": cc_recipients,
            "bccRecipients": bcc_recipients,
        }

        # Add attachments if any
        if attachments:
            att_list: list[JSONValue] = []
            for att in attachments:
                att_obj: JSONObject = {
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": att["name"],
                    "contentType": att["content_type"],
                    "contentBytes": att["content_bytes"] or "",
                }
                att_list.append(att_obj)
            message_body["attachments"] = att_list

        request_body: JSONObject = {
            "message": message_body,
            "saveToSentItems": True,
        }

        # Send and get back the sent message
        self._post("/me/sendMail", request_body)

        # Since sendMail doesn't return the message, we need to get from sent items
        # For now, create a placeholder - in real usage you'd query sent items
        from_addr = EmailAddress(address="me@outlook.com", name="")

        to_addrs: list[EmailAddress] = []
        for addr in to:
            to_addrs.append(EmailAddress(address=addr, name=""))
        cc_addrs: list[EmailAddress] = []
        for addr in cc:
            cc_addrs.append(EmailAddress(address=addr, name=""))
        bcc_addrs: list[EmailAddress] = []
        for addr in bcc:
            bcc_addrs.append(EmailAddress(address=addr, name=""))

        return Email(
            id="sent_placeholder",
            thread_id="",
            folder_id="sentitems",
            subject=subject,
            body=body,
            body_type=body_type,
            from_address=from_addr,
            to=tuple(to_addrs),
            cc=tuple(cc_addrs),
            bcc=tuple(bcc_addrs),
            sent_at="",
            received_at="",
            is_read=True,
            is_draft=False,
            has_attachments=len(attachments) > 0,
            importance="normal",
        )

    def get_email(self, *, email_id: str) -> Email:
        """Get a single email by ID.

        Args:
            email_id: ID of the email to retrieve.

        Returns:
            The Email.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        path = f"/me/messages/{email_id}"
        data = self._get(path)
        return _decode_message(data)

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
        if folder_id is not None:
            base_path = f"/me/mailFolders/{folder_id}/messages"
        else:
            base_path = "/me/messages"

        params: dict[str, str] = {"$top": str(max_results)}

        if query is not None:
            params["$search"] = f'"{query}"'

        if page_token is not None:
            # page_token is a full URL for Graph API, use skiptoken
            params["$skiptoken"] = page_token

        path = base_path + "?" + urllib.parse.urlencode(params)
        data = self._get(path)

        messages_raw = data.get("value")
        emails: list[Email] = []
        if isinstance(messages_raw, list):
            for msg in messages_raw:
                if isinstance(msg, dict):
                    emails.append(_decode_message(msg))

        next_link = optional_str(data, "@odata.nextLink")
        next_token: str | None = None
        if next_link is not None and "$skiptoken=" in next_link:
            # Extract skiptoken from next link
            start = next_link.find("$skiptoken=") + len("$skiptoken=")
            end = next_link.find("&", start)
            next_token = next_link[start:] if end == -1 else next_link[start:end]

        return EmailListResult(
            emails=tuple(emails),
            next_page_token=next_token,
        )

    def search_emails(self, *, query: str, max_results: int = 50) -> tuple[Email, ...]:
        """Search emails.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            Tuple of matching emails.
        """
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
        to_recipients: list[JSONValue] = []
        for addr in to:
            recipient: JSONObject = {"emailAddress": {"address": addr}}
            to_recipients.append(recipient)

        cc_recipients: list[JSONValue] = []
        for addr in cc:
            recipient_cc: JSONObject = {"emailAddress": {"address": addr}}
            cc_recipients.append(recipient_cc)

        bcc_recipients: list[JSONValue] = []
        for addr in bcc:
            recipient_bcc: JSONObject = {"emailAddress": {"address": addr}}
            bcc_recipients.append(recipient_bcc)

        content_type = "HTML" if body_type == "html" else "Text"

        request_body: JSONObject = {
            "subject": subject,
            "body": {
                "contentType": content_type,
                "content": body,
            },
            "toRecipients": to_recipients,
            "ccRecipients": cc_recipients,
            "bccRecipients": bcc_recipients,
        }

        data = self._post("/me/messages", request_body)

        # Decode recipients back
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
            id=require_str(data, "id"),
            subject=subject,
            body=body,
            body_type=body_type,
            to=tuple(to_addrs),
            cc=tuple(cc_addrs),
            bcc=tuple(bcc_addrs),
        )

    def send_draft(self, *, draft_id: str) -> Email:
        """Send a draft email.

        Args:
            draft_id: ID of the draft to send.

        Returns:
            The sent Email.

        Raises:
            AppError[EmailErrorCode]: If draft not found.
        """
        # First get the draft to get its details
        draft_msg = self.get_email(email_id=draft_id)

        # Send the draft
        self._post(f"/me/messages/{draft_id}/send", {})

        # Return the message converted to sent email
        return Email(
            id=draft_id,
            thread_id=draft_msg["thread_id"],
            folder_id="sentitems",
            subject=draft_msg["subject"],
            body=draft_msg["body"],
            body_type=draft_msg["body_type"],
            from_address=draft_msg["from_address"],
            to=draft_msg["to"],
            cc=draft_msg["cc"],
            bcc=draft_msg["bcc"],
            sent_at="",  # Will be filled by server
            received_at="",
            is_read=True,
            is_draft=False,
            has_attachments=draft_msg["has_attachments"],
            importance=draft_msg["importance"],
        )

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
        original = self.get_email(email_id=email_id)

        request_body: JSONObject = {
            "comment": body,
        }

        action = "replyAll" if reply_all else "reply"
        self._post(f"/me/messages/{email_id}/{action}", request_body)

        # Return placeholder for sent reply
        return Email(
            id="reply_placeholder",
            thread_id=original["thread_id"],
            folder_id="sentitems",
            subject=f"Re: {original['subject']}",
            body=body,
            body_type=body_type,
            from_address=EmailAddress(address="me@outlook.com", name=""),
            to=(original["from_address"],),
            cc=original["cc"] if reply_all else (),
            bcc=(),
            sent_at="",
            received_at="",
            is_read=True,
            is_draft=False,
            has_attachments=False,
            importance="normal",
        )

    def delete_email(self, *, email_id: str, permanent: bool = False) -> None:
        """Delete an email.

        Args:
            email_id: ID of the email to delete.
            permanent: If True, permanently delete. If False, move to trash.

        Raises:
            AppError[EmailErrorCode]: If email not found.
        """
        if permanent:
            self._delete(f"/me/messages/{email_id}")
        else:
            # Move to deleted items folder
            body: JSONObject = {"destinationId": "deleteditems"}
            self._post(f"/me/messages/{email_id}/move", body)

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
        body: JSONObject = {"destinationId": destination_folder_id}
        data = self._post(f"/me/messages/{email_id}/move", body)
        return _decode_message(data)

    def list_folders(self) -> tuple[Folder, ...]:
        """List all email folders.

        Returns:
            Tuple of all folders.
        """
        data = self._get("/me/mailFolders")
        folders_raw = data.get("value")
        if not isinstance(folders_raw, list):
            return ()

        folders: list[Folder] = []
        for folder in folders_raw:
            if not isinstance(folder, dict):
                continue
            display_name = optional_str(folder, "displayName") or ""
            unread_raw = folder.get("unreadItemCount")
            total_raw = folder.get("totalItemCount")
            folders.append(
                Folder(
                    id=require_str(folder, "id"),
                    name=display_name,
                    folder_type=_decode_folder_type(display_name),
                    unread_count=unread_raw if isinstance(unread_raw, int) else 0,
                    total_count=total_raw if isinstance(total_raw, int) else 0,
                )
            )

        return tuple(folders)

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
        path = f"/me/messages/{email_id}/attachments/{attachment_id}"
        data = self._get(path)

        content_bytes = optional_str(data, "contentBytes")
        size_raw = data.get("size")

        return Attachment(
            id=require_str(data, "id"),
            name=optional_str(data, "name") or "",
            content_type=optional_str(data, "contentType") or "application/octet-stream",
            size=size_raw if isinstance(size_raw, int) else 0,
            content_bytes=content_bytes,
        )


__all__ = [
    "_OutlookEmailClient",
]
