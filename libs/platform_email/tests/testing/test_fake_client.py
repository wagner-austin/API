"""Tests for FakeEmailClient in platform_email.testing module."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, EmailErrorCode

from platform_email.fake_hooks import (
    make_fake_attachment,
    make_fake_email,
    make_fake_folder,
)
from platform_email.fakes import (
    FakeEmailClient,
)


class TestFakeEmailClientSendEmail:
    """Tests for FakeEmailClient.send_email()."""

    def test_send_email_creates_email_with_id(self) -> None:
        """Test that send_email creates an email with unique ID."""
        client = FakeEmailClient()
        email = client.send_email(
            to=("recipient@test.com",),
            subject="Test Subject",
            body="Test body",
        )
        assert email["id"].startswith("fake_email_")
        assert email["subject"] == "Test Subject"
        assert email["body"] == "Test body"
        assert email["folder_id"] == "sent"

    def test_send_email_with_cc_and_bcc(self) -> None:
        """Test sending email with CC and BCC recipients."""
        client = FakeEmailClient()
        email = client.send_email(
            to=("to@test.com",),
            subject="Subject",
            body="Body",
            cc=("cc@test.com",),
            bcc=("bcc@test.com",),
        )
        assert len(email["to"]) == 1
        assert len(email["cc"]) == 1
        assert len(email["bcc"]) == 1

    def test_send_email_with_html_body(self) -> None:
        """Test sending email with HTML body type."""
        client = FakeEmailClient()
        email = client.send_email(
            to=("recipient@test.com",),
            subject="HTML Email",
            body="<p>Hello</p>",
            body_type="html",
        )
        assert email["body_type"] == "html"

    def test_send_email_with_attachments(self) -> None:
        """Test sending email with attachments."""
        client = FakeEmailClient()
        attachment = make_fake_attachment(attachment_id="att1")
        email = client.send_email(
            to=("recipient@test.com",),
            subject="With Attachment",
            body="See attachment",
            attachments=(attachment,),
        )
        assert email["has_attachments"] is True

    def test_send_email_tracks_sent_emails(self) -> None:
        """Test that sent emails are tracked."""
        client = FakeEmailClient()
        client.send_email(to=("a@test.com",), subject="First", body="First email")
        client.send_email(to=("b@test.com",), subject="Second", body="Second email")
        sent = client.get_sent_emails()
        assert len(sent) == 2
        assert sent[0]["subject"] == "First"
        assert sent[1]["subject"] == "Second"


class TestFakeEmailClientGetEmail:
    """Tests for FakeEmailClient.get_email()."""

    def test_get_email_returns_added_email(self) -> None:
        """Test getting an email that was added."""
        client = FakeEmailClient()
        email = make_fake_email(email_id="test123", subject="Test")
        client.add_email(email)

        result = client.get_email(email_id="test123")
        assert result["id"] == "test123"
        assert result["subject"] == "Test"

    def test_get_email_raises_for_not_found(self) -> None:
        """Test that get_email raises for nonexistent email."""
        client = FakeEmailClient()
        with pytest.raises(AppError) as exc_info:
            client.get_email(email_id="nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND


class TestFakeEmailClientListEmails:
    """Tests for FakeEmailClient.list_emails()."""

    def test_list_emails_returns_all_emails(self) -> None:
        """Test listing all emails."""
        client = FakeEmailClient()
        client.add_email(make_fake_email(email_id="e1"))
        client.add_email(make_fake_email(email_id="e2"))

        result = client.list_emails()
        assert len(result["emails"]) == 2

    def test_list_emails_filters_by_folder(self) -> None:
        """Test filtering emails by folder."""
        client = FakeEmailClient()
        client.add_email(make_fake_email(email_id="e1", folder_id="inbox"))
        client.add_email(make_fake_email(email_id="e2", folder_id="sent"))

        result = client.list_emails(folder_id="inbox")
        assert len(result["emails"]) == 1
        assert result["emails"][0]["folder_id"] == "inbox"

    def test_list_emails_filters_by_query(self) -> None:
        """Test filtering emails by search query."""
        client = FakeEmailClient()
        client.add_email(make_fake_email(email_id="e1", subject="Hello World"))
        client.add_email(make_fake_email(email_id="e2", subject="Goodbye"))

        result = client.list_emails(query="hello")
        assert len(result["emails"]) == 1
        assert "Hello" in result["emails"][0]["subject"]

    def test_list_emails_respects_max_results(self) -> None:
        """Test that max_results limits results."""
        client = FakeEmailClient()
        for i in range(10):
            client.add_email(make_fake_email(email_id=f"e{i}"))

        result = client.list_emails(max_results=3)
        assert len(result["emails"]) == 3


class TestFakeEmailClientSearchEmails:
    """Tests for FakeEmailClient.search_emails()."""

    def test_search_emails_returns_matches(self) -> None:
        """Test searching emails returns matching results."""
        client = FakeEmailClient()
        client.add_email(make_fake_email(email_id="e1", subject="Important meeting"))
        client.add_email(make_fake_email(email_id="e2", subject="Random stuff"))

        result = client.search_emails(query="important")
        assert len(result) == 1


class TestFakeEmailClientDrafts:
    """Tests for FakeEmailClient draft operations."""

    def test_create_draft_returns_draft(self) -> None:
        """Test creating a draft."""
        client = FakeEmailClient()
        draft = client.create_draft(
            to=("recipient@test.com",),
            subject="Draft Subject",
            body="Draft body",
        )
        assert draft["id"].startswith("fake_draft_")
        assert draft["subject"] == "Draft Subject"

    def test_send_draft_returns_email(self) -> None:
        """Test sending a draft."""
        client = FakeEmailClient()
        draft = client.create_draft(
            to=("recipient@test.com",),
            subject="Draft to Send",
            body="Will be sent",
        )
        email = client.send_draft(draft_id=draft["id"])
        assert email["subject"] == "Draft to Send"
        assert email["is_draft"] is False

    def test_send_draft_raises_for_not_found(self) -> None:
        """Test that send_draft raises for nonexistent draft."""
        client = FakeEmailClient()
        with pytest.raises(AppError) as exc_info:
            client.send_draft(draft_id="nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.DRAFT_NOT_FOUND


class TestFakeEmailClientReplyToEmail:
    """Tests for FakeEmailClient.reply_to_email()."""

    def test_reply_to_email_creates_reply(self) -> None:
        """Test replying to an email."""
        client = FakeEmailClient()
        original = make_fake_email(email_id="original", subject="Original")
        client.add_email(original)

        reply = client.reply_to_email(
            email_id="original",
            body="This is my reply",
        )
        assert "Re: Original" in reply["subject"]

    def test_reply_to_email_raises_for_not_found(self) -> None:
        """Test that reply_to_email raises for nonexistent email."""
        client = FakeEmailClient()
        with pytest.raises(AppError) as exc_info:
            client.reply_to_email(email_id="nonexistent", body="Reply")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND


class TestFakeEmailClientDeleteEmail:
    """Tests for FakeEmailClient.delete_email()."""

    def test_delete_email_tracks_deletion(self) -> None:
        """Test that deleted emails are tracked."""
        client = FakeEmailClient()
        email = make_fake_email(email_id="to_delete")
        client.add_email(email)

        client.delete_email(email_id="to_delete")
        deleted = client.get_deleted_emails()
        assert ("to_delete", False) in deleted

    def test_delete_email_permanent_flag(self) -> None:
        """Test permanent deletion flag."""
        client = FakeEmailClient()
        email = make_fake_email(email_id="to_delete_perm")
        client.add_email(email)

        client.delete_email(email_id="to_delete_perm", permanent=True)
        deleted = client.get_deleted_emails()
        assert ("to_delete_perm", True) in deleted

    def test_delete_email_raises_for_not_found(self) -> None:
        """Test that delete_email raises for nonexistent email."""
        client = FakeEmailClient()
        with pytest.raises(AppError) as exc_info:
            client.delete_email(email_id="nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND


class TestFakeEmailClientMoveEmail:
    """Tests for FakeEmailClient.move_email()."""

    def test_move_email_changes_folder(self) -> None:
        """Test moving an email to a different folder."""
        client = FakeEmailClient()
        email = make_fake_email(email_id="to_move", folder_id="inbox")
        client.add_email(email)

        moved = client.move_email(email_id="to_move", destination_folder_id="archive")
        assert moved["folder_id"] == "archive"

    def test_move_email_tracks_movement(self) -> None:
        """Test that moved emails are tracked."""
        client = FakeEmailClient()
        email = make_fake_email(email_id="to_move")
        client.add_email(email)

        client.move_email(email_id="to_move", destination_folder_id="sent")
        moved = client.get_moved_emails()
        assert ("to_move", "sent") in moved

    def test_move_email_raises_for_not_found(self) -> None:
        """Test that move_email raises for nonexistent email."""
        client = FakeEmailClient()
        with pytest.raises(AppError) as exc_info:
            client.move_email(email_id="nonexistent", destination_folder_id="folder")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND


class TestFakeEmailClientFolders:
    """Tests for FakeEmailClient folder operations."""

    def test_list_folders_returns_added_folders(self) -> None:
        """Test listing folders."""
        client = FakeEmailClient()
        client.add_folder(make_fake_folder(folder_id="inbox", name="Inbox"))
        client.add_folder(make_fake_folder(folder_id="sent", name="Sent"))

        folders = client.list_folders()
        assert len(folders) == 2


class TestFakeEmailClientAttachments:
    """Tests for FakeEmailClient attachment operations."""

    def test_get_attachment_returns_added_attachment(self) -> None:
        """Test getting an attachment."""
        client = FakeEmailClient()
        email = make_fake_email(email_id="email_with_att")
        client.add_email(email)
        attachment = make_fake_attachment(attachment_id="att1", name="doc.pdf")
        client.add_attachment("email_with_att", attachment)

        result = client.get_attachment(email_id="email_with_att", attachment_id="att1")
        assert result["name"] == "doc.pdf"

    def test_get_attachment_raises_for_no_attachments(self) -> None:
        """Test that get_attachment raises when email has no attachments."""
        client = FakeEmailClient()
        with pytest.raises(AppError) as exc_info:
            client.get_attachment(email_id="no_attachments", attachment_id="att1")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND

    def test_get_attachment_raises_for_not_found(self) -> None:
        """Test that get_attachment raises for nonexistent attachment."""
        client = FakeEmailClient()
        email = make_fake_email(email_id="email_with_att")
        client.add_email(email)
        client.add_attachment(
            "email_with_att",
            make_fake_attachment(attachment_id="att1"),
        )

        with pytest.raises(AppError) as exc_info:
            client.get_attachment(email_id="email_with_att", attachment_id="nonexistent")
        err: AppError[EmailErrorCode] = exc_info.value
        assert err.code == EmailErrorCode.EMAIL_NOT_FOUND

    def test_add_multiple_attachments_to_same_email(self) -> None:
        """Test adding multiple attachments to the same email."""
        client = FakeEmailClient()
        email = make_fake_email(email_id="multi_att")
        client.add_email(email)
        client.add_attachment("multi_att", make_fake_attachment(attachment_id="att1"))
        client.add_attachment("multi_att", make_fake_attachment(attachment_id="att2"))

        # Both attachments should be retrievable
        att1 = client.get_attachment(email_id="multi_att", attachment_id="att1")
        att2 = client.get_attachment(email_id="multi_att", attachment_id="att2")
        assert att1["id"] == "att1"
        assert att2["id"] == "att2"


class TestFakeEmailClientAddDraft:
    """Tests for FakeEmailClient.add_draft()."""

    def test_add_draft_then_send(self) -> None:
        """Test adding a draft and then sending it."""
        from platform_email.fake_hooks import (
            make_fake_draft,
        )

        client = FakeEmailClient()
        draft = make_fake_draft(draft_id="pre_added_draft", subject="Pre-added Draft")
        client.add_draft(draft)

        # Now send the draft
        email = client.send_draft(draft_id="pre_added_draft")
        assert email["subject"] == "Pre-added Draft"
        assert email["is_draft"] is False


class TestFakeEmailClientCreateDraftWithCcBcc:
    """Tests for FakeEmailClient.create_draft() with CC and BCC."""

    def test_create_draft_with_cc(self) -> None:
        """Test creating draft with CC recipients."""
        client = FakeEmailClient()
        draft = client.create_draft(
            to=("to@test.com",),
            subject="Draft with CC",
            body="Body",
            cc=("cc1@test.com", "cc2@test.com"),
        )
        assert len(draft["cc"]) == 2
        assert draft["cc"][0]["address"] == "cc1@test.com"
        assert draft["cc"][1]["address"] == "cc2@test.com"

    def test_create_draft_with_bcc(self) -> None:
        """Test creating draft with BCC recipients."""
        client = FakeEmailClient()
        draft = client.create_draft(
            to=("to@test.com",),
            subject="Draft with BCC",
            body="Body",
            bcc=("bcc@test.com",),
        )
        assert len(draft["bcc"]) == 1
        assert draft["bcc"][0]["address"] == "bcc@test.com"


class TestFakeEmailClientReplyAll:
    """Tests for FakeEmailClient.reply_to_email() with reply_all."""

    def test_reply_all_includes_original_recipients(self) -> None:
        """Test that reply_all includes original To and CC recipients."""

        client = FakeEmailClient()
        # Create an email with multiple recipients
        original = make_fake_email(
            email_id="original",
            subject="Group Email",
            from_address="sender@test.com",
            to=("me@test.com", "other@test.com"),  # me won't be in CC (excluded)
            cc=("cc1@test.com",),
        )
        # Update the to addresses to have the correct structure
        client._emails["original"] = original

        reply = client.reply_to_email(
            email_id="original",
            body="Reply to all",
            reply_all=True,
        )

        # Should reply to sender
        assert reply["to"][0]["address"] == "sender@test.com"
        # CC should include original to recipients (except test@example.com)
        # and original CC recipients
        cc_addresses = [addr["address"] for addr in reply["cc"]]
        assert "me@test.com" in cc_addresses
        assert "other@test.com" in cc_addresses
        assert "cc1@test.com" in cc_addresses

    def test_reply_all_excludes_test_at_example_com(self) -> None:
        """Test that reply_all excludes test@example.com from CC."""
        client = FakeEmailClient()
        # Create an email where one recipient is test@example.com
        original = make_fake_email(
            email_id="original2",
            subject="Test Email",
            from_address="sender@test.com",
            to=("test@example.com", "other@test.com"),
            cc=(),
        )
        client._emails["original2"] = original

        reply = client.reply_to_email(
            email_id="original2",
            body="Reply",
            reply_all=True,
        )

        # CC should NOT include test@example.com
        cc_addresses = [addr["address"] for addr in reply["cc"]]
        assert "test@example.com" not in cc_addresses
        assert "other@test.com" in cc_addresses
