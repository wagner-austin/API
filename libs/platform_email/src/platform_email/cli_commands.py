"""Graph API access and command handlers for the email CLI."""

from __future__ import annotations

import base64
import mimetypes
import os.path
import secrets
import urllib.parse

from platform_core.json_utils import JSONObject, JSONValue, load_json_str, narrow_json_to_dict

from platform_email.cli_auth import (
    ACCOUNTS,
    GRAPH_API_BASE,
    MICROSOFT_AUTH_URL,
    STYLE_DATE,
    STYLE_DIM,
    STYLE_ERROR,
    STYLE_FOLDER,
    STYLE_FROM,
    STYLE_HEADER,
    STYLE_READ,
    STYLE_SUCCESS,
    STYLE_UNREAD,
    _exchange_code_for_tokens,
    _generate_code_challenge,
    _generate_code_verifier,
    _get_env,
    _get_now,
    _get_token,
    _input,
    _print,
    _set_env,
)
from platform_email.config.outlook import (
    OUTLOOK_EMAIL_SCOPES,
)
from platform_email.testing import hooks


def _api_get(access_token: str, path: str) -> JSONObject:
    """Make GET request to Microsoft Graph API.

    Args:
        access_token: OAuth access token.
        path: API path (will be appended to base URL).

    Returns:
        Parsed JSON response.
    """
    url = f"{GRAPH_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = hooks.http_get(url, headers)
    raw_value = load_json_str(response)
    return narrow_json_to_dict(raw_value)


def _api_post(access_token: str, path: str, body: JSONObject) -> JSONObject:
    """Make POST request to Microsoft Graph API.

    Args:
        access_token: OAuth access token.
        path: API path.
        body: Request body.

    Returns:
        Parsed JSON response.
    """
    from platform_core.json_utils import dump_json_str

    url = f"{GRAPH_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    response = hooks.http_post(url, headers, dump_json_str(body))
    if not response.strip():
        return {}
    raw_value = load_json_str(response)
    return narrow_json_to_dict(raw_value)


# =============================================================================
# Display Helpers
# =============================================================================


def _format_recipients(addresses: str) -> list[JSONValue]:
    """Convert comma-separated email addresses to Graph API recipient format.

    Args:
        addresses: Comma-separated email addresses (e.g. "a@b.com,c@d.com").
            Empty string produces an empty list.

    Returns:
        List of recipient objects for the Graph API.
    """
    if not addresses:
        return []
    return [
        {"emailAddress": {"address": addr.strip()}} for addr in addresses.split(",") if addr.strip()
    ]


def _build_attachments(paths: tuple[str, ...]) -> list[JSONValue]:
    """Build Graph API attachment objects from file paths.

    Reads each file, base64-encodes its contents, and guesses its MIME type.
    Files must exist (validated via hooks.file_exists before calling).

    Args:
        paths: Tuple of file paths to attach.

    Returns:
        List of fileAttachment objects for the Graph API.
    """
    attachments: list[JSONValue] = []
    for path in paths:
        raw_bytes = hooks.read_file_bytes(path)
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        filename = os.path.basename(path)
        att: JSONObject = {
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": filename,
            "contentType": content_type,
            "contentBytes": encoded,
        }
        attachments.append(att)
    return attachments


def _display_message_rows(messages: list[JSONObject]) -> None:
    """Render email message rows to console.

    Args:
        messages: List of Graph API message objects (pre-validated as dicts).
    """
    for i, msg in enumerate(messages, 1):
        is_read_raw = msg.get("isRead")
        is_read = is_read_raw if isinstance(is_read_raw, bool) else True
        subject_raw = msg.get("subject")
        subject = subject_raw if isinstance(subject_raw, str) else "(no subject)"
        from_email = ""
        from_data = msg.get("from")
        if isinstance(from_data, dict):
            email_addr = from_data.get("emailAddress")
            if isinstance(email_addr, dict):
                addr_raw = email_addr.get("address")
                from_email = addr_raw if isinstance(addr_raw, str) else ""

        received_raw = msg.get("receivedDateTime")
        received = received_raw if isinstance(received_raw, str) else ""
        date_str = received[:10] if received else ""

        style = STYLE_UNREAD if not is_read else STYLE_READ
        unread_marker = "*" if not is_read else " "

        subject_display = subject[:50] if len(subject) > 50 else subject
        _print(f"  {unread_marker}[{STYLE_DIM}]{i:2}.[/] [{style}]{subject_display}[/]")
        _print(f"      [{STYLE_FROM}]{from_email}[/] - [{STYLE_DATE}]{date_str}[/]")


# =============================================================================
# Commands
# =============================================================================


def cmd_auth() -> None:
    """Authorize with Microsoft and save tokens."""
    account = ACCOUNTS[0]

    client_id = _get_env(account.client_id_env)
    client_secret = _get_env(account.client_secret_env)

    if not client_id or not client_secret:
        _print(f"[{STYLE_ERROR}]Missing credentials![/]")
        _print(f"Set {account.client_id_env} and {account.client_secret_env} in your .env file")
        _print("")
        _print("To get these:")
        _print("1. Go to https://portal.azure.com")
        _print("2. Search for 'App registrations' -> New registration")
        _print("3. Name: 'Email CLI', Redirect URI: Web -> http://localhost")
        _print("4. Copy Application (client) ID -> OUTLOOK_CLIENT_ID")
        _print("5. Certificates & secrets -> New client secret -> copy value")
        _print("   Save as OUTLOOK_CLIENT_SECRET")
        _print("6. API permissions -> Add: Mail.Read, Mail.Send, Mail.ReadWrite, offline_access")
        return

    # Generate PKCE values
    code_verifier = _generate_code_verifier()
    code_challenge = _generate_code_challenge(code_verifier)
    state = secrets.token_urlsafe(16)
    redirect_uri = "http://localhost"

    # Build auth URL
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": " ".join(OUTLOOK_EMAIL_SCOPES),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{MICROSOFT_AUTH_URL}?{urllib.parse.urlencode(params)}"

    _print("")
    _print(f"[{STYLE_HEADER}]Open this URL in your browser:[/]")
    _print("")
    _print(auth_url)
    _print("")
    _print("After signing in, you'll be redirected to a URL like:")
    _print("http://localhost/?code=XXXXX&state=YYYYY")
    _print("")

    code = _input("Paste the 'code' value from the URL: ").strip()

    if not code:
        _print(f"[{STYLE_ERROR}]No code provided[/]")
        return

    _print("")
    _print("Exchanging code for tokens...")

    response = _exchange_code_for_tokens(
        client_id, client_secret, code, code_verifier, redirect_uri
    )

    # Save tokens
    expires_at = int(_get_now().timestamp()) + response["expires_in"]
    _set_env(account.token_env, response["access_token"])
    _set_env(account.refresh_token_env, response["refresh_token"])
    _set_env(account.expires_at_env, str(expires_at))

    _print("")
    _print(f"[{STYLE_SUCCESS}]Authorization successful![/]")
    _print("")
    _print("Add these to your .env file to persist tokens:")
    _print(f"  {account.token_env}={response['access_token']}")
    _print(f"  {account.refresh_token_env}={response['refresh_token']}")
    _print(f"  {account.expires_at_env}={expires_at}")


def cmd_folders() -> None:
    """List email folders."""
    token = _get_token()
    if not token:
        _print(f"[{STYLE_ERROR}]Not authenticated. Run 'email auth' first.[/]")
        return

    data = _api_get(token, "/me/mailFolders")
    folders = data.get("value", [])

    if not isinstance(folders, list):
        _print(f"[{STYLE_ERROR}]Invalid response[/]")
        return

    _print("")
    _print(f"[{STYLE_HEADER}]Email Folders:[/]")
    _print("")

    for folder in folders:
        if not isinstance(folder, dict):
            continue
        name = folder.get("displayName", "Unknown")
        unread = folder.get("unreadItemCount", 0)
        total = folder.get("totalItemCount", 0)

        unread_str = f" ({unread} unread)" if unread else ""
        _print(f"  [{STYLE_FOLDER}]{name}[/]{unread_str} - {total} total")


def cmd_list(folder: str = "inbox", count: int = 10) -> None:
    """List recent emails.

    Args:
        folder: Folder name (inbox, sent, drafts, etc).
        count: Number of emails to show.
    """
    token = _get_token()
    if not token:
        _print(f"[{STYLE_ERROR}]Not authenticated. Run 'email auth' first.[/]")
        return

    # Map friendly names to folder paths
    folder_map = {
        "inbox": "inbox",
        "sent": "sentitems",
        "drafts": "drafts",
        "trash": "deleteditems",
        "junk": "junkemail",
    }
    folder_path = folder_map.get(folder.lower(), folder)

    path = f"/me/mailFolders/{folder_path}/messages?$top={count}&$orderby=receivedDateTime%20desc"
    data = _api_get(token, path)
    messages = data.get("value", [])

    if not isinstance(messages, list):
        _print(f"[{STYLE_ERROR}]Invalid response[/]")
        return

    validated: list[JSONObject] = [m for m in messages if isinstance(m, dict)]

    _print("")
    _print(f"[{STYLE_HEADER}]Recent Emails ({folder}):[/]")
    _print("")

    _display_message_rows(validated)


def cmd_read(index: int) -> None:
    """Read an email by index from the inbox.

    Args:
        index: Email index (1-based) from the list command.
    """
    token = _get_token()
    if not token:
        _print(f"[{STYLE_ERROR}]Not authenticated. Run 'email auth' first.[/]")
        return

    # Fetch messages to get the ID
    path = "/me/mailFolders/inbox/messages?$top=20&$orderby=receivedDateTime%20desc"
    data = _api_get(token, path)
    messages = data.get("value", [])

    if not isinstance(messages, list):
        _print(f"[{STYLE_ERROR}]Invalid response[/]")
        return

    if index < 1 or index > len(messages):
        _print(f"[{STYLE_ERROR}]Invalid index. Use 1-{len(messages)}[/]")
        return

    msg = messages[index - 1]
    if not isinstance(msg, dict):
        _print(f"[{STYLE_ERROR}]Invalid message[/]")
        return

    # Get full message
    msg_id = msg.get("id", "")
    full_msg = _api_get(token, f"/me/messages/{msg_id}")

    subject_raw = full_msg.get("subject", "(no subject)")
    subject = subject_raw if isinstance(subject_raw, str) else "(no subject)"
    from_email = ""
    from_name = ""
    from_data = full_msg.get("from")
    if isinstance(from_data, dict):
        email_addr = from_data.get("emailAddress")
        if isinstance(email_addr, dict):
            addr_val = email_addr.get("address")
            name_val = email_addr.get("name")
            from_email = addr_val if isinstance(addr_val, str) else ""
            from_name = name_val if isinstance(name_val, str) else ""

    received_raw = full_msg.get("receivedDateTime", "")
    received_str = received_raw if isinstance(received_raw, str) else ""
    received = received_str[:16].replace("T", " ")

    body_content = ""
    body_data = full_msg.get("body")
    if isinstance(body_data, dict):
        content_raw = body_data.get("content")
        content_type_raw = body_data.get("contentType")
        content_type = content_type_raw if isinstance(content_type_raw, str) else "text"
        body_content = content_raw if isinstance(content_raw, str) else ""
        if content_type == "html":
            # Strip HTML tags (simple approach)
            import re

            body_content = re.sub(r"<[^>]+>", "", body_content)
            body_content = body_content.strip()

    _print("")
    _print(f"[{STYLE_HEADER}]{subject}[/]")
    _print(f"[{STYLE_FROM}]From:[/] {from_name} <{from_email}>")
    _print(f"[{STYLE_DATE}]Date:[/] {received}")
    _print("")
    _print("-" * 60)
    _print("")
    _print(body_content[:2000])  # Limit body length
    if len(body_content) > 2000:
        _print(f"\n[{STYLE_DIM}]... (truncated)[/]")


def cmd_send(
    to: str,
    subject: str,
    body_file: str,
    *,
    cc: str = "",
    bcc: str = "",
    html: bool = False,
    attachments: tuple[str, ...] = (),
) -> None:
    """Send an email with body read from a file.

    Args:
        to: Recipient email address.
        subject: Email subject.
        body_file: Path to file containing email body.
        cc: Comma-separated CC email addresses.
        bcc: Comma-separated BCC email addresses.
        html: If True, send as HTML with body wrapped in <pre> tags.
        attachments: Tuple of file paths to attach.
    """
    token = _get_token()
    if not token:
        _print(f"[{STYLE_ERROR}]Not authenticated. Run 'email auth' first.[/]")
        return

    if not hooks.file_exists(body_file):
        _print(f"[{STYLE_ERROR}]Body file not found: {body_file}[/]")
        return

    for att_path in attachments:
        if not hooks.file_exists(att_path):
            _print(f"[{STYLE_ERROR}]Attachment not found: {att_path}[/]")
            return

    body = hooks.read_file(body_file)

    # Determine content type and format body
    if html:
        content_type = "HTML"
        content = f'<pre style="font-family: inherit;">{body}</pre>'
    else:
        content_type = "Text"
        content = body

    msg_body: JSONObject = {
        "subject": subject,
        "body": {
            "contentType": content_type,
            "content": content,
        },
        "toRecipients": _format_recipients(to),
        "ccRecipients": _format_recipients(cc),
        "bccRecipients": _format_recipients(bcc),
    }

    if attachments:
        msg_body["attachments"] = _build_attachments(attachments)

    message: JSONObject = {"message": msg_body}

    _api_post(token, "/me/sendMail", message)

    parts = [f"Email sent to {to}"]
    if cc:
        parts.append(f"CC: {cc}")
    if bcc:
        parts.append(f"BCC: {bcc}")
    if attachments:
        filenames = [os.path.basename(p) for p in attachments]
        parts.append(f"Attachments: {', '.join(filenames)}")
    _print(f"[{STYLE_SUCCESS}]{' | '.join(parts)}[/]")


def cmd_search(query: str, count: int = 10) -> None:
    """Search emails by keyword.

    Args:
        query: Search query string (uses Microsoft Graph KQL syntax).
        count: Maximum number of results to return.
    """
    token = _get_token()
    if not token:
        _print(f"[{STYLE_ERROR}]Not authenticated. Run 'email auth' first.[/]")
        return

    encoded_query = urllib.parse.quote(query)
    path = f'/me/messages?$search="{encoded_query}"&$top={count}'
    data = _api_get(token, path)
    messages = data.get("value", [])

    if not isinstance(messages, list):
        _print(f"[{STYLE_ERROR}]Invalid response[/]")
        return

    validated: list[JSONObject] = [m for m in messages if isinstance(m, dict)]

    _print("")
    _print(f'[{STYLE_HEADER}]Search Results for "{query}":[/]')
    _print("")

    if not validated:
        _print(f"  [{STYLE_DIM}]No results found[/]")
        return

    _display_message_rows(validated)


# =============================================================================
# Argument Parsing
# =============================================================================
