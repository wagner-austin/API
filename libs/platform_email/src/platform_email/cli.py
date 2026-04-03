#!/usr/bin/env python
"""Email CLI - Check and send emails via Outlook."""

from __future__ import annotations

import argparse
import base64
import hashlib
import mimetypes
import os.path
import secrets
import urllib.parse
from datetime import datetime
from typing import TypedDict

from platform_core.json_utils import JSONObject, JSONValue, load_json_str, narrow_json_to_dict

from platform_email.config.outlook import (
    OUTLOOK_EMAIL_SCOPES,
    outlook_auth_url,
    outlook_token_url,
)
from platform_email.testing import hooks

# =============================================================================
# Styles (for rich console output)
# =============================================================================

STYLE_HEADER = "bold cyan"
STYLE_FROM = "bold yellow"
STYLE_SUBJECT = "white"
STYLE_DATE = "green"
STYLE_UNREAD = "bold white"
STYLE_READ = "dim white"
STYLE_ERROR = "bold red"
STYLE_SUCCESS = "bold green"
STYLE_DIM = "dim"
STYLE_FOLDER = "cyan"

# =============================================================================
# Token Types
# =============================================================================

# Use "common" tenant for multi-tenant support
MICROSOFT_AUTH_URL: str = outlook_auth_url("common")
MICROSOFT_TOKEN_URL: str = outlook_token_url("common")


class TokenRefreshResponse(TypedDict):
    """Response from Microsoft OAuth token refresh endpoint."""

    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str


def require_str(data: JSONObject, key: str) -> str:
    """Require a string value from a JSON object.

    Args:
        data: JSON object to extract from.
        key: Key to look up.

    Returns:
        String value.

    Raises:
        KeyError: If key not found.
        TypeError: If value is not a string.
    """
    value = data[key]
    if not isinstance(value, str):
        msg = f"Expected str for {key}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def require_int(data: JSONObject, key: str) -> int:
    """Require an int value from a JSON object.

    Args:
        data: JSON object to extract from.
        key: Key to look up.

    Returns:
        Int value.

    Raises:
        KeyError: If key not found.
        TypeError: If value is not an int.
    """
    value = data[key]
    if not isinstance(value, int):
        msg = f"Expected int for {key}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def decode_token_response(data: JSONObject) -> TokenRefreshResponse:
    """Decode token response from JSON.

    Args:
        data: JSON object from token endpoint.

    Returns:
        Typed TokenRefreshResponse.

    Raises:
        KeyError: If required field missing.
        TypeError: If field has wrong type.
    """
    return TokenRefreshResponse(
        access_token=require_str(data, "access_token"),
        refresh_token=require_str(data, "refresh_token"),
        expires_in=require_int(data, "expires_in"),
        token_type=require_str(data, "token_type"),
    )


# =============================================================================
# Account Config
# =============================================================================


class Account:
    """Account configuration."""

    def __init__(
        self,
        name: str,
        email: str,
        token_env: str,
        refresh_token_env: str,
        expires_at_env: str,
        client_id_env: str,
        client_secret_env: str,
    ) -> None:
        """Initialize account.

        Args:
            name: Display name for the account.
            email: Email address associated with the account.
            token_env: Environment variable name for access token.
            refresh_token_env: Environment variable name for refresh token.
            expires_at_env: Environment variable name for token expiration.
            client_id_env: Environment variable name for client ID.
            client_secret_env: Environment variable name for client secret.
        """
        self.name = name
        self.email = email
        self.token_env = token_env
        self.refresh_token_env = refresh_token_env
        self.expires_at_env = expires_at_env
        self.client_id_env = client_id_env
        self.client_secret_env = client_secret_env


# Default account - user can customize
ACCOUNTS = [
    Account(
        name="Outlook",
        email="",  # Will be filled after auth
        token_env="OUTLOOK_ACCESS_TOKEN",
        refresh_token_env="OUTLOOK_REFRESH_TOKEN",
        expires_at_env="OUTLOOK_TOKEN_EXPIRES_AT",
        client_id_env="OUTLOOK_CLIENT_ID",
        client_secret_env="OUTLOOK_CLIENT_SECRET",
    ),
]


# =============================================================================
# Environment Helpers
# =============================================================================


def _get_env(key: str) -> str | None:
    """Get environment variable.

    Args:
        key: Environment variable name.

    Returns:
        Value if found, None otherwise.
    """
    return hooks.cli_get_env(key)


def _set_env(key: str, value: str) -> None:
    """Set environment variable in cache.

    Args:
        key: Environment variable name.
        value: Value to set.
    """
    hooks.cli_set_env(key, value)


def _get_now() -> datetime:
    """Get current datetime.

    Returns:
        Current datetime.
    """
    return hooks.cli_get_now()


def _is_token_expired(expires_at_str: str) -> bool:
    """Check if token is expired or will expire within 60 seconds.

    Args:
        expires_at_str: Unix timestamp as string.

    Returns:
        True if token is expired or expiring soon.
    """
    expires_at = int(expires_at_str)
    current_time = int(_get_now().timestamp())
    buffer_seconds = 60
    return current_time >= (expires_at - buffer_seconds)


# =============================================================================
# PKCE Helpers
# =============================================================================


def _generate_code_verifier() -> str:
    """Generate a PKCE code verifier.

    Returns:
        Random code verifier string.
    """
    return secrets.token_urlsafe(64)[:128]


def _generate_code_challenge(verifier: str) -> str:
    """Generate a PKCE code challenge from verifier.

    Args:
        verifier: Code verifier string.

    Returns:
        Base64url-encoded SHA256 hash.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# =============================================================================
# Token Operations
# =============================================================================


def _refresh_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> TokenRefreshResponse:
    """Refresh an access token using the refresh token.

    Args:
        client_id: OAuth client ID.
        client_secret: OAuth client secret.
        refresh_token: Refresh token.

    Returns:
        TokenRefreshResponse with new access token.

    Raises:
        OSError: If refresh fails.
        KeyError: If response missing required fields.
        TypeError: If response fields have wrong types.
    """
    body_params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": " ".join(OUTLOOK_EMAIL_SCOPES),
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = hooks.http_post(MICROSOFT_TOKEN_URL, headers, body)
    raw_value = load_json_str(response)
    data = narrow_json_to_dict(raw_value)
    return decode_token_response(data)


def _exchange_code_for_tokens(
    client_id: str,
    client_secret: str,
    code: str,
    code_verifier: str,
    redirect_uri: str,
) -> TokenRefreshResponse:
    """Exchange authorization code for tokens.

    Args:
        client_id: OAuth client ID.
        client_secret: OAuth client secret.
        code: Authorization code from user.
        code_verifier: PKCE code verifier.
        redirect_uri: Redirect URI used in auth request.

    Returns:
        TokenRefreshResponse with tokens.

    Raises:
        OSError: If exchange fails.
        KeyError: If response missing required fields.
        TypeError: If response fields have wrong types.
    """
    body_params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "scope": " ".join(OUTLOOK_EMAIL_SCOPES),
    }
    body = urllib.parse.urlencode(body_params)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = hooks.http_post(MICROSOFT_TOKEN_URL, headers, body)
    raw_value = load_json_str(response)
    data = narrow_json_to_dict(raw_value)
    return decode_token_response(data)


def _get_valid_token_for_account(account: Account) -> str | None:
    """Get a valid access token for an account, refreshing if expired.

    Args:
        account: Account to get token for.

    Returns:
        Valid access token, or None if no token configured.
    """
    access_token = _get_env(account.token_env)
    if not access_token:
        return None

    refresh_token = _get_env(account.refresh_token_env)
    expires_at = _get_env(account.expires_at_env)

    # If we have expiration info and token is expired, refresh
    if expires_at and refresh_token and _is_token_expired(expires_at):
        client_id = _get_env(account.client_id_env)
        client_secret = _get_env(account.client_secret_env)

        if client_id and client_secret:
            response = _refresh_token(client_id, client_secret, refresh_token)
            access_token = response["access_token"]
            new_expires_at = int(_get_now().timestamp()) + response["expires_in"]

            # Update cache with new token
            _set_env(account.token_env, access_token)
            _set_env(account.refresh_token_env, response["refresh_token"])
            _set_env(account.expires_at_env, str(new_expires_at))

    return access_token


def _get_token() -> str | None:
    """Get valid token for the default account.

    Returns:
        Valid access token if found, None otherwise.
    """
    return _get_valid_token_for_account(ACCOUNTS[0])


# =============================================================================
# Console Helpers
# =============================================================================


def _print(message: str) -> None:
    """Print message to console.

    Args:
        message: Message to print.
    """
    hooks.console_output(message)


def _input(prompt: str) -> str:
    """Get input from user.

    Args:
        prompt: Prompt to display.

    Returns:
        User input string.
    """
    return hooks.console_input(prompt)


# =============================================================================
# API Helpers
# =============================================================================

GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"


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
