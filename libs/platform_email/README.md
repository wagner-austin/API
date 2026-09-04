# platform_email

Multi-provider email library supporting Microsoft Outlook (Graph API) and Gmail, with a CLI for managing emails across accounts.

## Installation

### Within this monorepo

Add to your package's `pyproject.toml`:

```toml
[tool.poetry.dependencies]
platform-email = { path = "../platform_email", develop = true }
```

Then install:

```bash
poetry install
```

### External (if published to PyPI)

```bash
poetry add platform-email
```

## Features

- **Multi-Provider Support**: Microsoft Outlook (Graph API) and Gmail
- **OAuth 2.0 Authentication**: Secure PKCE-based authentication with automatic token refresh
- **Strict Typing**: mypy strict mode, no Any, 100% test coverage
- **Testable Design**: Hooks-based dependency injection for full testability
- **CLI Tool**: Terminal interface for listing, reading, and sending emails

## Dependencies

- `platform-core` - OAuth types, PKCE utilities, JSON utilities, error handling

## Setup

### Microsoft Outlook (Graph API)

#### 1. Create Azure App Registration

1. Go to https://portal.azure.com → "App registrations" → "New registration"
2. Name: "Platform Email" (or your choice)
3. Supported account types: "Accounts in any organizational directory and personal Microsoft accounts"
4. Redirect URI: Select "Web" → `http://localhost`

#### 2. Configure API Permissions

1. Go to "API permissions" → "Add a permission" → "Microsoft Graph" → "Delegated permissions"
2. Add: `Mail.Read`, `Mail.Send`, `Mail.ReadWrite`, `offline_access`

#### 3. Create Client Secret

1. Go to "Certificates & secrets" → "New client secret"
2. Copy the **Value** (not the Secret ID)

#### 4. Create .env File

Create `.env` in the package directory:

```bash
OUTLOOK_CLIENT_ID=your-application-client-id
OUTLOOK_CLIENT_SECRET=your-client-secret-value
```

#### 5. Authenticate

Run the auth command:

```bash
cd libs/platform_email
poetry run python -m platform_email.cli auth
```

This will:
1. Generate an authorization URL
2. Open your browser to sign in
3. Redirect to `http://localhost?code=...` (page won't load, that's expected)
4. Copy the `code` parameter from the URL and paste it back
5. Save tokens to `.env` for future use

### Gmail

#### 1. Create Google Cloud Project

1. Go to https://console.cloud.google.com/
2. Create new project
3. Enable "Gmail API" in APIs & Services

#### 2. Create OAuth 2.0 Credentials

1. Go to APIs & Services → Credentials
2. Create Credentials → OAuth client ID
3. Application type: "Desktop app"
4. Download JSON → save as `~/.google/email_credentials.json`

## CLI Usage

The CLI provides commands for managing emails via Microsoft Outlook.

### Commands

```bash
# Authenticate with Outlook
poetry run python -m platform_email.cli auth

# List email folders
poetry run python -m platform_email.cli folders

# List recent emails (default: inbox, 10 emails)
poetry run python -m platform_email.cli list
poetry run python -m platform_email.cli list --folder sent --count 20

# Read an email by index
poetry run python -m platform_email.cli read 1
poetry run python -m platform_email.cli read 3

# Send an email (body must be in a file to prevent shell truncation)
poetry run python -m platform_email.cli send recipient@example.com "Subject" body.txt
poetry run python -m platform_email.cli send recipient@example.com "Subject" body.txt --cc a@b.com,c@d.com
poetry run python -m platform_email.cli send recipient@example.com "Subject" body.html --html --bcc secret@x.com

# Send with file attachments (repeatable)
poetry run python -m platform_email.cli send recipient@example.com "Subject" body.txt --attach report.pdf
poetry run python -m platform_email.cli send recipient@example.com "Subject" body.txt --attach file1.pdf --attach file2.zip

# Search emails by keyword
poetry run python -m platform_email.cli search "TU+11"
poetry run python -m platform_email.cli search "invoice" -n 20
```

### CLI Configuration

The CLI reads OAuth tokens from environment variables. Create a `.env` file in the package directory:

```bash
# OAuth Credentials (required)
OUTLOOK_CLIENT_ID=your_client_id
OUTLOOK_CLIENT_SECRET=your_client_secret

# Tokens (saved automatically after auth)
OUTLOOK_ACCESS_TOKEN=EwA...
OUTLOOK_REFRESH_TOKEN=M.C...
OUTLOOK_TOKEN_EXPIRES_AT=1735200000
```

### Automatic Token Refresh

The CLI automatically refreshes expired access tokens using the refresh token. When a token expires:

1. The CLI checks `TOKEN_EXPIRES_AT` before each API call
2. If expired (or expiring within 60 seconds), it uses the refresh token to get a new access token
3. The new token is cached in memory for subsequent calls

This means you only need to authenticate once - the CLI handles token refresh automatically.

### CLI Output

The CLI uses Rich-style formatting with color-coded output:
- **Headers**: Bold cyan
- **From addresses**: Bold yellow
- **Dates**: Green
- **Unread emails**: Bold white with asterisk marker
- **Read emails**: Dim white
- **Errors**: Bold red
- **Success messages**: Bold green

## Quick Start (Python API)

### Outlook Client

```python
from platform_email import (
    outlook_email_client,
    OutlookOAuthConfig,
    OAuthTokens,
)

# Create client with tokens
tokens = OAuthTokens(
    access_token="your_access_token",
    refresh_token="your_refresh_token",
    expires_at=1735200000,
    token_type="Bearer",
)

client = outlook_email_client(tokens=tokens)

# List folders
folders = client.list_folders()
for folder in folders:
    print(f"{folder['name']}: {folder['unread_count']} unread")

# List emails
result = client.list_emails(folder_id="inbox", max_results=10)
for email in result["emails"]:
    print(f"From: {email['from_address']['address']}")
    print(f"Subject: {email['subject']}")

# Read a specific email
email = client.get_email(email_id="AAMk...")
print(f"Body: {email['body']}")

# Send email
email = client.send_email(
    to=("recipient@example.com",),
    subject="Hello from Platform Email",
    body="This is a test email.",
    body_type="text",
)

# Send with attachments
from platform_email import Attachment
import base64

attachment = Attachment(
    id="",
    name="document.pdf",
    content_type="application/pdf",
    size=1024,
    content_bytes=base64.b64encode(pdf_bytes).decode(),
)

email = client.send_email(
    to=("recipient@example.com",),
    subject="With attachment",
    body="See attached.",
    attachments=(attachment,),
)
```

### Gmail Client

```python
from platform_email import gmail_email_client, OAuthTokens

tokens = OAuthTokens(
    access_token="your_access_token",
    refresh_token="your_refresh_token",
    expires_at=1735200000,
    token_type="Bearer",
)

client = gmail_email_client(tokens=tokens)

# Same interface as Outlook
folders = client.list_folders()
result = client.list_emails(max_results=10)
```

## Email Client Protocol

Both Outlook and Gmail clients implement the same protocol:

```python
class EmailClientProtocol(Protocol):
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
    ) -> Email: ...

    def get_email(self, *, email_id: str) -> Email: ...

    def list_emails(
        self,
        *,
        folder_id: str | None = None,
        query: str | None = None,
        max_results: int = 50,
        page_token: str | None = None,
    ) -> EmailListResult: ...

    def search_emails(self, *, query: str, max_results: int = 50) -> tuple[Email, ...]: ...

    def create_draft(self, *, to: tuple[str, ...], subject: str, body: str) -> Email: ...

    def send_draft(self, *, draft_id: str) -> Email: ...

    def reply_to_email(self, *, email_id: str, body: str, reply_all: bool = False) -> Email: ...

    def delete_email(self, *, email_id: str, permanent: bool = False) -> None: ...

    def move_email(self, *, email_id: str, destination_folder_id: str) -> Email: ...

    def list_folders(self) -> tuple[Folder, ...]: ...

    def get_attachment(self, *, email_id: str, attachment_id: str) -> Attachment: ...
```

## Testing

The library uses a hooks-based design for full testability without mocking:

```python
from platform_email import (
    hooks,
    reset_hooks,
    FakeEmailClient,
    make_fake_email,
    make_fake_folder,
    make_fake_http_get,
    make_fake_http_post,
)

# Use fake client for testing
client = FakeEmailClient()
client.add_folder(make_fake_folder(folder_id="inbox", name="Inbox"))
client.add_email(
    make_fake_email(
        email_id="msg1",
        subject="Test Email",
        from_address={"address": "sender@example.com", "name": "Sender"},
    )
)

# List emails with fake client
result = client.list_emails(folder_id="inbox")
assert len(result["emails"]) == 1

# Use fake HTTP responses for real client testing
hooks.http_get = make_fake_http_get('{"value": []}')
hooks.http_post = make_fake_http_post('{"id": "new123"}')

# Reset to production hooks
reset_hooks()
```

### Hooks Architecture

The `HooksContainer` provides dependency injection for all external dependencies:

| Hook | Type | Description |
|------|------|-------------|
| `http_get` | `HttpGetHook` | HTTP GET requests |
| `http_post` | `HttpPostHook` | HTTP POST requests |
| `http_patch` | `HttpPatchHook` | HTTP PATCH requests |
| `http_delete` | `HttpDeleteHook` | HTTP DELETE requests |
| `load_outlook_tokens` | `LoadOutlookTokensHook` | Load Outlook OAuth tokens |
| `save_outlook_tokens` | `SaveOutlookTokensHook` | Save Outlook OAuth tokens |
| `load_outlook_config` | `LoadOutlookConfigHook` | Load Outlook OAuth config |
| `load_gmail_tokens` | `LoadGmailTokensHook` | Load Gmail OAuth tokens |
| `save_gmail_tokens` | `SaveGmailTokensHook` | Save Gmail OAuth tokens |
| `load_gmail_credentials` | `LoadGmailCredentialsHook` | Load Gmail OAuth credentials |
| `open_browser` | `OpenBrowserHook` | Open browser for OAuth |
| `current_time` | `CurrentTimeHook` | Get current timestamp |
| `read_file` | `ReadFileHook` | Read file contents (text) |
| `read_file_bytes` | `ReadFileBytesHook` | Read file contents (binary) |
| `write_file` | `WriteFileHook` | Write file contents |
| `file_exists` | `FileExistsHook` | Check file existence |
| `console_output` | `ConsoleOutputHook` | Console output |
| `console_input` | `ConsoleInputHook` | Console input |
| `cli_get_env` | `CliGetEnvHook` | CLI environment variables |
| `cli_set_env` | `CliSetEnvHook` | CLI environment variable setter |
| `cli_get_now` | `CliGetNowHook` | CLI current datetime |

### Testing Helpers

| Helper | Description |
|--------|-------------|
| `FakeEmailClient` | In-memory email client implementing full protocol |
| `make_fake_http_get(response)` | Returns fixed response for GET requests |
| `make_fake_http_post(response)` | Returns fixed response for POST requests |
| `make_fake_http_patch(response)` | Returns fixed response for PATCH requests |
| `make_fake_http_delete()` | No-op for DELETE requests |
| `make_raising_http_get(error)` | Raises specified exception on GET |
| `make_raising_http_post(error)` | Raises specified exception on POST |
| `make_fake_email(...)` | Create an Email with defaults |
| `make_fake_folder(...)` | Create a Folder with defaults |
| `make_fake_attachment(...)` | Create an Attachment with defaults |
| `make_fake_tokens(tokens)` | Returns fixed tokens |
| `make_fake_outlook_config(config)` | Returns fixed Outlook config |
| `reset_hooks()` | Reset all hooks to production implementations |

## API Reference

### Types

| Type | Description |
|------|-------------|
| `Email` | Email message with id, subject, body, from/to/cc/bcc addresses, attachments |
| `EmailAddress` | Email address with `address` and `name` fields |
| `EmailListResult` | List of emails with optional `next_page_token` |
| `Folder` | Email folder with id, name, folder_type, unread/total counts |
| `FolderType` | Literal type: "inbox", "sent", "drafts", "trash", "spam", "archive", "custom" |
| `Attachment` | File attachment with id, name, content_type, size, content_bytes |
| `BodyType` | Literal type: "text", "html" |
| `Draft` | Draft email |
| `OAuthTokens` | Access/refresh token pair with expiration |
| `OAuthCredentials` | OAuth client credentials |
| `OutlookOAuthConfig` | Outlook-specific OAuth configuration with tenant_id |

### Auth Functions

| Function | Description |
|----------|-------------|
| `outlook_load_or_authorize` | Load cached tokens or run OAuth flow for Outlook |
| `outlook_authorize` | Run Outlook OAuth authorization flow |
| `outlook_build_auth_url` | Build Microsoft OAuth URL |
| `outlook_exchange_code_for_tokens` | Exchange auth code for tokens |
| `outlook_refresh_access_token` | Refresh expired access token |
| `gmail_load_or_authorize` | Load cached tokens or run OAuth flow for Gmail |
| `gmail_authorize` | Run Gmail OAuth authorization flow |

### Client Factory Functions

| Function | Description |
|----------|-------------|
| `outlook_email_client(tokens)` | Create Outlook email client |
| `gmail_email_client(tokens)` | Create Gmail email client |

### CLI Commands

| Command | Description |
|---------|-------------|
| `auth` | Authenticate with Outlook |
| `folders` | List email folders |
| `list` | List recent emails (-f/--folder, -n/--count) |
| `read <index>` | Read email by index from list |
| `send <to> <subject> <body_file>` | Send an email (--cc, --bcc, --html, --attach) |
| `search <query>` | Search emails by keyword (-n/--count) |

### Error Handling

```python
from platform_core.errors import AppError, EmailErrorCode

try:
    client.get_email(email_id="invalid")
except AppError as e:
    if e.code == EmailErrorCode.EMAIL_NOT_FOUND:
        print("Email does not exist")
    elif e.code == EmailErrorCode.AUTH_FAILED:
        print("Need to re-authenticate")
```

| Error Code | Description |
|------------|-------------|
| `CREDENTIALS_NOT_FOUND` | OAuth credentials file missing or invalid |
| `TOKEN_EXPIRED` | OAuth token needs refresh |
| `AUTH_FAILED` | OAuth authentication failed |
| `EMAIL_API_ERROR` | General API error (includes network errors) |
| `EMAIL_NOT_FOUND` | Email ID not found |
| `FOLDER_NOT_FOUND` | Folder ID not found |

## Configuration

### File-Based (Development)

Default paths:

| Path | Description |
|------|-------------|
| `~/.microsoft/email_credentials.json` | Outlook OAuth credentials |
| `~/.microsoft/email_tokens.json` | Outlook access/refresh tokens |
| `~/.google/email_credentials.json` | Gmail OAuth credentials |
| `~/.google/email_tokens.json` | Gmail access/refresh tokens |

### Environment Variables (Production/CLI)

For deployment or CLI usage, configure via environment variables:

**Outlook Credentials** (from Azure Portal):
```bash
OUTLOOK_CLIENT_ID=your_application_client_id
OUTLOOK_CLIENT_SECRET=your_client_secret
```

**Outlook Tokens** (saved after auth):
```bash
OUTLOOK_ACCESS_TOKEN=EwA...
OUTLOOK_REFRESH_TOKEN=M.C...
OUTLOOK_TOKEN_EXPIRES_AT=1735200000
```

**Gmail Credentials** (from Google Cloud Console):
```bash
GMAIL_CLIENT_ID=your_client_id.apps.googleusercontent.com
GMAIL_CLIENT_SECRET=GOCSPX-your_secret
GMAIL_REDIRECT_URI=http://localhost
```

**Gmail Tokens** (after OAuth authorization):
```bash
GMAIL_ACCESS_TOKEN=ya29.your_access_token
GMAIL_REFRESH_TOKEN=1//your_refresh_token
GMAIL_TOKEN_EXPIRES_AT=1735200000
```

## Development

```bash
cd libs/platform_email
make check  # Run lint + tests
make lint   # Run linting only
make test   # Run tests only
```

## Requirements

- Python 3.11+
- platform-core (OAuth types, PKCE utilities, error handling, JSON utilities)
- 100% test coverage enforced
