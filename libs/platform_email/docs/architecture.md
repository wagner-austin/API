# Architecture: platform_email Library

## Overview

The `platform_email` library provides multi-provider email integration supporting Microsoft Outlook (Graph API) and Gmail. It handles OAuth 2.0 authentication, email CRUD operations, folder management, attachment handling, and a CLI for managing emails.

## Dependencies

- `platform-core` — OAuth types/utilities (`OAuthCredentials`, `OAuthTokens`, PKCE functions), JSON helpers (`require_*`, `load_json_str`, `JSONObject`), error codes (`AppError`, `EmailErrorCode`)

No external HTTP libraries — uses Python stdlib (`urllib.request`, `webbrowser`) for OAuth and API calls.

## Directory Structure

```
libs/platform_email/
├── pyproject.toml
├── README.md
├── Makefile
├── .gitignore
├── .env                          # Local config (gitignored)
├── docs/
│   └── architecture.md
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/platform_email/
│   ├── __init__.py               # Public exports
│   ├── py.typed                  # PEP 561 marker
│   ├── client.py                 # Factory functions
│   ├── cli.py                    # Command-line interface
│   ├── auth/
│   │   ├── __init__.py           # Auth exports
│   │   ├── common.py             # Shared auth utilities
│   │   ├── outlook.py            # Microsoft OAuth flow
│   │   └── gmail.py              # Google OAuth flow
│   ├── config/
│   │   ├── __init__.py           # Config exports
│   │   ├── outlook.py            # Microsoft URLs, scopes
│   │   └── gmail.py              # Google URLs, scopes
│   ├── providers/
│   │   ├── __init__.py           # Provider exports
│   │   ├── protocol.py           # EmailClientProtocol
│   │   ├── outlook.py            # Outlook client implementation
│   │   └── gmail.py              # Gmail client implementation
│   ├── types/
│   │   ├── __init__.py           # Type exports
│   │   ├── email.py              # Email, EmailAddress, EmailListResult
│   │   ├── folder.py             # Folder, FolderType
│   │   ├── attachment.py         # Attachment
│   │   ├── draft.py              # Draft
│   │   └── oauth.py              # OutlookOAuthConfig, OAuthCredentials, OAuthTokens
│   └── testing.py                # Protocols, hooks, fakes, factories
└── tests/
    ├── __init__.py
    ├── conftest.py               # Autouse hook reset fixture
    ├── auth/
    │   ├── test_outlook_auth.py
    │   └── test_gmail_auth.py
    ├── providers/
    │   ├── test_outlook.py
    │   └── test_gmail.py
    ├── cli/
    │   └── test_cli.py
    ├── testing/
    │   ├── test_fake_client.py
    │   ├── test_factories.py
    │   ├── test_production_hooks.py
    │   └── test_production_loaders.py
    └── scripts/
        └── test_guard_cli.py
```

## API Endpoints

### Microsoft Graph API (Outlook)

Base URL: `https://graph.microsoft.com/v1.0`

- `GET /me/mailFolders` — List folders
- `GET /me/mailFolders/{id}/messages` — List messages in folder
- `GET /me/messages/{id}` — Get single message
- `POST /me/sendMail` — Send email
- `POST /me/messages` — Create draft
- `POST /me/messages/{id}/send` — Send draft
- `POST /me/messages/{id}/reply` — Reply to message
- `POST /me/messages/{id}/replyAll` — Reply all
- `DELETE /me/messages/{id}` — Delete message
- `PATCH /me/messages/{id}` — Move message (change folder)
- `GET /me/messages/{id}/attachments/{attachId}` — Get attachment

### Gmail API

Base URL: `https://gmail.googleapis.com/gmail/v1`

- `GET /users/me/labels` — List folders (labels)
- `GET /users/me/messages` — List messages
- `GET /users/me/messages/{id}` — Get single message
- `POST /users/me/messages/send` — Send email
- `POST /users/me/drafts` — Create draft
- `POST /users/me/drafts/{id}/send` — Send draft
- `DELETE /users/me/messages/{id}` — Delete message
- `POST /users/me/messages/{id}/modify` — Move message (change labels)
- `GET /users/me/messages/{id}/attachments/{id}` — Get attachment

## Key Modules

### 1. types/ — Domain Models

All types are `TypedDict`s with encode/decode round-trip functions:

- **email.py** — `Email`, `EmailAddress`, `EmailListResult`, `BodyType`, `EmailImportance`
- **folder.py** — `Folder`, `FolderType` (inbox, sent, drafts, trash, spam, archive, custom)
- **attachment.py** — `Attachment` (with base64-encoded content_bytes)
- **draft.py** — `Draft`
- **oauth.py** — `OutlookOAuthConfig`, re-exports `OAuthCredentials`/`OAuthTokens` from platform_core

### 2. auth/ — OAuth 2.0 Flows

Both providers follow the same pattern: build auth URL → exchange code for tokens → refresh expired tokens → full authorize flow → load-or-authorize convenience function.

- **outlook.py** — Microsoft OAuth with PKCE, multi-tenant support (`tenant_id = "common"`)
- **gmail.py** — Google OAuth with PKCE
- **common.py** — Shared auth utilities

### 3. providers/ — Email Client Implementations

- **protocol.py** — `EmailClientProtocol` defining the full email API: send, get, list, search, create/send draft, reply, delete, move, list folders, get attachment
- **outlook.py** — `_OutlookEmailClient` implementing the protocol via Microsoft Graph API
- **gmail.py** — `_GmailEmailClient` implementing the protocol via Gmail API, with MIME encoding/decoding

### 4. client.py — Factory Functions

Two factory functions (`outlook_email_client`, `gmail_email_client`) that take `OAuthTokens` and return an `EmailClientProtocol` instance.

### 5. cli.py — Command-Line Interface

**Commands:** `auth`, `folders`, `list`, `read`, `send`, `search`

- `cmd_send` reads body from a file path, supports `--cc`, `--bcc`, `--html` (wraps in `<pre>`), and `--attach` (repeatable, base64-encodes files)
- `cmd_search` uses Microsoft Graph KQL `$search=` syntax
- `cmd_list` and `cmd_search` share `_display_message_rows` for rendering

**Argument decoding:** Each command has a `TypedDict` (`ListArgs`, `ReadArgs`, `SendArgs`, `SearchArgs`) and a `decode_*` function that safely extracts typed values from `argparse.Namespace`.

**Account system:** `Account` dataclass maps environment variable names for tokens/credentials. Currently configured for Outlook with automatic token refresh.

### 6. testing.py — Protocols, Hooks, Fakes

**Hooks-based dependency injection:** All external I/O goes through the `HooksContainer` singleton — HTTP calls, file I/O, browser opens, token loading/saving, console output, timestamps, and path resolution. Tests replace hooks with lambdas or fakes; production hooks are registered via `_init_production_hooks()`.

**Hook categories:**
- HTTP (get, post, patch, delete)
- Token/config loading and saving (Outlook and Gmail)
- System (browser, console I/O, timestamps, file read/write/exists, binary file read)
- Path resolution (all four use a single `GetPathHook` type)
- CLI-specific (env get/set, current datetime)

**FakeEmailClient:** Full `EmailClientProtocol` implementation backed by in-memory dicts. Supports adding test emails/drafts/folders/attachments and inspecting sent/deleted/moved emails.

**Factory functions:** Helpers like `make_fake_email`, `make_fake_tokens`, `make_fake_http_get`, `make_raising_http_post`, etc. for concise test setup.

**Guard script hooks:** `guard.py` has its own injectable hooks for `find_monorepo_root` and `load_orchestrator`, enabling isolated testing of the monorepo guard runner.

## Environment Configuration

### Development (File-based)

- `~/.microsoft/email_credentials.json` — Outlook OAuth config
- `~/.microsoft/email_tokens.json` — Outlook access/refresh tokens
- `~/.google/email_credentials.json` — Gmail OAuth credentials
- `~/.google/email_tokens.json` — Gmail access/refresh tokens
- `libs/platform_email/.env` — CLI environment variables

### Production (Environment Variables)

Outlook: `OUTLOOK_CLIENT_ID`, `OUTLOOK_CLIENT_SECRET`, `OUTLOOK_ACCESS_TOKEN`, `OUTLOOK_REFRESH_TOKEN`, `OUTLOOK_TOKEN_EXPIRES_AT`

Gmail: `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REDIRECT_URI`, `GMAIL_ACCESS_TOKEN`, `GMAIL_REFRESH_TOKEN`, `GMAIL_TOKEN_EXPIRES_AT`

## Test Coverage

- 100% statement and branch coverage required
- Round-trip validation for all encode/decode pairs
- OAuth flows tested with fake HTTP responses
- Email clients tested with fake HTTP hooks
- FakeEmailClient protocol compliance verified
- CLI commands tested with fake hooks
- CLI argument parsing and TypedDict decoding tested
- Guard script tested using injectable hooks
