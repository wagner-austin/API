"""Multi-provider email library supporting Microsoft Outlook and Gmail.

This library provides:
- OAuth 2.0 authentication with Microsoft Graph API and Gmail API
- Email sending, reading, searching, and management
- Draft creation and sending
- Folder management
- Attachment handling
- TypedDict-based strict typing throughout

Example usage (Outlook):
    from platform_email import (
        outlook_load_or_authorize,
        outlook_email_client,
    )

    # Authenticate (opens browser first time)
    tokens = outlook_load_or_authorize()
    client = outlook_email_client(tokens=tokens)

    # List folders
    folders = client.list_folders()

    # Send an email
    email = client.send_email(
        to=("recipient@example.com",),
        subject="Hello",
        body="Hello, World!",
    )

Example usage (Gmail):
    from platform_email import (
        gmail_load_or_authorize,
        gmail_email_client,
    )

    # Authenticate (opens browser first time)
    tokens = gmail_load_or_authorize()
    client = gmail_email_client(tokens=tokens)

    # Search emails
    results = client.search_emails(query="from:sender@example.com")
"""

# Errors - from platform_core
from platform_core.errors import AppError, EmailErrorCode

# Auth - Common
from platform_email.auth.common import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
    is_token_expired,
)

# Auth - Gmail
from platform_email.auth.gmail import (
    authorize_gmail,
    build_gmail_auth_url,
    exchange_gmail_code_for_tokens,
    get_valid_gmail_tokens,
    gmail_load_or_authorize,
    refresh_gmail_access_token,
)

# Auth - Outlook
from platform_email.auth.outlook import (
    authorize_outlook,
    build_outlook_auth_url,
    exchange_outlook_code_for_tokens,
    get_valid_outlook_tokens,
    outlook_load_or_authorize,
    refresh_outlook_access_token,
)

# Client factories
from platform_email.client import gmail_email_client, outlook_email_client

# Config - Gmail
from platform_email.config.gmail import (
    DEFAULT_GMAIL_CREDENTIALS_PATH,
    DEFAULT_GMAIL_TOKENS_PATH,
    GMAIL_API_BASE,
    GMAIL_AUTH_URL,
    GMAIL_EMAIL_SCOPES,
    GMAIL_TOKEN_URL,
    get_gmail_credentials_path,
    get_gmail_tokens_path,
)

# Config - Outlook
from platform_email.config.outlook import (
    DEFAULT_OUTLOOK_CREDENTIALS_PATH,
    DEFAULT_OUTLOOK_TOKENS_PATH,
    OUTLOOK_API_BASE,
    OUTLOOK_EMAIL_SCOPES,
    get_outlook_credentials_path,
    get_outlook_tokens_path,
    outlook_auth_url,
    outlook_token_url,
)
from platform_email.fake_hooks import (
    make_fake_attachment,
    make_fake_console,
    make_fake_current_time,
    make_fake_draft,
    make_fake_email,
    make_fake_file_system,
    make_fake_folder,
    make_fake_gmail_credentials,
    make_fake_http_delete,
    make_fake_http_get,
    make_fake_http_send,
    make_fake_no_tokens,
    make_fake_outlook_config,
    make_fake_tokens,
    make_raising_http_delete,
    make_raising_http_get,
    make_raising_http_send,
)

# Testing
from platform_email.fakes import (
    FakeEmailClient,
)
from platform_email.testing import (
    EmailClientProtocol,
    HooksContainer,
    HTTPErrorProtocol,
    hooks,
    reset_hooks,
)

# Types
from platform_email.types import (
    Attachment,
    BodyType,
    Draft,
    Email,
    EmailAddress,
    EmailImportance,
    EmailListResult,
    Folder,
    FolderType,
    GmailOAuthConfig,
    OAuthCredentials,
    OAuthTokenResponse,
    OAuthTokens,
    OutlookOAuthConfig,
    TokenType,
    decode_attachment,
    decode_draft,
    decode_email,
    decode_email_address,
    decode_email_list_result,
    decode_folder,
    decode_gmail_oauth_config,
    decode_oauth_credentials,
    decode_oauth_token_response,
    decode_oauth_tokens,
    decode_outlook_oauth_config,
    encode_attachment,
    encode_draft,
    encode_email,
    encode_email_address,
    encode_email_list_result,
    encode_folder,
    encode_gmail_oauth_config,
    encode_oauth_credentials,
    encode_oauth_token_response,
    encode_oauth_tokens,
    encode_outlook_oauth_config,
)

__all__ = [
    "DEFAULT_GMAIL_CREDENTIALS_PATH",
    "DEFAULT_GMAIL_TOKENS_PATH",
    "DEFAULT_OUTLOOK_CREDENTIALS_PATH",
    "DEFAULT_OUTLOOK_TOKENS_PATH",
    "GMAIL_API_BASE",
    "GMAIL_AUTH_URL",
    "GMAIL_EMAIL_SCOPES",
    "GMAIL_TOKEN_URL",
    "OUTLOOK_API_BASE",
    "OUTLOOK_EMAIL_SCOPES",
    "AppError",
    "Attachment",
    "BodyType",
    "Draft",
    "Email",
    "EmailAddress",
    "EmailClientProtocol",
    "EmailErrorCode",
    "EmailImportance",
    "EmailListResult",
    "FakeEmailClient",
    "Folder",
    "FolderType",
    "GmailOAuthConfig",
    "HTTPErrorProtocol",
    "HooksContainer",
    "OAuthCredentials",
    "OAuthTokenResponse",
    "OAuthTokens",
    "OutlookOAuthConfig",
    "TokenType",
    "authorize_gmail",
    "authorize_outlook",
    "build_gmail_auth_url",
    "build_outlook_auth_url",
    "decode_attachment",
    "decode_draft",
    "decode_email",
    "decode_email_address",
    "decode_email_list_result",
    "decode_folder",
    "decode_gmail_oauth_config",
    "decode_oauth_credentials",
    "decode_oauth_token_response",
    "decode_oauth_tokens",
    "decode_outlook_oauth_config",
    "encode_attachment",
    "encode_draft",
    "encode_email",
    "encode_email_address",
    "encode_email_list_result",
    "encode_folder",
    "encode_gmail_oauth_config",
    "encode_oauth_credentials",
    "encode_oauth_token_response",
    "encode_oauth_tokens",
    "encode_outlook_oauth_config",
    "exchange_gmail_code_for_tokens",
    "exchange_outlook_code_for_tokens",
    "generate_code_challenge",
    "generate_code_verifier",
    "generate_state",
    "get_gmail_credentials_path",
    "get_gmail_tokens_path",
    "get_outlook_credentials_path",
    "get_outlook_tokens_path",
    "get_valid_gmail_tokens",
    "get_valid_outlook_tokens",
    "gmail_email_client",
    "gmail_load_or_authorize",
    "hooks",
    "is_token_expired",
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
    "make_fake_tokens",
    "make_raising_http_delete",
    "make_raising_http_get",
    "make_raising_http_send",
    "outlook_auth_url",
    "outlook_email_client",
    "outlook_load_or_authorize",
    "outlook_token_url",
    "refresh_gmail_access_token",
    "refresh_outlook_access_token",
    "reset_hooks",
]
