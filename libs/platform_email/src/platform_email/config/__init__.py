"""Configuration module for email providers.

Re-exports configuration from both Outlook and Gmail submodules.
"""

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
    "get_gmail_credentials_path",
    "get_gmail_tokens_path",
    "get_outlook_credentials_path",
    "get_outlook_tokens_path",
    "outlook_auth_url",
    "outlook_token_url",
]
