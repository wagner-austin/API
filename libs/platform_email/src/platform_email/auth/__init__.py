"""OAuth authentication module for email providers.

Re-exports authentication flows for both Outlook and Gmail.
"""

from platform_email.auth.common import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
    is_token_expired,
)
from platform_email.auth.gmail import (
    authorize_gmail,
    build_gmail_auth_url,
    exchange_gmail_code_for_tokens,
    get_valid_gmail_tokens,
    gmail_load_or_authorize,
    refresh_gmail_access_token,
)
from platform_email.auth.outlook import (
    authorize_outlook,
    build_outlook_auth_url,
    exchange_outlook_code_for_tokens,
    get_valid_outlook_tokens,
    outlook_load_or_authorize,
    refresh_outlook_access_token,
)

__all__ = [
    "authorize_gmail",
    "authorize_outlook",
    "build_gmail_auth_url",
    "build_outlook_auth_url",
    "exchange_gmail_code_for_tokens",
    "exchange_outlook_code_for_tokens",
    "generate_code_challenge",
    "generate_code_verifier",
    "generate_state",
    "get_valid_gmail_tokens",
    "get_valid_outlook_tokens",
    "gmail_load_or_authorize",
    "is_token_expired",
    "outlook_load_or_authorize",
    "refresh_gmail_access_token",
    "refresh_outlook_access_token",
]
