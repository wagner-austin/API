"""TypedDict definitions for email types.

Re-exports all types from submodules for convenient access.
"""

from platform_email.types.attachment import (
    Attachment,
    decode_attachment,
    encode_attachment,
)
from platform_email.types.draft import (
    Draft,
    decode_draft,
    encode_draft,
)
from platform_email.types.email import (
    BodyType,
    Email,
    EmailAddress,
    EmailImportance,
    EmailListResult,
    decode_email,
    decode_email_address,
    decode_email_list_result,
    encode_email,
    encode_email_address,
    encode_email_list_result,
)
from platform_email.types.folder import (
    Folder,
    FolderType,
    decode_folder,
    encode_folder,
)
from platform_email.types.oauth import (
    GmailOAuthConfig,
    OAuthCredentials,
    OAuthTokenResponse,
    OAuthTokens,
    OutlookOAuthConfig,
    TokenType,
    decode_gmail_oauth_config,
    decode_oauth_credentials,
    decode_oauth_token_response,
    decode_oauth_tokens,
    decode_outlook_oauth_config,
    encode_gmail_oauth_config,
    encode_oauth_credentials,
    encode_oauth_token_response,
    encode_oauth_tokens,
    encode_outlook_oauth_config,
)

__all__ = [
    "Attachment",
    "BodyType",
    "Draft",
    "Email",
    "EmailAddress",
    "EmailImportance",
    "EmailListResult",
    "Folder",
    "FolderType",
    "GmailOAuthConfig",
    "OAuthCredentials",
    "OAuthTokenResponse",
    "OAuthTokens",
    "OutlookOAuthConfig",
    "TokenType",
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
]
