"""Factory functions for creating email clients.

Provides outlook_email_client() and gmail_email_client() for creating
email clients for each provider.
"""

from __future__ import annotations

from platform_email.providers.gmail import _GmailEmailClient
from platform_email.providers.outlook import _OutlookEmailClient
from platform_email.testing import EmailClientProtocol
from platform_email.types import OAuthTokens


def outlook_email_client(*, tokens: OAuthTokens) -> EmailClientProtocol:
    """Create a Microsoft Outlook email client.

    Args:
        tokens: OAuth tokens from outlook_load_or_authorize().

    Returns:
        EmailClientProtocol implementation for Outlook.

    Example:
        >>> from platform_email import outlook_load_or_authorize, outlook_email_client
        >>> tokens = outlook_load_or_authorize()
        >>> client = outlook_email_client(tokens=tokens)
        >>> folders = client.list_folders()
    """
    client: EmailClientProtocol = _OutlookEmailClient(access_token=tokens["access_token"])
    return client


def gmail_email_client(*, tokens: OAuthTokens) -> EmailClientProtocol:
    """Create a Google Gmail email client.

    Args:
        tokens: OAuth tokens from gmail_load_or_authorize().

    Returns:
        EmailClientProtocol implementation for Gmail.

    Example:
        >>> from platform_email import gmail_load_or_authorize, gmail_email_client
        >>> tokens = gmail_load_or_authorize()
        >>> client = gmail_email_client(tokens=tokens)
        >>> folders = client.list_folders()
    """
    client: EmailClientProtocol = _GmailEmailClient(access_token=tokens["access_token"])
    return client


__all__ = [
    "gmail_email_client",
    "outlook_email_client",
]
