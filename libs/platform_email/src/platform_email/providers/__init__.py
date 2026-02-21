"""Email provider implementations.

Re-exports provider clients for internal use.
"""

from platform_email.providers.gmail import _GmailEmailClient
from platform_email.providers.outlook import _OutlookEmailClient
from platform_email.providers.protocol import EmailClientProtocol, HTTPErrorProtocol

__all__ = [
    "EmailClientProtocol",
    "HTTPErrorProtocol",
    "_GmailEmailClient",
    "_OutlookEmailClient",
]
