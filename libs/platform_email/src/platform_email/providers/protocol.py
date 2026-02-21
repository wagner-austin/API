"""EmailClientProtocol definition.

Re-exports the protocol from testing module for providers to implement.
"""

from platform_email.testing import EmailClientProtocol, HTTPErrorProtocol

__all__ = [
    "EmailClientProtocol",
    "HTTPErrorProtocol",
]
