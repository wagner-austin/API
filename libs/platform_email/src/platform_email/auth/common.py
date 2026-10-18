"""Token expiry for both providers, read against the hooks' clock.

The PKCE helpers and the authorization page come from platform_core.oauth,
which each flow imports directly.
"""

from __future__ import annotations

from platform_core.oauth import (
    is_token_expired as _core_is_token_expired,
)

from platform_email.testing import hooks
from platform_email.types import OAuthTokens


def is_token_expired(tokens: OAuthTokens, *, buffer_seconds: int = 60) -> bool:
    """Check if access token is expired or will expire soon.

    Uses centralized is_token_expired from platform_core.oauth with
    current time from hooks.

    Args:
        tokens: OAuth tokens to check.
        buffer_seconds: Consider expired if within this many seconds of expiry.

    Returns:
        True if token is expired or will expire within buffer.
    """
    return _core_is_token_expired(
        tokens,
        hooks.current_time(),
        buffer_seconds=buffer_seconds,
    )


__all__ = [
    "is_token_expired",
]
