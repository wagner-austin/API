"""Common OAuth utilities shared between providers.

Re-exports PKCE and token utilities from platform_core.oauth.
"""

from __future__ import annotations

from platform_core.oauth import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
)
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
    "generate_code_challenge",
    "generate_code_verifier",
    "generate_state",
    "is_token_expired",
]
