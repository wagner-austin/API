# Architecture: platform_core Library

## Overview

The `platform_core` library provides shared platform utilities including error handling, validation, logging, health checks, typed event schemas, service clients, and OAuth 2.0 utilities. It serves as the foundation for all services and libraries in the monorepo.

## Directory Structure

```
libs/platform_core/
├── pyproject.toml
├── README.md
├── Makefile
├── docs/
│   └── architecture-plan.md
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/platform_core/
│   ├── __init__.py              # Public exports
│   ├── py.typed                 # PEP 561 marker
│   ├── errors.py                # AppError, ErrorCode enums
│   ├── json_utils.py            # JSON parsing and validation
│   ├── validators.py            # Input validation functions
│   ├── logging.py               # Structured logging setup
│   ├── health.py                # Health check endpoints
│   ├── security.py              # API key dependencies
│   ├── request_context.py       # Request ID middleware
│   ├── fastapi.py               # FastAPI integration
│   ├── http_client.py           # HTTP client protocols
│   ├── http_utils.py            # HTTP utilities
│   ├── oauth.py                 # OAuth 2.0 PKCE and token utilities
│   ├── oauth_types.py           # OAuth TypedDicts with encode/decode
│   ├── oauth_testing.py         # OAuth test utilities (public)
│   ├── testing.py               # HTTP fakes and test utilities
│   ├── data_bank_client.py      # Data bank API client
│   ├── data_bank_events.py      # Data bank event schemas
│   ├── data_bank_protocol.py    # Data bank protocols
│   ├── model_trainer_client.py  # Model trainer API client
│   ├── job_events.py            # Generic job event schemas
│   ├── job_keys.py              # Job key utilities
│   ├── job_types.py             # Job type definitions
│   ├── queues.py                # Queue name constants
│   ├── trainer_keys.py          # Trainer key utilities
│   ├── trainer_metrics_events.py # Trainer metrics schemas
│   ├── digits_metrics_events.py # Digits metrics schemas
│   ├── covenant_metrics_events.py # Covenant metrics schemas
│   └── config/                  # Service configuration loaders
│       ├── __init__.py
│       ├── _test_hooks.py
│       ├── _utils.py
│       ├── covenant_radar.py
│       ├── data_bank.py
│       ├── discordbot.py
│       ├── handwriting_ai.py
│       ├── model_trainer.py
│       └── turkic_api.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_*.py
```

## OAuth 2.0 Module

The OAuth module provides reusable, provider-agnostic OAuth 2.0 utilities.

### oauth_types.py - TypedDicts

```python
TokenType = Literal["Bearer"]


class OAuthCredentials(TypedDict):
    """OAuth 2.0 client credentials."""

    client_id: str
    client_secret: str
    redirect_uri: str


class OAuthTokens(TypedDict):
    """OAuth 2.0 access and refresh tokens."""

    access_token: str
    refresh_token: str
    expires_at: int
    token_type: TokenType


class OAuthTokenResponse(TypedDict):
    """Response from OAuth token endpoint."""

    access_token: str
    refresh_token: str | None  # Only on initial auth
    expires_in: int
    token_type: str
```

Each TypedDict has corresponding `encode_*` and `decode_*` functions with `require_*` validation.

### oauth.py - Core Functions

```python
# Hook types for dependency injection
HttpPostHook = Callable[[str, dict[str, str], str], str]
CurrentTimeHook = Callable[[], int]


# PKCE (Proof Key for Code Exchange)
def generate_code_verifier(*, length: int = 64) -> str: ...
def generate_code_challenge(verifier: str) -> str: ...
def generate_state() -> str: ...


# Token utilities
def is_token_expired(
    tokens: OAuthTokens, current_time: int, *, buffer_seconds: int = 60
) -> bool: ...


def build_authorization_url(
    auth_endpoint: str,
    client_id: str,
    redirect_uri: str,
    *,
    code_challenge: str,
    state: str,
    scopes: tuple[str, ...],
    access_type: str = "offline",
    prompt: str = "consent",
) -> str: ...


# Token exchange
def exchange_authorization_code(
    token_endpoint: str,
    credentials: OAuthCredentials,
    code: str,
    code_verifier: str,
    *,
    http_post: HttpPostHook,
    current_time: int,
) -> OAuthTokens: ...


def refresh_access_token(
    token_endpoint: str,
    credentials: OAuthCredentials,
    refresh_token: str,
    *,
    http_post: HttpPostHook,
    current_time: int,
) -> OAuthTokens: ...
```

### oauth_testing.py - Test Utilities

Public test utilities for OAuth testing:

```python
# HTTP hook fakes
def make_fake_http_post(response: str) -> HttpPostHook: ...
def make_raising_http_post(exc: BaseException) -> HttpPostHook: ...
def make_sequenced_http_post(responses: list[str | BaseException]) -> HttpPostHook: ...

# Time hook fakes
def make_fake_current_time(timestamp: int) -> CurrentTimeHook: ...
def make_advancing_current_time(start: int, increment: int = 1) -> CurrentTimeHook: ...

# Response helpers
def make_token_response_json(
    *, access_token: str = "test_access_token",
    refresh_token: str | None = "test_refresh_token",
    expires_in: int = 3600, token_type: str = "Bearer",
) -> str: ...

def make_error_response_json(
    *, error: str = "invalid_grant", error_description: str | None = None,
) -> str: ...

# Factory functions
def make_test_credentials(...) -> OAuthCredentials: ...
def make_test_tokens(...) -> OAuthTokens: ...
def make_test_token_response(...) -> OAuthTokenResponse: ...
```

### OAuthErrorCode

```python
class OAuthErrorCode(ErrorCodeBase):
    AUTH_FAILED = "oauth_auth_failed"
    INVALID_GRANT = "oauth_invalid_grant"
    INVALID_STATE = "oauth_invalid_state"
    TOKEN_EXPIRED = "oauth_token_expired"
    TOKEN_EXCHANGE_FAILED = "oauth_token_exchange_failed"
    TOKEN_REFRESH_FAILED = "oauth_token_refresh_failed"
    MISSING_REFRESH_TOKEN = "oauth_missing_refresh_token"
    TOKEN_ENDPOINT_ERROR = "oauth_token_endpoint_error"
```

## Design Principles

### Hook-Based Dependency Injection

Functions accept hooks as parameters rather than using module-level state:

```python
# Good: Dependencies passed explicitly
tokens = exchange_authorization_code(
    token_endpoint="...",
    credentials=creds,
    code=code,
    code_verifier=verifier,
    http_post=hooks.http_post,  # Injected
    current_time=hooks.current_time(),  # Injected
)

# Tests inject fakes
tokens = exchange_authorization_code(
    ...,
    http_post=make_fake_http_post(make_token_response_json()),
    current_time=1735200000,
)
```

### Strict Typing

- No `Any`, `cast`, `type: ignore`, or `.pyi` files
- All TypedDicts have encode/decode with `require_*` validation
- Literal types for constrained values
- Immutable data structures (tuples, not lists)

### Testing Pattern

- Libs export `testing.py` with public test utilities
- Services use `_test_hooks.py` for internal dependency injection
- No mocks - use fakes with explicit behavior
- 100% statement and branch coverage required

## Consumer Integration

### platform_calendar

Uses centralized OAuth types and PKCE functions:

```python
# types.py - Re-exports from platform_core
from platform_core.oauth_types import OAuthCredentials as OAuthCredentials
from platform_core.oauth_types import OAuthTokens as OAuthTokens
from platform_core.oauth_types import decode_oauth_token_response as decode_oauth_token_response

# auth.py - Uses centralized PKCE
from platform_core.oauth import (
    generate_code_challenge,
    generate_code_verifier,
    generate_state,
)
from platform_core.oauth import is_token_expired as _core_is_token_expired

# testing.py - Re-exports test utilities
from platform_core.oauth_testing import make_error_response_json, make_token_response_json
```

### Future OAuth Consumers

Any service or library needing OAuth can:

1. Import types from `platform_core.oauth_types`
2. Use PKCE/token functions from `platform_core.oauth`
3. Use test utilities from `platform_core.oauth_testing`
4. Handle errors with `OAuthErrorCode`

## Test Coverage

- 100% statement and branch coverage required
- Tests for each encode/decode pair with round-trip validation
- Tests for PKCE generation (uniqueness, format, determinism)
- Tests for token expiry edge cases
- Tests for exchange/refresh success and error paths
- Tests for all testing utility functions
