# Role-Based Token System Refactor

## Document Status

| Field | Value |
|-------|-------|
| **Author** | Platform Engineering |
| **Created** | 2025-11-28 |
| **Status** | PROPOSED |
| **Scope** | All services, clients, and shared libraries |

---

## Executive Summary

This document proposes a comprehensive refactor to replace the current fragmented API key authentication system with a unified role-based token system. The new system will provide:

1. **Centralized token management** via a new `platform_auth` library
2. **Role-based access control (RBAC)** with typed permissions
3. **Service-to-service authentication** with scoped tokens
4. **Consistent enforcement** across all services
5. **Full type safety** with no `Any`, casts, or type ignores

---

## Current State Analysis

### Authentication Patterns by Component

#### 1. data-bank-api (Permission-Based Multi-Key)

**Location:** `services/data-bank-api/src/data_bank_api/api/routes/files.py`

```
Current Pattern:
- api_upload_keys: frozenset[str]  # Keys allowed to upload
- api_read_keys: frozenset[str]    # Keys allowed to read
- api_delete_keys: frozenset[str]  # Keys allowed to delete
```

**Config:** `libs/platform_core/src/platform_core/config/data_bank.py`
- Uses `_require_env_csv("API_UPLOAD_KEYS")` for comma-separated key lists
- Read/delete keys default to upload keys if not specified
- Permission check via `_ensure_auth(cfg, perm, req)`

**Issues:**
- Keys are static strings with no metadata
- No token expiration
- No audit trail
- Permission model is service-specific, not reusable

#### 2. handwriting-ai (Single Static Key)

**Location:** `services/handwriting-ai/src/handwriting_ai/api/app.py`

```
Current Pattern:
- HANDWRITING_API_KEY (required env var)
- SECURITY__API_KEY (optional override)
- api_key_enabled: bool (toggle)
```

**Config:** `libs/platform_core/src/platform_core/config/handwriting_ai.py`
- Uses `HandwritingAiSecurityConfig(TypedDict)` with `api_key` and `api_key_enabled`
- Protection via `create_api_key_dependency()` from `platform_core.security`

**Issues:**
- All-or-nothing access
- No granular permissions (e.g., predict vs admin)
- Key toggle can disable auth entirely

#### 3. Model-Trainer (Single Static Key)

**Location:** `services/Model-Trainer/src/model_trainer/api/middleware.py`

```
Current Pattern:
- SECURITY__API_KEY env var
- Same create_api_key_dependency() mechanism
```

**Config:** `libs/platform_core/src/platform_core/config/model_trainer.py`
- `ModelTrainerSecurityConfig(TypedDict)` with single `api_key: str`

**Issues:**
- No distinction between training ops vs status checks
- No service identity tracking

#### 4. turkic-api (Uses External Service Key)

**Location:** `libs/platform_core/src/platform_core/config/turkic_api.py`

```
Current Pattern:
- TURKIC_DATA_BANK_API_KEY for data-bank-api access
- No protection on own endpoints
```

**Issues:**
- Stores keys for other services in its config
- Cross-service key management is ad-hoc

#### 5. qr-api (No Authentication)

**Location:** `services/qr-api/src/qr_api/app.py`

```
Current Pattern:
- No authentication implemented
- All endpoints public
```

#### 6. transcript-api (No Authentication)

**Location:** `services/transcript-api/src/transcript_api/app.py`

```
Current Pattern:
- No authentication implemented
- Uses external API keys (OpenAI, YouTube) for outbound calls only
```

#### 7. DiscordBot Client (Multiple Service Keys)

**Location:** `libs/platform_core/src/platform_core/config/discordbot.py`

```
Current Pattern:
- HANDWRITING_API_KEY for handwriting-ai
- MODEL_TRAINER_API_KEY for Model-Trainer
- YOUTUBE_API_KEY for external YouTube API
- OPENAI_API_KEY for external OpenAI API
```

**Issues:**
- Each service requires separate key management
- No unified identity for the bot
- Key proliferation as services grow

### Shared Security Infrastructure

**Location:** `libs/platform_core/src/platform_core/security.py`

```python
class ApiKeyCheckFn(Protocol):
    def __call__(self, x_api_key: str | None = ...) -> None: ...

def create_api_key_dependency(
    *,
    required_key: str,
    error_code: ErrorCodeBase,
    http_status: int | None = None,
    header_name: str = "X-API-Key",
    message: str = "Unauthorized",
) -> ApiKeyCheckFn
```

**Strengths:**
- Protocol-based, type-safe
- Uses `__import__()` pattern for FastAPI imports
- No `Any` types

**Weaknesses:**
- Only supports single static key comparison
- No roles, scopes, or permissions
- No token introspection

### Environment Variable Patterns

| Service | Key Variables | Pattern |
|---------|--------------|---------|
| data-bank-api | `API_UPLOAD_KEYS`, `API_READ_KEYS`, `API_DELETE_KEYS` | CSV frozenset |
| handwriting-ai | `HANDWRITING_API_KEY`, `SECURITY__API_KEY` | Single string |
| Model-Trainer | `SECURITY__API_KEY` | Single string |
| turkic-api | `TURKIC_DATA_BANK_API_KEY` | Single string |
| qr-api | None | N/A |
| transcript-api | None (external only) | N/A |
| DiscordBot | Multiple per-service keys | Single strings |

---

## Problem Statement

### Core Issues

1. **Fragmented Authentication**: Each service implements its own auth pattern
2. **No Role Hierarchy**: Only "has key" or "no key" - no gradations
3. **Static Secrets**: Keys never expire, no rotation support
4. **No Service Identity**: Cannot distinguish which client is making requests
5. **Permission Proliferation**: data-bank-api has 3 key sets; others have 1 or 0
6. **Audit Gap**: No logging of which token was used for which operation
7. **Inconsistent Headers**: `X-API-Key` vs `X-Api-Key` capitalization variance
8. **Test Complexity**: Tests must mock different auth patterns per service

### Technical Debt Risks

1. **Drift**: New services may implement yet another auth pattern
2. **Security**: Long-lived static keys increase breach exposure
3. **Maintenance**: Key rotation requires coordinated env var updates
4. **Observability**: Cannot correlate requests to callers

---

## Proposed Solution: Role-Based Token System

### Design Principles

1. **Strict Typing**: All token data uses `TypedDict`/`Protocol` - no `Any`
2. **Fail-Fast**: No fallbacks, best-effort, or try/except recovery
3. **Parse at Boundary**: Token validation happens at HTTP layer
4. **Protocol-First**: Service interfaces defined as Protocols
5. **Zero External Dependencies**: No OAuth servers, JWT libraries with native code
6. **100% Test Coverage**: Branch coverage required for all auth code

### Token Architecture

#### Token Structure

```python
# libs/platform_auth/src/platform_auth/token_types.py

from typing import Final, Literal, NewType, TypedDict

# Canonical role names (exhaustive)
Role = Literal[
    "admin",           # Full access to all operations
    "service",         # Service-to-service communication
    "operator",        # Read + write operations (no admin)
    "reader",          # Read-only access
    "uploader",        # Write-only access (e.g., data-bank upload)
]

# Permission scopes (per-service granularity)
Scope = Literal[
    # data-bank-api
    "databank:upload",
    "databank:read",
    "databank:delete",
    "databank:admin",

    # handwriting-ai
    "handwriting:predict",
    "handwriting:admin",
    "handwriting:models:read",
    "handwriting:models:write",

    # Model-Trainer
    "trainer:runs:read",
    "trainer:runs:write",
    "trainer:tokenizers:read",
    "trainer:tokenizers:write",
    "trainer:admin",

    # turkic-api
    "turkic:corpus:read",
    "turkic:corpus:write",
    "turkic:transliterate",
    "turkic:admin",

    # qr-api
    "qr:generate",
    "qr:admin",

    # transcript-api
    "transcript:captions",
    "transcript:stt",
    "transcript:admin",
]

KeyId = NewType("KeyId", str)


class TokenPayload(TypedDict):
    """Validated token payload after decoding."""
    token_id: str           # Unique identifier (for revocation, audit)
    subject: str            # Who/what this token represents (user ID, service name)
    role: Role              # Primary role
    scopes: tuple[Scope, ...]  # Granted scopes (immutable, validated)
    issued_at: int          # Unix timestamp (seconds)
    expires_at: int         # Unix timestamp (0 = never expires)
    issuer: str             # Token issuer identifier


class TokenHeader(TypedDict):
    """Token header for algorithm and key selection."""
    alg: Literal["HS256"]   # HMAC-SHA256 only
    typ: Literal["JWT"]
    kid: KeyId              # Key identifier for rotation


ROLE_SCOPES: Final[dict[Role, frozenset[Scope]]] = {
    "admin": frozenset([
        "databank:upload", "databank:read", "databank:delete", "databank:admin",
        "handwriting:predict", "handwriting:admin", "handwriting:models:read", "handwriting:models:write",
        "trainer:runs:read", "trainer:runs:write", "trainer:tokenizers:read", "trainer:tokenizers:write", "trainer:admin",
        "turkic:corpus:read", "turkic:corpus:write", "turkic:transliterate", "turkic:admin",
        "qr:generate", "qr:admin",
        "transcript:captions", "transcript:stt", "transcript:admin",
    ]),
    "service": frozenset([
        "databank:upload", "databank:read",
        "handwriting:predict", "handwriting:models:read",
        "trainer:runs:read", "trainer:runs:write", "trainer:tokenizers:read",
        "turkic:corpus:read", "turkic:transliterate",
        "qr:generate",
        "transcript:captions", "transcript:stt",
    ]),
    "operator": frozenset([
        "databank:upload", "databank:read",
        "handwriting:predict", "handwriting:models:read",
        "trainer:runs:read", "trainer:runs:write",
        "turkic:corpus:read", "turkic:transliterate",
        "qr:generate",
        "transcript:captions", "transcript:stt",
    ]),
    "reader": frozenset([
        "databank:read",
        "handwriting:predict", "handwriting:models:read",
        "trainer:runs:read", "trainer:tokenizers:read",
        "turkic:corpus:read",
        "qr:generate",
        "transcript:captions",
    ]),
    "uploader": frozenset([
        "databank:upload",
    ]),
}

SCOPE_SET: Final[frozenset[Scope]] = frozenset(scope for scopes in ROLE_SCOPES.values() for scope in scopes)
```

#### Token Codec (HMAC-SHA256 JWT)

```python
# libs/platform_auth/src/platform_auth/codec.py

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import time
from typing import Final, Mapping

from platform_core.json_utils import dump_json_bytes, load_json_bytes

from .errors import TokenDecodeError, TokenExpiredError, TokenSignatureError
from .token_types import KeyId, Role, Scope, TokenHeader, TokenPayload, ROLE_SCOPES, SCOPE_SET

_HEADER_BASE: Final[dict[str, str]] = {"alg": "HS256", "typ": "JWT"}
_ROLE_LEVELS: Final[dict[Role, int]] = {
    "admin": 100,
    "service": 80,
    "operator": 60,
    "reader": 40,
    "uploader": 20,
}


def _b64_encode(data: bytes) -> bytes:
    """URL-safe base64 encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _b64_decode(data: bytes) -> bytes:
    """URL-safe base64 decode with padding restoration."""
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data = data + b"=" * padding
    return base64.urlsafe_b64decode(data, validate=True)


def _compute_signature(message: bytes, secret: bytes) -> bytes:
    """Compute HMAC-SHA256 signature."""
    return hmac.new(secret, message, hashlib.sha256).digest()


def _require_valid_secret(secret: bytes) -> None:
    if len(secret) < 32:
        raise ValueError("secret must be at least 32 bytes")


def _require_non_empty(value: str, field: str) -> None:
    if value.strip() == "":
        raise ValueError(f"{field} cannot be empty")


def _validate_scopes(role: Role, scopes: tuple[Scope, ...]) -> tuple[Scope, ...]:
    if len(scopes) == 0:
        raise ValueError("scopes cannot be empty")
    scope_set = set(scopes)
    if not scope_set.issubset(SCOPE_SET):
        invalid = scope_set.difference(SCOPE_SET)
        raise ValueError(f"invalid scopes: {sorted(invalid)}")
    allowed = ROLE_SCOPES[role]
    if not scope_set.issubset(allowed):
        disallowed = scope_set.difference(allowed)
        raise ValueError(f"scopes not permitted for role {role}: {sorted(disallowed)}")
    # Preserve input order but ensure uniqueness
    seen: set[Scope] = set()
    ordered: list[Scope] = []
    for scope in scopes:
        if scope not in seen:
            seen.add(scope)
            ordered.append(scope)
    return tuple(ordered)


def encode_token(
    *,
    token_id: str,
    subject: str,
    role: Role,
    scopes: tuple[Scope, ...],
    issued_at: int,
    expires_at: int,
    issuer: str,
    secret: bytes,
    key_id: KeyId,
) -> str:
    """Encode a token payload into a signed JWT string.

    Args:
        token_id: Unique token identifier.
        subject: Token subject (user/service identifier).
        role: Primary role.
        scopes: Granted scopes (validated against role).
        issued_at: Unix timestamp of issuance (seconds).
        expires_at: Unix timestamp of expiration (0 for no expiry, must be >= issued_at when non-zero).
        issuer: Token issuer.
        secret: HMAC secret key (must be >= 32 bytes).
        key_id: Identifier for the secret (embedded in header).

    Returns:
        Signed JWT string.

    Raises:
        ValueError: If inputs are invalid.
    """
    _require_valid_secret(secret)
    _require_non_empty(token_id, "token_id")
    _require_non_empty(subject, "subject")
    _require_non_empty(issuer, "issuer")
    _require_non_empty(key_id, "key_id")
    if issued_at < 0:
        raise ValueError("issued_at must be non-negative")
    if expires_at < 0:
        raise ValueError("expires_at must be non-negative")
    if expires_at != 0 and expires_at < issued_at:
        raise ValueError("expires_at must be >= issued_at or 0")

    validated_scopes = _validate_scopes(role, scopes)

    header: TokenHeader = {
        "alg": "HS256",
        "typ": "JWT",
        "kid": key_id,
    }
    payload_dict: dict[str, str | int | list[str]] = {
        "tid": token_id,
        "sub": subject,
        "role": role,
        "scp": list(validated_scopes),
        "iat": issued_at,
        "exp": expires_at,
        "iss": issuer,
    }
    header_bytes = dump_json_bytes(header)
    header_b64 = _b64_encode(header_bytes)
    payload_bytes = dump_json_bytes(payload_dict)
    payload_b64 = _b64_encode(payload_bytes)

    message = header_b64 + b"." + payload_b64
    signature = _compute_signature(message, secret)
    signature_b64 = _b64_encode(signature)

    return (message + b"." + signature_b64).decode("ascii")


def decode_token(
    token: str,
    *,
    secrets: Mapping[KeyId, bytes],
    now: int | None = None,
) -> TokenPayload:
    """Decode and validate a JWT token.

    Args:
        token: The JWT string to decode.
        secrets: HMAC secrets keyed by key identifier for signature verification.
        now: Current Unix timestamp (defaults to time.time()).

    Returns:
        Validated TokenPayload.

    Raises:
        TokenDecodeError: If token format is invalid.
        TokenSignatureError: If signature verification fails.
        TokenExpiredError: If token has expired.
    """
    if now is None:
        now = int(time.time())

    parts = token.split(".")
    if len(parts) != 3:
        raise TokenDecodeError("invalid token format: expected 3 parts")

    header_b64, payload_b64, signature_b64 = parts
    try:
        header_bytes = _b64_decode(header_b64.encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise TokenDecodeError("invalid header encoding") from exc
    header_raw = load_json_bytes(header_bytes)
    header = _validate_header(header_raw)

    secret = secrets.get(header["kid"])
    if secret is None:
        raise TokenSignatureError("unknown key id")
    _require_valid_secret(secret)

    message = (header_b64 + "." + payload_b64).encode("ascii")
    expected_sig = _compute_signature(message, secret)
    try:
        actual_sig = _b64_decode(signature_b64.encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise TokenDecodeError("invalid signature encoding") from exc

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise TokenSignatureError("invalid token signature")

    try:
        payload_bytes = _b64_decode(payload_b64.encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise TokenDecodeError("invalid payload encoding") from exc
    payload_raw = load_json_bytes(payload_bytes)

    return _validate_payload(payload_raw, now)


def _validate_header(raw: object) -> TokenHeader:
    if not isinstance(raw, dict):
        raise TokenDecodeError("header is not an object")
    alg = raw.get("alg")
    typ = raw.get("typ")
    kid = raw.get("kid")
    if alg != "HS256":
        raise TokenDecodeError("unsupported alg")
    if typ != "JWT":
        raise TokenDecodeError("unsupported typ")
    if not isinstance(kid, str) or kid.strip() == "":
        raise TokenDecodeError("missing or invalid kid")
    return {
        "alg": "HS256",
        "typ": "JWT",
        "kid": KeyId(kid),
    }


def _validate_payload(raw: object, now: int) -> TokenPayload:
    """Validate raw JSON payload and convert to TokenPayload.

    Raises:
        TokenDecodeError: If payload structure is invalid.
        TokenExpiredError: If token has expired.
    """
    if not isinstance(raw, dict):
        raise TokenDecodeError("payload is not an object")

    tid = raw.get("tid")
    sub = raw.get("sub")
    role = raw.get("role")
    scp = raw.get("scp")
    iat = raw.get("iat")
    exp = raw.get("exp")
    iss = raw.get("iss")

    if not isinstance(tid, str) or tid.strip() == "":
        raise TokenDecodeError("missing or invalid tid")
    if not isinstance(sub, str) or sub.strip() == "":
        raise TokenDecodeError("missing or invalid sub")
    if not isinstance(role, str):
        raise TokenDecodeError("missing or invalid role")
    if role not in _ROLE_LEVELS:
        raise TokenDecodeError(f"unknown role: {role}")
    if not isinstance(scp, list):
        raise TokenDecodeError("missing or invalid scp")
    if not isinstance(iat, int) or iat < 0:
        raise TokenDecodeError("missing or invalid iat")
    if not isinstance(exp, int) or exp < 0:
        raise TokenDecodeError("missing or invalid exp")
    if exp != 0 and exp < iat:
        raise TokenDecodeError("exp must be >= iat or 0")
    if not isinstance(iss, str) or iss.strip() == "":
        raise TokenDecodeError("missing or invalid iss")

    validated_scopes: list[Scope] = []
    scope_set: set[Scope] = set()
    for scope in scp:
        if not isinstance(scope, str):
            raise TokenDecodeError("scope must be string")
        if scope not in SCOPE_SET:
            raise TokenDecodeError(f"unknown scope: {scope}")
        scope_typed: Scope = scope
        if scope_typed not in scope_set:
            scope_set.add(scope_typed)
            validated_scopes.append(scope_typed)

    role_typed: Role = role
    allowed_scopes = ROLE_SCOPES[role_typed]
    if not scope_set.issubset(allowed_scopes):
        disallowed = scope_set.difference(allowed_scopes)
        raise TokenDecodeError(f"scopes not permitted for role {role_typed}: {sorted(disallowed)}")

    if exp > 0 and exp < now:
        raise TokenExpiredError(f"token expired at {exp}")

    return {
        "token_id": tid,
        "subject": sub,
        "role": role_typed,
        "scopes": tuple(validated_scopes),
        "issued_at": iat,
        "expires_at": exp,
        "issuer": iss,
    }
```

#### Error Types

```python
# libs/platform_auth/src/platform_auth/errors.py

from __future__ import annotations


class TokenError(Exception):
    """Base class for token-related errors."""


class TokenDecodeError(TokenError):
    """Raised when token cannot be decoded."""


class TokenSignatureError(TokenError):
    """Raised when token signature is invalid."""


class TokenExpiredError(TokenError):
    """Raised when token has expired."""


class InsufficientScopeError(TokenError):
    """Raised when token lacks required scope."""

    def __init__(self, required: str, available: tuple[str, ...]) -> None:
        super().__init__(f"requires scope '{required}', have: {available}")
        self.required = required
        self.available = available


class InsufficientRoleError(TokenError):
    """Raised when token role is insufficient."""

    def __init__(self, required: str, actual: str) -> None:
        super().__init__(f"requires role '{required}', have: '{actual}'")
        self.required = required
        self.actual = actual
```

#### FastAPI Integration

```python
# libs/platform_auth/src/platform_auth/fastapi_integration.py

from __future__ import annotations

from typing import Protocol

from platform_core.errors import AppError, ErrorCode

from .codec import decode_token
from .errors import InsufficientScopeError
from .token_types import KeyId, Role, Scope, TokenPayload


class TokenCheckFn(Protocol):
    """Protocol for token check function signature."""

    def __call__(self, authorization: str | None = ...) -> TokenPayload: ...


class ScopeCheckFn(Protocol):
    """Protocol for scope-requiring check function signature."""

    def __call__(self, authorization: str | None = ...) -> TokenPayload: ...


class _DependsCtor(Protocol):
    """Constructor protocol for FastAPI Depends."""

    def __call__(
        self,
        dependency: TokenCheckFn | ScopeCheckFn | None = ...,
    ) -> object: ...


class _HeaderCtor(Protocol):
    """Constructor protocol for FastAPI Header."""

    def __call__(
        self,
        default: str | None = ...,
        *,
        alias: str | None = ...,
    ) -> str | None: ...


def _get_depends() -> _DependsCtor:
    """Get FastAPI Depends constructor with typed interface."""
    fastapi_mod = __import__("fastapi")
    depends_cls: _DependsCtor = getattr(fastapi_mod, "Depends")
    return depends_cls


def _get_header() -> _HeaderCtor:
    """Get FastAPI Header constructor with typed interface."""
    fastapi_mod = __import__("fastapi")
    header_cls: _HeaderCtor = getattr(fastapi_mod, "Header")
    return header_cls


def create_token_dependency(
    *,
    secrets: dict[KeyId, bytes],
    header_name: str = "Authorization",
    bearer_prefix: str = "Bearer ",
) -> TokenCheckFn:
    """Create a FastAPI dependency that validates bearer tokens.

    Args:
        secrets: HMAC secrets keyed by key id for verification.
        header_name: HTTP header to read token from.
        bearer_prefix: Expected prefix before token value.

    Returns:
        A callable dependency that returns TokenPayload on success.
    """
    header_ctor = _get_header()

    def _check(authorization: str | None = header_ctor(default=None, alias=header_name)) -> TokenPayload:
        if authorization is None or authorization.strip() == "":
            raise AppError(ErrorCode.UNAUTHORIZED, "missing authorization header", 401)

        if not authorization.startswith(bearer_prefix):
            raise AppError(ErrorCode.UNAUTHORIZED, "invalid authorization format", 401)

        token_str = authorization[len(bearer_prefix):]
        return decode_token(token_str, secrets=secrets)

    return _check


def create_scope_dependency(
    *,
    secrets: dict[KeyId, bytes],
    required_scope: Scope,
    header_name: str = "Authorization",
    bearer_prefix: str = "Bearer ",
) -> ScopeCheckFn:
    """Create a FastAPI dependency that requires a specific scope.

    Args:
        secret: HMAC secret for token verification.
        required_scope: The scope that must be present.
        header_name: HTTP header to read token from.
        bearer_prefix: Expected prefix before token value.

    Returns:
        A callable dependency that returns TokenPayload if scope is present.
    """
    base_check = create_token_dependency(
        secrets=secrets,
        header_name=header_name,
        bearer_prefix=bearer_prefix,
    )

    def _check_scope(authorization: str | None = _get_header()(default=None, alias=header_name)) -> TokenPayload:
        payload = base_check(authorization)

        if required_scope not in payload["scopes"]:
            raise AppError(
                ErrorCode.FORBIDDEN,
                f"requires scope '{required_scope}'",
                403,
            )

        return payload

    return _check_scope


def create_role_dependency(
    *,
    secrets: dict[KeyId, bytes],
    minimum_role: Role,
    header_name: str = "Authorization",
    bearer_prefix: str = "Bearer ",
) -> TokenCheckFn:
    """Create a FastAPI dependency that requires minimum role level.

    Role hierarchy (highest to lowest):
        admin > service > operator > reader > uploader

    Args:
        secret: HMAC secret for token verification.
        minimum_role: Minimum role required.
        header_name: HTTP header to read token from.
        bearer_prefix: Expected prefix before token value.

    Returns:
        A callable dependency that returns TokenPayload if role is sufficient.
    """
    role_levels: dict[Role, int] = {
        "admin": 100,
        "service": 80,
        "operator": 60,
        "reader": 40,
        "uploader": 20,
    }

    required_level = role_levels[minimum_role]
    base_check = create_token_dependency(
        secrets=secrets,
        header_name=header_name,
        bearer_prefix=bearer_prefix,
    )

    def _check_role(authorization: str | None = _get_header()(default=None, alias=header_name)) -> TokenPayload:
        payload = base_check(authorization)

        actual_level = role_levels.get(payload["role"], 0)
        if actual_level < required_level:
            raise AppError(
                ErrorCode.FORBIDDEN,
                f"requires role '{minimum_role}' or higher",
                403,
            )

        return payload

    return _check_role
```

### Service-Specific Configuration

#### Environment Variable Schema

Each service will use a consistent pattern:

```bash
# Token verification secrets (shared across all services, rotation-friendly)
# Semicolon-separated list of key_id:base64secret pairs
AUTH_TOKEN_SECRETS=primary:<base64-encoded-32-byte-secret>

# Primary key id to use when minting new tokens
AUTH_TOKEN_PRIMARY_KEY_ID=primary
```

#### Configuration Types

```python
# libs/platform_auth/src/platform_auth/config.py

from __future__ import annotations

import base64
from typing import TypedDict

from platform_core.config._utils import _require_env_str
from platform_auth.token_types import KeyId


class AuthConfig(TypedDict):
    """Authentication configuration."""
    primary_secret_id: KeyId
    secrets: dict[KeyId, bytes]


def _decode_secret(value: str) -> bytes:
    secret_bytes = base64.b64decode(value, validate=True)
    if len(secret_bytes) < 32:
        raise RuntimeError("token secret must decode to at least 32 bytes")
    return secret_bytes


def load_auth_config() -> AuthConfig:
    """Load authentication configuration from environment.

    Environment Variables:
        AUTH_TOKEN_SECRETS (required): Semicolon-separated list of key_id:base64secret pairs.
            Example: primary:dGhpc19pc19hX3NlY3JldF9zdHJpbmc=;rollover:bW9yZV9ieXRlc19oZXJl
        AUTH_TOKEN_PRIMARY_KEY_ID (required): Key id that must be present in AUTH_TOKEN_SECRETS and used for new tokens.

    Raises:
        RuntimeError: If configuration is missing or invalid.
    """
    secrets_raw = _require_env_str("AUTH_TOKEN_SECRETS")
    primary_id_raw = _require_env_str("AUTH_TOKEN_PRIMARY_KEY_ID")

    secrets: dict[KeyId, bytes] = {}
    for entry in secrets_raw.split(";"):
        parts = entry.split(":", 1)
        if len(parts) != 2:
            raise RuntimeError("AUTH_TOKEN_SECRETS entries must be key_id:base64secret")
        key_id_raw, secret_b64 = parts
        if key_id_raw.strip() == "":
            raise RuntimeError("AUTH_TOKEN_SECRETS key_id cannot be empty")
        key_id = KeyId(key_id_raw)
        secrets[key_id] = _decode_secret(secret_b64)

    primary_id = KeyId(primary_id_raw)
    if primary_id not in secrets:
        raise RuntimeError("primary key id must be present in AUTH_TOKEN_SECRETS")

    return {
        "primary_secret_id": primary_id,
        "secrets": secrets,
    }
```

---

## Implementation Plan

### Phase 1: Core Library (platform_auth)

#### New Library Structure

```
libs/platform_auth/
├── src/platform_auth/
│   ├── __init__.py
│   ├── token_types.py      # Role, Scope, TokenPayload TypedDicts
│   ├── codec.py            # encode_token(), decode_token()
│   ├── errors.py           # TokenError hierarchy
│   ├── fastapi_integration.py  # create_token_dependency(), etc.
│   ├── config.py           # load_auth_config()
│   └── testing.py          # Test utilities for generating tokens
├── tests/
│   ├── test_codec.py
│   ├── test_fastapi_integration.py
│   ├── test_config.py
│   └── test_errors.py
├── pyproject.toml
├── Makefile
└── README.md
```

#### Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.11"
platform_core = { path = "../platform_core", develop = true }

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
pytest-cov = "^4.0"
pytest-xdist = "^3.0"
mypy = "^1.8"
ruff = "^0.2"
```

#### Deliverables

1. `token_types.py` with `Role`, `Scope`, `TokenPayload` definitions
2. `codec.py` with HMAC-SHA256 JWT encode/decode (no external JWT library)
3. `errors.py` with typed exception hierarchy
4. `fastapi_integration.py` with Protocol-based FastAPI dependencies
5. `config.py` with `load_auth_config()`
6. `testing.py` with `make_test_token()` utility
7. 100% branch coverage tests

### Phase 2: Service Migration

#### Migration Order

1. **data-bank-api** (highest traffic, tests new system)
2. **handwriting-ai** (moderately complex, has scope distinctions)
3. **Model-Trainer** (similar to handwriting-ai)
4. **turkic-api** (depends on data-bank-api)
5. **qr-api** (currently unprotected, adds protection)
6. **transcript-api** (currently unprotected, adds protection)

#### Per-Service Changes

##### data-bank-api

**Before (files.py):**
```python
def _ensure_auth(cfg: Settings, perm: Permission, req: Request) -> None:
    allowed = (
        cfg["api_upload_keys"] if perm == "upload"
        else cfg["api_read_keys"] if perm == "read"
        else cfg["api_delete_keys"]
    )
    if len(allowed) == 0:
        return
    key = req.headers.get("X-API-Key")
    if key is None or key.strip() == "":
        raise AppError(ErrorCode.UNAUTHORIZED, "missing API key", 401)
    if key not in allowed:
        raise AppError(ErrorCode.FORBIDDEN, "invalid API key for permission", 403)
```

**After (files.py):**
```python
from platform_auth import KeyId, Scope, TokenPayload, create_scope_dependency

def build_router(storage: Storage, auth_secrets: dict[KeyId, bytes]) -> APIRouter:
    router = APIRouter()

    upload_dep = create_scope_dependency(secrets=auth_secrets, required_scope="databank:upload")
    read_dep = create_scope_dependency(secrets=auth_secrets, required_scope="databank:read")
    delete_dep = create_scope_dependency(secrets=auth_secrets, required_scope="databank:delete")

    def _upload(
        file: Annotated[UploadFile, File(...)],
        token: Annotated[TokenPayload, Depends(upload_dep)],
    ) -> FileUploadResponse:
        # token.subject available for audit logging
        ct = file.content_type or "application/octet-stream"
        meta = storage.save_stream(file.file, ct)
        return {...}
```

**Config changes:**
```python
# config/data_bank.py - simplified
class DataBankSettings(TypedDict):
    redis_url: str
    data_root: str
    min_free_gb: int
    delete_strict_404: bool
    max_file_bytes: int
    # Remove: api_upload_keys, api_read_keys, api_delete_keys
```

##### handwriting-ai

**Before:**
```python
api_key_dep: ApiKeyCheckFn = create_api_key_dependency(
    required_key=api_required_key,
    error_code=ErrorCode.UNAUTHORIZED,
    http_status=401,
)
```

**After:**
```python
from platform_auth import KeyId, create_role_dependency, create_scope_dependency

# Predict endpoint - requires handwriting:predict scope
predict_dep = create_scope_dependency(
    secrets=auth_config["secrets"],
    required_scope="handwriting:predict",
)

# Admin endpoints - requires admin role
admin_dep = create_role_dependency(
    secrets=auth_config["secrets"],
    minimum_role="admin",
)
```

##### Model-Trainer

**Scope Assignments:**
- `POST /runs/train` → `trainer:runs:write`
- `GET /runs/{id}` → `trainer:runs:read`
- `POST /tokenizers/train` → `trainer:tokenizers:write`
- `GET /tokenizers/{id}` → `trainer:tokenizers:read`

##### turkic-api

**Scope Assignments:**
- `POST /corpus/stream` → `turkic:corpus:read`
- `POST /transliterate` → `turkic:transliterate`
- Admin endpoints → `turkic:admin`

##### qr-api (New Protection)

**Add authentication where previously none existed:**
```python
# routes/qr.py
from platform_auth import KeyId, TokenPayload, create_scope_dependency

def build_router(auth_secrets: dict[KeyId, bytes]) -> APIRouter:
    qr_dep = create_scope_dependency(secrets=auth_secrets, required_scope="qr:generate")

    def _generate(
        request: QRRequest,
        token: Annotated[TokenPayload, Depends(qr_dep)],
    ) -> Response:
        ...
```

##### transcript-api (New Protection)

**Add authentication where previously none existed:**
```python
# routes/transcripts.py
from platform_auth import KeyId, TokenPayload, create_scope_dependency

def build_router(auth_secrets: dict[KeyId, bytes]) -> APIRouter:
    captions_dep = create_scope_dependency(secrets=auth_secrets, required_scope="transcript:captions")
    stt_dep = create_scope_dependency(secrets=auth_secrets, required_scope="transcript:stt")
```

### Phase 3: Client Migration

#### DiscordBot Changes

**Before:**
```python
# Multiple separate keys
handwriting_cfg: HandwritingConfig = {
    "api_url": _optional_env_str("HANDWRITING_API_URL"),
    "api_key": _optional_env_str("HANDWRITING_API_KEY"),
    ...
}
```

**After:**
```python
# Single service token with multiple scopes
class ServiceTokenConfig(TypedDict):
    token: str  # JWT with service role and required scopes

service_cfg: ServiceTokenConfig = {
    "token": _require_env_str("SERVICE_AUTH_TOKEN"),
}
```

**HTTP Client Changes:**
```python
# Before
headers["X-API-Key"] = self._api_key

# After
headers["Authorization"] = f"Bearer {self._token}"
```

### Phase 4: Token Management Tooling

#### CLI Tool for Token Generation

```python
# scripts/generate_token.py
"""Generate service tokens for deployment.

Usage:
    python -m scripts.generate_token \
        --subject discordbot \
        --role service \
        --scopes handwriting:predict,trainer:runs:read,qr:generate \
        --expires-days 365 \
        --secret-file /path/to/secret.key
"""

import argparse
import time
import uuid

from platform_auth import ROLE_SCOPES, encode_token, Role, Scope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--role", required=True, choices=["admin", "service", "operator", "reader", "uploader"])
    parser.add_argument("--scopes", required=True, help="Comma-separated scopes")
    parser.add_argument("--expires-days", type=int, default=365)
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--key-id", required=True)
    args = parser.parse_args()

    with open(args.secret_file, "rb") as f:
        secret = f.read()

    scopes = tuple(s.strip() for s in args.scopes.split(",") if s.strip())
    if len(scopes) == 0:
        raise SystemExit("at least one scope is required")
    now = int(time.time())
    expires = now + (args.expires_days * 86400) if args.expires_days > 0 else 0

    role: Role = args.role
    allowed_scopes = ROLE_SCOPES[role]
    for scope in scopes:
        if scope not in allowed_scopes:
            raise SystemExit(f"scope '{scope}' is not permitted for role '{role}'")

    token = encode_token(
        token_id=str(uuid.uuid4()),
        subject=args.subject,
        role=role,
        scopes=scopes,
        issued_at=now,
        expires_at=expires,
        issuer="platform-auth-cli",
        secret=secret,
        key_id=args.key_id,
    )

    print(token)


if __name__ == "__main__":
    main()
```

---

## Migration Strategy

### Phase 1: Library rollout (Week 1)

1. Ship `platform_auth` with strict validation and key-id support.
2. Enable token-only auth on services behind feature flags, but do **not** accept legacy keys.
3. Configure `AUTH_TOKEN_SECRETS` and `AUTH_TOKEN_PRIMARY_KEY_ID` in all environments.

### Phase 2: Service migration (Week 2)

1. Update each service to require bearer tokens via `create_scope_dependency` / `create_role_dependency`.
2. Remove all `X-API-Key` handlers and related environment variables.
3. Deploy services; verify 401/403 behavior with tokens using required scopes/roles.

### Phase 3: Client migration (Week 3)

1. Generate tokens per client using the primary key id and scoped grants.
2. Update clients to send `Authorization: Bearer <token>`.
3. Deploy clients; confirm traffic succeeds only with tokens.

### Phase 4: Rotation-ready posture (Week 4)

1. Keep at least two secrets configured during rotations (primary + previous) using `AUTH_TOKEN_SECRETS`.
2. Mint new tokens with the primary key id; decode allows any configured secret.
3. After propagation, remove retired secrets from configuration and redeploy.

---

## Testing Requirements

### Unit Tests (per module)

1. **codec.py**: Encode/decode round-trip, signature verification, expiration
2. **errors.py**: Exception hierarchy, message formatting
3. **fastapi_integration.py**: Dependency injection, error mapping
4. **config.py**: Environment parsing, validation errors (including malformed secret lists and missing primary id)
5. Header validation: reject bad `kid`, unknown `kid`, bad base64, wrong `alg`/`typ`

### Integration Tests (per service)

1. Valid token -> 200
2. Expired token -> 401
3. Invalid signature -> 401
4. Missing header -> 401
5. Wrong scope -> 403
6. Insufficient role -> 403
7. Unknown key id -> 401

### Coverage Requirements

- **Statements**: 100%
- **Branches**: 100%
- **No coverage exclusions**

### Test Utilities

```python
# libs/platform_auth/src/platform_auth/testing.py

import time
import uuid

from .codec import encode_token
from .token_types import KeyId, Role, Scope, TokenPayload


_TEST_SECRET = b"test-secret-key-32-bytes-minimum"
_TEST_KEY_ID = KeyId("test")


def make_test_token(
    *,
    subject: str = "test-subject",
    role: Role = "admin",
    scopes: tuple[Scope, ...] | None = None,
    expired: bool = False,
) -> str:
    """Generate a test token for integration testing.

    Args:
        subject: Token subject.
        role: Token role.
        scopes: Token scopes (defaults to all scopes for role).
        expired: If True, token is already expired.

    Returns:
        Signed JWT string.
    """
    from .token_types import ROLE_SCOPES

    now = int(time.time())
    if scopes is None:
        scopes = tuple(ROLE_SCOPES[role])

    expires = now - 3600 if expired else now + 3600

    return encode_token(
        token_id=str(uuid.uuid4()),
        subject=subject,
        role=role,
        scopes=scopes,
        issued_at=now,
        expires_at=expires,
        issuer="test",
        secret=_TEST_SECRET,
        key_id=_TEST_KEY_ID,
    )


def get_test_secret() -> bytes:
    """Get the test secret for token verification."""
    return _TEST_SECRET


def get_test_secrets() -> dict[KeyId, bytes]:
    """Get the secret mapping for token verification."""
    return {_TEST_KEY_ID: _TEST_SECRET}
```

---

## Security Considerations

### Token Storage

- Tokens are stored in environment variables (same as current keys)
- No change to secrets management infrastructure required
- Consider secrets manager integration as future enhancement

### Secret Rotation

1. Generate a new secret and append it to `AUTH_TOKEN_SECRETS` with a distinct key id; set `AUTH_TOKEN_PRIMARY_KEY_ID` to the new id.
2. Deploy to all services (they verify with any configured secret while minting only with the primary).
3. Regenerate all client tokens using the primary key id.
4. Deploy client updates.
5. Remove retired secrets from `AUTH_TOKEN_SECRETS` once no tokens reference them, then redeploy.

### Scope Enforcement

- Scopes are checked at route level, not middleware
- Failed scope checks return 403 (not 401) to distinguish from auth failure
- Audit logs include token subject and scope for incident response

### Token Revocation

Current design: No revocation (tokens valid until expiry)

Future enhancement options:
1. Short token lifetimes (1 hour) with refresh mechanism
2. Token ID blocklist in Redis
3. Version number in token, increment to invalidate all tokens

---

## Code Standards Compliance

### Typing Rules

- All types explicit, no `Any`
- No `cast()` usage
- No `type: ignore` comments
- No `.pyi` stub files
- TypedDict for all structured data
- Protocol for all interfaces

### Error Handling

- Errors propagate (no try/except for recovery)
- All failure modes documented in docstrings
- Test expectations cover all error branches

### Testing

- 100% statement coverage
- 100% branch coverage
- No `noqa` comments
- No coverage exclusions

### Pattern Compliance

- `__import__()` for dynamic module loading
- `getattr()` with Protocol type annotation
- JSON parsing via `platform_core.json_utils`
- No Pydantic, no dataclasses in src/

---

## Rollback Plan

If issues are discovered after deployment:

1. **Immediate**: Roll back to previous deploy if token validation regression discovered.
2. **Short-term**: Add the prior signing secret to `AUTH_TOKEN_SECRETS` while keeping the current primary; redeploy to accept both while issuing only with the primary.
3. **Investigation**: Check logs for token decode errors, scope mismatches, or forbidden role/scope attempts.
4. **Fix forward**: Patch and redeploy with corrected validation; remove retired secrets after verification.

---

## Success Metrics

1. **100% test coverage** for platform_auth library
2. **No auth-related incidents** in first 30 days post-migration
3. **Reduced key count**: From ~15 separate keys to a single rotating secret set + scoped tokens
4. **Successful secret rotations** completed without downtime

---

## Appendix A: Full Type Definitions

```python
# Complete Role and Scope definitions

from typing import Literal

Role = Literal["admin", "service", "operator", "reader", "uploader"]

Scope = Literal[
    # data-bank-api
    "databank:upload",
    "databank:read",
    "databank:delete",
    "databank:admin",

    # handwriting-ai
    "handwriting:predict",
    "handwriting:admin",
    "handwriting:models:read",
    "handwriting:models:write",

    # Model-Trainer
    "trainer:runs:read",
    "trainer:runs:write",
    "trainer:tokenizers:read",
    "trainer:tokenizers:write",
    "trainer:admin",

    # turkic-api
    "turkic:corpus:read",
    "turkic:corpus:write",
    "turkic:transliterate",
    "turkic:admin",

    # qr-api
    "qr:generate",
    "qr:admin",

    # transcript-api
    "transcript:captions",
    "transcript:stt",
    "transcript:admin",
]
```

---

## Appendix B: Environment Variable Migration

### Before (Current State)

```bash
# data-bank-api
API_UPLOAD_KEYS=key1,key2
API_READ_KEYS=key1,key2,key3
API_DELETE_KEYS=key1

# handwriting-ai
HANDWRITING_API_KEY=secret123
SECURITY__API_KEY=secret123

# Model-Trainer
SECURITY__API_KEY=trainer-key

# turkic-api
TURKIC_DATA_BANK_API_KEY=key1

# DiscordBot
HANDWRITING_API_KEY=secret123
MODEL_TRAINER_API_KEY=trainer-key
```

### After (New State)

```bash
# All services (shared, rotation-friendly)
AUTH_TOKEN_SECRETS=primary:<base64-encoded-32-byte-secret>
AUTH_TOKEN_PRIMARY_KEY_ID=primary

# DiscordBot (example)
SERVICE_AUTH_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Appendix C: Audit Log Format

```json
{
  "timestamp": "2025-11-28T10:30:00Z",
  "level": "INFO",
  "service": "data-bank-api",
  "event": "file_uploaded",
  "request_id": "req-abc123",
  "token_subject": "discordbot",
  "token_role": "service",
  "token_scope_used": "databank:upload",
  "file_id": "file-xyz789",
  "size_bytes": 1048576
}
```
