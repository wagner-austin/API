"""Type definitions for grandma-api.

Re-exports types from api/schemas for convenience.
"""

from __future__ import annotations

from grandma_api.api.schemas.translate import (
    TranslationResponse,
    decode_translation_response,
    encode_translation_response,
    require_translation_response,
)

__all__ = [
    "TranslationResponse",
    "decode_translation_response",
    "encode_translation_response",
    "require_translation_response",
]
