"""Translation backends for platform_translate.

Provides pluggable translation backends with a common protocol.
"""

from platform_translate.backends.anthropic import AnthropicBackend
from platform_translate.backends.protocol import TranslationBackendProtocol

__all__ = [
    "AnthropicBackend",
    "TranslationBackendProtocol",
]
