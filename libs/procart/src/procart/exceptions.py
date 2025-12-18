from __future__ import annotations


class ConfigError(Exception):
    """Configuration error for procart typed decoders.

    Raised when a Python config object fails validation in an internal decoder
    (e.g., missing keys, wrong types, or unknown selector values). These errors
    are considered user/configuration faults and should not be caught in core
    logic; allow them to propagate.

    Raises:
        ConfigError: Always raised by decoders when a configuration is invalid.
    """

    pass


__all__ = ["ConfigError"]
