"""The one error type the fleet's HTTP layer maps to a 4xx.

Its own module so that :mod:`tankpit_bot.service.fleet_config` can
raise it without importing the registry that also raises it -- the two
would otherwise cycle.
"""

from __future__ import annotations


class FleetError(RuntimeError):
    """A fleet operation the HTTP layer maps to a 4xx response."""


__all__ = [
    "FleetError",
]
