from __future__ import annotations

__all__ = [
    "DForbiddenError",
    "DHTTPExceptionError",
    "DNotFoundError",
]


class DHTTPExceptionError(Exception):
    pass


class DForbiddenError(Exception):
    pass


class DNotFoundError(Exception):
    pass
