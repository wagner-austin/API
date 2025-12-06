from __future__ import annotations

from collections.abc import Mapping


def add_correlation_header(
    headers: Mapping[str, str] | None,
    request_id: str,
    *,
    header_name: str = "X-Request-ID",
) -> dict[str, str]:
    """Return a copy of headers including a correlation header.

    Accepts any mapping for input and returns a new plain dict with the
    correlation header set to the provided request_id.
    """
    base: dict[str, str] = dict(headers or {})
    base[header_name] = request_id
    return base


__all__ = ["add_correlation_header"]
