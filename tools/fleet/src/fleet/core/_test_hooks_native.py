"""The native runtime transport seam, using the fleet's existing SSH route."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from platform_core.json_utils import JSONObject

from fleet.core.native_rpc import attempt_read_threads


class ReadThreadsProtocol(Protocol):
    """Read only metadata for the supplied registered thread identities."""

    def __call__(
        self, argv: Sequence[str], thread_ids: tuple[str, ...], *, timeout_seconds: int
    ) -> tuple[tuple[JSONObject | str, ...], str | None]:
        """Return native thread objects or propagate a transport/contract fault."""


read: ReadThreadsProtocol = attempt_read_threads

__all__ = ["ReadThreadsProtocol", "read"]
