"""The native runtime transport seams, using the fleet's existing SSH route."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from platform_core.json_utils import JSONObject

from fleet.core.native_rpc import attempt_read_threads
from fleet.core.native_turns import WakeDelivery, WakeRequest, attempt_deliver_wake


class ReadThreadsProtocol(Protocol):
    """Read only metadata for the supplied registered thread identities."""

    def __call__(
        self, argv: Sequence[str], thread_ids: tuple[str, ...], *, timeout_seconds: int
    ) -> tuple[tuple[JSONObject | str, ...], str | None]:
        """Return native thread objects or propagate a transport/contract fault."""


class DeliverWakeProtocol(Protocol):
    """Read one thread's open turns and deliver its wake when it is idle."""

    def __call__(
        self,
        argv: Sequence[str],
        thread_id: str,
        wake: WakeRequest | None,
        open_turns: tuple[str, ...],
        *,
        timeout_seconds: int,
    ) -> tuple[WakeDelivery | None, str | None]:
        """Return what was done, or a named transport or refusal failure."""


read: ReadThreadsProtocol = attempt_read_threads
deliver: DeliverWakeProtocol = attempt_deliver_wake

__all__ = ["DeliverWakeProtocol", "ReadThreadsProtocol", "deliver", "read"]
