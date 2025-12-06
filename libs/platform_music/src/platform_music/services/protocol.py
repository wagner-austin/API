from __future__ import annotations

from typing import Protocol, runtime_checkable

from platform_music.models import PlayRecord


@runtime_checkable
class MusicServiceProto(Protocol):
    """Abstract protocol for a music streaming service client.

    All implementations must return strictly-typed records with no Any.
    """

    def get_listening_history(
        self,
        *,
        start_date: str,
        end_date: str,
        limit: int | None = None,
    ) -> list[PlayRecord]:
        """Fetch listening history within the inclusive date range."""
        ...
