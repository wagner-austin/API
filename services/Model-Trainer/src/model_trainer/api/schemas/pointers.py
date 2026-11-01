from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class ArtifactPointer(TypedDict, total=True):
    storage: Literal["local"]
    file_id: str
