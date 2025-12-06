from __future__ import annotations

from typing import Literal, TypedDict

ECCLevel = Literal["L", "M", "Q", "H"]


class QRPayload(TypedDict, total=False):
    url: str
    ecc: ECCLevel
    box_size: int
    border: int
    fill_color: str
    back_color: str


class QROptions(TypedDict, total=True):
    url: str
    ecc: ECCLevel
    box_size: int
    border: int
    fill_color: str  # hex #RGB or #RRGGBB only (validated)
    back_color: str  # hex #RGB or #RRGGBB only (validated)


__all__ = [
    "ECCLevel",
    "QROptions",
    "QRPayload",
]
