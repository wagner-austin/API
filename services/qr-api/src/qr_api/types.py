from __future__ import annotations

from enum import StrEnum
from typing import TypedDict


class ECCLevel(StrEnum):
    """QR error-correction level, as segno and the request body spell it."""

    L = "L"
    M = "M"
    Q = "Q"
    H = "H"


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
