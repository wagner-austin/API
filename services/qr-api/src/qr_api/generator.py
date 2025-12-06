from __future__ import annotations

import io
from importlib import import_module
from types import ModuleType
from typing import BinaryIO, Protocol, TypeGuard

from .types import QROptions


class _SegnoQRCode(Protocol):
    def save(
        self,
        out: BinaryIO | io.BytesIO | str,
        *,
        kind: str,
        scale: int,
        border: int,
        dark: str,
        light: str,
    ) -> None:
        """Write the code to PNG."""


class _SegnoModule(Protocol):
    def make(self, content: str, *, error: str) -> _SegnoQRCode:
        """Create a QR code object."""


def _is_segno_module(candidate: ModuleType) -> TypeGuard[_SegnoModule]:
    return hasattr(candidate, "make")


def _load_segno_module() -> _SegnoModule:
    module = import_module("segno")
    if not _is_segno_module(module):
        raise RuntimeError("segno module does not expose make()")
    return module


def generate_png(opts: QROptions) -> bytes:
    module = _load_segno_module()
    qr = module.make(opts["url"], error=opts["ecc"])
    buf = io.BytesIO()
    qr.save(
        buf,
        kind="png",
        scale=opts["box_size"],
        border=opts["border"],
        dark=opts["fill_color"],
        light=opts["back_color"],
    )
    return buf.getvalue()


__all__ = ["generate_png"]
