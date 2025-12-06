from __future__ import annotations

import binascii
import struct
import zlib
from typing import Protocol

from platform_music.models import WrappedResult


class RendererProto(Protocol):
    def render_wrapped(self, result: WrappedResult) -> bytes: ...


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = binascii.crc32(chunk_type + data) & 0xFFFFFFFF
    crc_bytes = struct.pack(">I", crc)
    return length + chunk_type + data + crc_bytes


def _png_from_rgb(width: int, height: int, r: int, g: int, b: int) -> bytes:
    # PNG signature
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    # IHDR: width, height, bit depth 8,
    # color type 2 (truecolor), compression 0, filter 0, interlace 0
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out += _png_chunk(b"IHDR", ihdr)
    # Image data: each row starts with filter type 0
    row = bytes([0] + [r, g, b] * width)
    data = b"".join(row for _ in range(height))
    comp = zlib.compress(data)
    out += _png_chunk(b"IDAT", comp)
    out += _png_chunk(b"IEND", b"")
    return bytes(out)


class _SimpleRenderer:
    def render_wrapped(self, result: WrappedResult) -> bytes:
        # Derive a simple color from key fields to avoid placeholders
        base = (result["year"] + result["total_scrobbles"]) % 255
        r = (base + 40) % 255
        g = (base + 90) % 255
        b = (base + 140) % 255
        return _png_from_rgb(2, 2, r, g, b)


def build_renderer() -> RendererProto:
    return _SimpleRenderer()


__all__ = ["RendererProto", "build_renderer"]
