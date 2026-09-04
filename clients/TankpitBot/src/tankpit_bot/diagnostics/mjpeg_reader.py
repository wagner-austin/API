"""Split a multipart MJPEG byte stream into frames, incrementally.

Separate from :mod:`tankpit_bot.diagnostics.stream_quality` because the
two answer different questions and fail differently. This one is about
the wire format and is exercised against bytes; the other is about what
the numbers mean.

Incremental by construction: a live-view stream never ends, so a reader
that needs the whole body first cannot read one at all. Feed whatever
arrives, take whatever frames completed, keep the remainder.
"""

from __future__ import annotations

_JPEG_MAGIC = b"\xff\xd8\xff"
_HEADER_END = b"\r\n\r\n"


def frames_from_buffer(buffer: bytes, boundary: bytes) -> tuple[list[bytes], bytes]:
    """Take every complete frame out of ``buffer``.

    A part is complete only once the NEXT boundary has arrived, because
    that is what proves no more body bytes are coming for it. The final
    partial part is therefore returned as the remainder rather than
    yielded, which is what stops a truncated JPEG being counted as a
    frame and skewing every number downstream.

    Args:
        buffer: Bytes received so far, starting at or before a boundary.
        boundary: The multipart boundary token including its leading
            dashes, e.g. ``b"--tankpitbotframe"``.

    Returns:
        The complete frames in order, and the unconsumed remainder to
        pass back in with the next read.

    Raises:
        ValueError: If the boundary is empty. A caller that passes one
            would otherwise get an infinite supply of zero-length
            frames rather than an error.
    """
    if not boundary:
        raise ValueError("boundary must not be empty")

    frames: list[bytes] = []
    rest = buffer
    while True:
        start = rest.find(boundary)
        if start == -1:
            return frames, rest
        following = rest.find(boundary, start + len(boundary))
        if following == -1:
            return frames, rest[start:]
        part = rest[start + len(boundary) : following]
        header_end = part.find(_HEADER_END)
        if header_end != -1:
            body = part[header_end + len(_HEADER_END) :].rstrip(b"\r\n")
            if body.startswith(_JPEG_MAGIC):
                frames.append(body)
        rest = rest[following:]


def boundary_from_content_type(content_type: str) -> bytes:
    """Extract the multipart boundary token from a ``Content-Type``.

    The token is the sender's, never reconstructed: a reader that
    assumes a boundary cannot split a stream produced with a different
    one, and the failure looks like an empty stream rather than a
    mismatch.

    Args:
        content_type: The header value, e.g.
            ``"multipart/x-mixed-replace; boundary=frame42"``.

    Returns:
        The boundary with its leading dashes, ready for
        :func:`frames_from_buffer`.

    Raises:
        ValueError: If the header carries no ``boundary=`` parameter.
    """
    marker = "boundary="
    index = content_type.find(marker)
    if index == -1:
        raise ValueError(f"no boundary in content type {content_type!r}")
    token = content_type[index + len(marker) :].split(";")[0].strip().strip('"')
    if not token:
        raise ValueError(f"empty boundary in content type {content_type!r}")
    return b"--" + token.encode("ascii")


__all__ = [
    "boundary_from_content_type",
    "frames_from_buffer",
]
