from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue, load_json_bytes, load_json_str

from platform_music.models import PlayRecord, Track
from platform_music.services.protocol import MusicServiceProto


def _extract_video_id(url: str) -> str | None:
    # Support common formats: watch?v=ID, youtu.be/ID
    vpos = url.find("v=")
    if vpos != -1:
        tail = url[vpos + 2 :]
        amp = tail.find("&")
        return tail[:amp] if amp != -1 else tail
    # short form: last path segment
    slash = url.rfind("/")
    if slash != -1:
        seg = url[slash + 1 :]
        if seg:
            return seg
    return None


def _decode_takeout_entry(raw: JSONValue) -> PlayRecord | None:
    """Decode a single Google Takeout watch history entry to PlayRecord.

    Returns None for non-YouTube Music entries to allow filtering by product.
    Raises AppError on structural errors for candidate entries.
    """
    if not isinstance(raw, dict):
        raise AppError(ErrorCode.INVALID_INPUT, "takeout item must be an object", 400)

    products = raw.get("products")
    if not (isinstance(products, list) and any(p == "YouTube Music" for p in products)):
        # Not a YT Music entry - ignore
        return None

    title = raw.get("title")
    played_at = raw.get("time")
    subs = raw.get("subtitles")
    url_val = raw.get("titleUrl")

    if not isinstance(title, str) or title.strip() == "":
        raise AppError(ErrorCode.INVALID_INPUT, "title must be non-empty string", 400)
    if not isinstance(played_at, str) or played_at.strip() == "":
        raise AppError(ErrorCode.INVALID_INPUT, "time must be non-empty ISO string", 400)

    artist_name: str
    if isinstance(subs, list) and len(subs) > 0 and isinstance(subs[0], dict):
        nm = subs[0].get("name")
        artist_name = nm if isinstance(nm, str) and nm.strip() != "" else "Unknown"
    else:
        artist_name = "Unknown"

    vid: str | None = url_val if isinstance(url_val, str) else None
    if vid is not None:
        vid = _extract_video_id(vid) or vid

    track: Track = {
        "id": vid if isinstance(vid, str) and vid != "" else "yt:" + title,
        "title": title,
        "artist_name": artist_name,
        # Duration is not available in Takeout; default to 0
        "duration_ms": 0,
        "service": "youtube_music",
    }
    return {"track": track, "played_at": played_at, "service": "youtube_music"}


def _expect_list(doc: JSONValue) -> list[JSONValue]:
    if not isinstance(doc, list):
        raise AppError(ErrorCode.INVALID_INPUT, "takeout root must be a list", 400)
    return doc


def decode_takeout_json(doc: Sequence[JSONValue]) -> list[PlayRecord]:
    """Decode a Takeout watch history JSON value into PlayRecords.

    Accepts the exported list from Google Takeout. Filters entries to those
    explicitly marked as coming from YouTube Music via the `products` field.
    """
    out: list[PlayRecord] = []
    for item in doc:
        rec = _decode_takeout_entry(item)
        if rec is not None:
            out.append(rec)
    return out


class _BytesIOLike(Protocol):
    pass


class _BytesIOFactory(Protocol):
    def __call__(self, initial_bytes: bytes) -> _BytesIOLike: ...


class _ZipFileLike(Protocol):
    def namelist(self) -> list[str]: ...

    def read(self, name: str) -> bytes: ...

    def close(self) -> None: ...


class _ZipFileFactory(Protocol):
    def __call__(self, file: _BytesIOLike) -> _ZipFileLike: ...


def parse_takeout_bytes(raw: bytes, *, content_type: str) -> list[PlayRecord]:
    """Parse YouTube Takeout file content (JSON or ZIP) to PlayRecords.

    content_type is used as a hint; ZIP signature is detected from bytes.
    """
    # ZIP local file header signature
    is_zip = len(raw) >= 4 and raw[:4] == b"PK\x03\x04"
    if is_zip or content_type.lower() in ("application/zip", "application/x-zip-compressed"):
        # Parse first matching JSON file that likely contains watch history
        imp = __import__
        io_mod = imp("io")
        zipfile_mod = imp("zipfile")
        bytesio: _BytesIOFactory = io_mod.BytesIO
        zip_ctor: _ZipFileFactory = zipfile_mod.ZipFile
        b = bytesio(raw)
        zf: _ZipFileLike = zip_ctor(b)
        try:
            names: list[str] = zf.namelist()
            # Prefer files with both 'watch' and 'history' in name; fall back to '*history*.json'
            candidate: str | None = None
            for name in names:
                low = name.lower()
                if low.endswith(".json") and "history" in low and "youtube" in low:
                    # Take YouTube and YouTube Music history exports first
                    candidate = name
                    break
            if candidate is None:
                for name in names:
                    low = name.lower()
                    if low.endswith(".json") and "history" in low:
                        candidate = name
                        break
            if candidate is None:
                raise AppError(ErrorCode.INVALID_INPUT, "no history json found in zip", 400)
            data = zf.read(candidate)
        finally:
            zf.close()
        doc = load_json_bytes(data)
        return decode_takeout_json(_expect_list(doc))

    # Treat as raw JSON
    text = raw.decode("utf-8")
    doc2 = load_json_str(text)
    return decode_takeout_json(_expect_list(doc2))


class _StaticHistoryService(MusicServiceProto):
    """In-memory history provider implementing MusicServiceProto.

    Filters by ISO-formatted UTC time bounds using lexicographic comparison.
    """

    def __init__(self, plays: list[PlayRecord]) -> None:
        # Store a shallow copy to prevent mutation by callers
        self._plays: list[PlayRecord] = list(plays)

    def get_listening_history(
        self, *, start_date: str, end_date: str, limit: int | None = None
    ) -> list[PlayRecord]:
        filtered = [p for p in self._plays if start_date <= p["played_at"] <= end_date]
        if limit is not None:
            filtered = filtered[:limit]
        return list(filtered)


def static_service_from_plays(plays: list[PlayRecord]) -> MusicServiceProto:
    """Build a MusicServiceProto that serves the provided PlayRecords."""
    return _StaticHistoryService(plays)


def decode_stored_plays(doc: JSONValue) -> list[PlayRecord]:
    """Validate a JSONValue loaded from Redis as a list[PlayRecord]."""
    if not isinstance(doc, list):
        raise AppError(ErrorCode.INVALID_INPUT, "stored plays must be a list", 400)
    out: list[PlayRecord] = []
    for i, item in enumerate(doc):
        if not isinstance(item, dict):
            raise AppError(ErrorCode.INVALID_INPUT, f"play {i} must be object", 400)
        track_val = item.get("track")
        played_at = item.get("played_at")
        service = item.get("service")
        if not isinstance(track_val, dict):
            raise AppError(ErrorCode.INVALID_INPUT, f"play {i} missing track", 400)
        tid = track_val.get("id")
        ttl = track_val.get("title")
        art = track_val.get("artist_name")
        dur = track_val.get("duration_ms")
        svc2 = track_val.get("service")
        if not (
            isinstance(tid, str)
            and isinstance(ttl, str)
            and isinstance(art, str)
            and isinstance(dur, int)
            and isinstance(played_at, str)
            and isinstance(service, str)
            and isinstance(svc2, str)
        ):
            raise AppError(ErrorCode.INVALID_INPUT, f"play {i} has invalid fields", 400)
        track: Track = {
            "id": tid,
            "title": ttl,
            "artist_name": art,
            "duration_ms": int(dur),
            "service": "youtube_music",
        }
        out.append({"track": track, "played_at": played_at, "service": "youtube_music"})
    return out


__all__ = [
    "decode_stored_plays",
    "decode_takeout_json",
    "parse_takeout_bytes",
    "static_service_from_plays",
]
