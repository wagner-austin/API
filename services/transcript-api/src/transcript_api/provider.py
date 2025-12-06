from __future__ import annotations

from typing import Protocol, runtime_checkable

from platform_core.errors import AppError, TranscriptErrorCode
from platform_core.logging import get_logger

from .types import (
    DEFAULT_TRANSCRIPT_LANGS,
    RawTranscriptItem,
    TranscriptOptions,
    TranscriptSegment,
)


class DirectTranscriptUnavailableError(Exception):
    """Raised by clients when direct transcripts are not available."""


class TranscriptListingError(Exception):
    """Raised by clients when listing transcripts for a video fails."""


class TranscriptLanguageUnavailableError(Exception):
    """Raised when no transcript exists for a preferred language."""


class TranscriptTranslateUnavailableError(Exception):
    """Raised when translation fallback for transcripts is not available."""


class _DirectNotFoundError(Exception):
    """Internal sentinel to signal direct transcript unavailability."""


@runtime_checkable
class TranscriptProvider(Protocol):
    def fetch(self, video_id: str, opts: TranscriptOptions) -> list[TranscriptSegment]: ...


@runtime_checkable
class TranscriptResource(Protocol):
    def fetch(self) -> list[RawTranscriptItem]: ...


@runtime_checkable
class TranscriptListing(Protocol):
    def find_transcript(self, languages: list[str]) -> TranscriptResource | None: ...
    def translate(self, language: str) -> TranscriptResource: ...


@runtime_checkable
class YouTubeTranscriptClient(Protocol):
    """Abstraction over the underlying YouTube transcript backend."""

    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]: ...

    def list_transcripts(self, video_id: str) -> TranscriptListing: ...


class YouTubeTranscriptProvider:
    def __init__(self, api: YouTubeTranscriptClient) -> None:
        self._logger = get_logger(__name__)
        self._api = api

    def fetch(self, video_id: str, opts: TranscriptOptions) -> list[TranscriptSegment]:
        langs = opts["preferred_langs"] or DEFAULT_TRANSCRIPT_LANGS
        try:
            raw = self._fetch_raw(video_id, langs)
        except _DirectNotFoundError as exc:
            self._logger.info(
                "Direct transcript fetch unavailable; attempting translate fallback: %s", exc
            )
            raw_fallback = self._fallback_listing(video_id, langs)
            out_fb: list[TranscriptSegment] = []
            for item in raw_fallback:
                text = item["text"]
                if not text.strip():
                    continue
                start = item["start"]
                duration = item["duration"]
                out_fb.append(TranscriptSegment(text=text, start=start, duration=duration))
            return out_fb

        out: list[TranscriptSegment] = []
        for item in raw:
            text = item["text"]
            if not text.strip():
                continue
            start = item["start"]
            duration = item["duration"]
            out.append(TranscriptSegment(text=text, start=start, duration=duration))
        return out

    def _fetch_raw(self, video_id: str, langs: list[str]) -> list[RawTranscriptItem]:
        try:
            raw_obj: list[RawTranscriptItem] = self._api.get_transcript(video_id, languages=langs)
        except DirectTranscriptUnavailableError as exc:
            self._logger.info("Direct transcript unavailable for %s: %s", video_id, exc)
            raise _DirectNotFoundError("no direct transcript") from None
        return raw_obj

    def _fallback_listing(self, video_id: str, langs: list[str]) -> list[RawTranscriptItem]:
        listing = self._list_transcripts(video_id)
        try:
            chosen = self._choose_transcript(listing, langs)
        except AppError:
            try:
                chosen = listing.translate(langs[0])
            except TranscriptTranslateUnavailableError as exc:
                self._logger.info("Translate fallback unavailable for lang %s: %s", langs[0], exc)
                raise AppError(
                    TranscriptErrorCode.TRANSCRIPT_TRANSLATE_UNAVAILABLE,
                    "No transcript is available for the preferred language",
                    400,
                ) from None
        if chosen is None:
            raise AppError(
                TranscriptErrorCode.TRANSCRIPT_UNAVAILABLE,
                "No transcript is available for this video",
                400,
            ) from None
        return chosen.fetch()

    def _list_transcripts(self, video_id: str) -> TranscriptListing:
        try:
            listing = self._api.list_transcripts(video_id)
        except TranscriptListingError as exc:
            self._logger.info("Transcript listing failed for %s: %s", video_id, exc)
            raise AppError(
                TranscriptErrorCode.TRANSCRIPT_LISTING_FAILED,
                "The video is unavailable or transcripts are disabled",
                400,
            ) from None
        return listing

    def _choose_transcript(
        self, listing: TranscriptListing, langs: list[str]
    ) -> TranscriptResource | None:
        for code in langs:
            try:
                transcript = listing.find_transcript([code])
            except TranscriptLanguageUnavailableError as exc:
                self._logger.info("No transcript in language %s (%s)", code, exc)
                raise AppError(
                    TranscriptErrorCode.TRANSCRIPT_LANGUAGE_UNAVAILABLE,
                    "No transcript in the preferred languages",
                    400,
                ) from None
            if transcript is not None:
                return transcript
        return None


def _as_float(
    val: str | int | float | bool | None | dict[str, str | int | float] | list[str | int | float],
) -> float:
    """Convert value to float with runtime validation.

    Accepts int, float, or str. Returns 0.0 for invalid types/values.
    """
    if isinstance(val, int | float):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        try:
            return float(s)
        except ValueError as exc:
            get_logger(__name__).warning("Invalid float value: %r (%s)", val, exc)
            return 0.0
    return 0.0


def _coerce_raw_items(
    obj: (
        list[dict[str, str | int | float] | str | int | float | None]
        | dict[str, str | int | float]
        | str
        | int
        | float
        | None
    ),
) -> list[RawTranscriptItem]:
    """Validate and coerce raw transcript items with runtime type checking.

    Accepts list of dict-like items. Raises UserInputError for invalid structure.
    """
    if not isinstance(obj, list):
        raise AppError(
            TranscriptErrorCode.TRANSCRIPT_PAYLOAD_INVALID,
            "Unexpected transcript payload format",
            400,
        )

    out: list[RawTranscriptItem] = []
    for item in obj:
        # The RawTranscriptItem is already a TypedDict, so we can assume its structure
        # Further validation could be added if needed, but for now, rely on its type.
        if not isinstance(item, dict):
            # This case should ideally not happen if the upstream is correctly typed
            # but is kept for robustness against dynamic input or incorrect upstream types.
            get_logger(__name__).warning("Unexpected item type in RawTranscriptItem list")
            continue

        # Explicitly validate the types within the RawTranscriptItem if necessary.
        # For now, we trust the RawTranscriptItem TypedDict definition.
        text_str = item.get("text", "")
        start_v = _as_float(item.get("start", 0.0))
        dur_v = _as_float(item.get("duration", 0.0))

        # This will construct a new RawTranscriptItem, ensuring type correctness
        typed_item: RawTranscriptItem = {
            "text": str(text_str),  # Ensure it's string
            "start": float(start_v),  # Ensure it's float
            "duration": float(dur_v),  # Ensure it's float
        }
        out.append(typed_item)
    return out
