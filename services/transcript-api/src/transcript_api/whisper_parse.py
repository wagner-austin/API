from __future__ import annotations

from platform_core.logging import get_logger

from .types import (
    SupportsModelDump,
    SupportsToDictRecursive,
    TranscriptSegment,
    VerboseResponseTD,
    VerboseSegmentTD,
)

RawVerboseBase = dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]
RawVerboseExtended = dict[
    str,
    str | int | float | bool | None | list[dict[str, str | int | float | dict[str, int]] | int],
]
RawVerboseAny = RawVerboseBase | RawVerboseExtended


def _coerce_verbose_response(raw: RawVerboseAny) -> VerboseResponseTD:
    """Validate and coerce a raw dict into a VerboseResponseTD.

    Performs strict runtime validation - raises ValueError on any invalid structure.
    """
    text_val = raw.get("text")
    if not isinstance(text_val, str):
        raise ValueError("verbose response missing text")
    segs_val = raw.get("segments")
    if not isinstance(segs_val, list):
        raise ValueError("verbose response missing segments list")
    segs: list[VerboseSegmentTD] = []
    for it in segs_val:
        if not isinstance(it, dict):
            raise ValueError("segment must be object")
        text = str(it.get("text", ""))
        # Do not filter empty text here; keep encoder pure
        start_any = it.get("start")
        end_any = it.get("end")
        if start_any is None or end_any is None:
            raise ValueError("segment missing start/end")
        if not isinstance(start_any, int | float | str):
            raise ValueError("segment start must be numeric")
        if not isinstance(end_any, int | float | str):
            raise ValueError("segment end must be numeric")
        start = _as_float(start_any)
        end = _as_float(end_any)
        segs.append({"text": text, "start": start, "end": end})
    return {"text": text_val, "segments": segs}


def to_verbose_dict(
    obj: (RawVerboseBase | SupportsToDictRecursive | SupportsModelDump | list[str]),
) -> VerboseResponseTD:
    """Convert OpenAI SDK response into a strictly typed verbose response.

    Accepts dict-compatible payloads exposed by common SDKs via methods like
    `to_dict_recursive()` or `model_dump()`; otherwise raises ValueError.
    Validates input at runtime - any invalid structure raises ValueError.
    """
    if isinstance(obj, SupportsToDictRecursive):
        return _coerce_verbose_response(obj.to_dict_recursive())
    if isinstance(obj, SupportsModelDump):
        return _coerce_verbose_response(obj.model_dump())
    if isinstance(obj, dict):
        return _coerce_verbose_response(obj)
    raise ValueError("Unsupported verbose object")


def convert_verbose_to_segments(data: VerboseResponseTD) -> list[TranscriptSegment]:
    segs_raw: list[VerboseSegmentTD] = data["segments"]
    out: list[TranscriptSegment] = []
    for item in segs_raw:
        text = str(item["text"]).strip()
        if not text:
            continue
        start = float(item["start"])  # already coerced
        end = float(item["end"])  # already coerced
        duration = max(0.0, end - start)
        out.append(TranscriptSegment(text=text, start=start, duration=duration))
    return out


def _as_float(val: int | float | str | None) -> float:
    if isinstance(val, int | float):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if s and _is_numeric_str(s):
            return float(s)
    return 0.0


def _is_numeric_str(s: str) -> bool:
    digits = set("0123456789")
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
        if not s:
            return False
    dot_seen = False
    digit_seen = True  # A numeric string must have at least one digit
    for ch in s:
        if ch == ".":
            if dot_seen:
                return False
            dot_seen = True
        elif ch in digits:
            digit_seen = True
        else:
            return False
    return digit_seen


logger = get_logger(__name__)
