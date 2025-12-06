from __future__ import annotations

import pytest
from platform_core.errors import AppError, TranscriptErrorCode

from transcript_api.youtube import canonicalize_youtube_url, extract_video_id, validate_youtube_url


def test_extract_watch_and_shorts_and_be_hosts() -> None:
    vid = "dQw4w9WgXcQ"
    assert extract_video_id(f"https://www.youtube.com/watch?v={vid}") == vid
    assert extract_video_id(f"https://www.youtube.com/shorts/{vid}") == vid
    assert extract_video_id(f"https://youtu.be/{vid}") == vid


def test_extract_invalid_hosts_and_ids() -> None:
    with pytest.raises(AppError) as exc1:
        _ = extract_video_id("")
    assert exc1.value.code is TranscriptErrorCode.YOUTUBE_URL_REQUIRED
    with pytest.raises(AppError) as exc2:
        _ = extract_video_id("https://example.com/x")
    assert exc2.value.code is TranscriptErrorCode.YOUTUBE_URL_UNSUPPORTED
    with pytest.raises(AppError) as exc3:
        _ = extract_video_id("https://www.youtube.com/watch?v=too_short")
    assert exc3.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID


def test_extract_handles_valueerror_from_urlsplit() -> None:
    # This produces a ValueError("Invalid IPv6 URL") inside urllib
    with pytest.raises(AppError) as exc:
        _ = extract_video_id("https://[::1")
    assert exc.value.code is TranscriptErrorCode.YOUTUBE_URL_INVALID


def test_validate_and_canonicalize() -> None:
    vid = "dQw4w9WgXcQ"
    canon = canonicalize_youtube_url(f"https://youtu.be/{vid}")
    assert canon == f"https://www.youtube.com/watch?v={vid}"
    out = validate_youtube_url(f"https://www.youtube.com/watch?v={vid}")
    assert out == f"https://www.youtube.com/watch?v={vid}"


def test_watch_missing_v_param_raises() -> None:
    # Exercise _extract_watch_id branch where query param is missing
    with pytest.raises(AppError) as exc:
        _ = extract_video_id("https://www.youtube.com/watch")
    assert exc.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID


def test_youtube_non_matching_path_segments_raises() -> None:
    # Not watch/shorts/live; leaves vid None
    with pytest.raises(AppError) as exc:
        _ = extract_video_id("https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw")
    assert exc.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID


def test_youtu_be_missing_id_raises() -> None:
    with pytest.raises(AppError) as exc:
        _ = extract_video_id("https://youtu.be/")
    assert exc.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID
