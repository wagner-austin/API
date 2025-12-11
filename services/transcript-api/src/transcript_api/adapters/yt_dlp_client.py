from __future__ import annotations

import os
from collections.abc import Sequence

from platform_core.json_utils import JSONValue
from typing_extensions import TypedDict

from .. import _test_hooks
from .._test_hooks import YtDlpFactoryProto
from ..stt_provider import ProbeDownloadClient
from ..types import FormatTD, RequestedDownloadTD, SubtitleResultTD, YtInfoTD


class _YtDlpOpts(TypedDict, total=False):
    quiet: bool
    no_warnings: bool
    skip_download: bool
    cachedir: bool


class _YtDlpDownloadOpts(TypedDict, total=False):
    format: str
    outtmpl: str
    quiet: bool
    noprogress: bool
    no_warnings: bool
    restrictfilenames: bool
    overwrites: bool
    cachedir: bool
    cookiefile: str


class _YtDlpSubtitleOpts(TypedDict, total=False):
    writesubtitles: bool
    writeautomaticsub: bool
    subtitleslangs: list[str]
    subtitlesformat: str
    skip_download: bool
    outtmpl: str
    quiet: bool
    no_warnings: bool
    cachedir: bool
    cookiefile: str


def _coerce_format_item(f: dict[str, JSONValue]) -> FormatTD | None:
    item: FormatTD = {}
    vcodec = f.get("vcodec")
    if isinstance(vcodec, str):
        item["vcodec"] = vcodec
    acodec = f.get("acodec")
    if isinstance(acodec, str):
        item["acodec"] = acodec
    abr = f.get("abr")
    if isinstance(abr, (int, float)):
        item["abr"] = float(abr)
    filesize = f.get("filesize")
    if isinstance(filesize, int):
        item["filesize"] = filesize
    filesize_approx = f.get("filesize_approx")
    if isinstance(filesize_approx, int):
        item["filesize_approx"] = filesize_approx
    return item if item else None


def _coerce_formats(raw_formats: Sequence[dict[str, JSONValue] | int]) -> list[FormatTD]:
    formats: list[FormatTD] = []
    for f in raw_formats:
        if isinstance(f, dict):
            item = _coerce_format_item(f)
            if item:
                formats.append(item)
    return formats


def _coerce_requested_downloads(
    raw_reqs: Sequence[dict[str, JSONValue] | int],
) -> list[RequestedDownloadTD]:
    reqs: list[RequestedDownloadTD] = []
    for r in raw_reqs:
        if isinstance(r, dict):
            filepath = r.get("filepath")
            if isinstance(filepath, str):
                reqs.append({"filepath": filepath})
    return reqs


def _coerce_id(raw: dict[str, JSONValue]) -> str | None:
    """Extract and validate 'id' field."""
    vid = raw.get("id")
    return vid if isinstance(vid, str) else None


def _coerce_duration(raw: dict[str, JSONValue]) -> float | None:
    """Extract and validate 'duration' field."""
    dur = raw.get("duration")
    return float(dur) if isinstance(dur, (int, float)) else None


def _coerce_formats_from_raw(raw: dict[str, JSONValue]) -> list[FormatTD]:
    """Extract and coerce 'formats' list."""
    raw_formats = raw.get("formats")
    if not isinstance(raw_formats, list):
        return []
    # Narrow list items to dict before passing to _coerce_formats
    formats_dicts: list[dict[str, JSONValue]] = [
        item for item in raw_formats if isinstance(item, dict)
    ]
    return _coerce_formats(formats_dicts)


def _coerce_requested_downloads_from_raw(raw: dict[str, JSONValue]) -> list[RequestedDownloadTD]:
    """Extract and coerce 'requested_downloads' list."""
    raw_reqs = raw.get("requested_downloads")
    if not isinstance(raw_reqs, list):
        return []
    # Narrow list items to dict before passing
    reqs_dicts: list[dict[str, JSONValue]] = [item for item in raw_reqs if isinstance(item, dict)]
    return _coerce_requested_downloads(reqs_dicts)


def _coerce_yt_info(raw: dict[str, JSONValue]) -> YtInfoTD:
    """Coerce raw yt_dlp output (dict) into strictly typed YtInfoTD."""
    out: YtInfoTD = {}

    vid = _coerce_id(raw)
    if vid:
        out["id"] = vid

    dur = _coerce_duration(raw)
    if dur is not None:
        out["duration"] = dur

    formats = _coerce_formats_from_raw(raw)
    if formats:
        out["formats"] = formats

    reqs = _coerce_requested_downloads_from_raw(raw)
    if reqs:
        out["requested_downloads"] = reqs

    return out


def _get_yt_dlp_factory() -> YtDlpFactoryProto:
    """Get the yt-dlp factory from test hooks."""
    return _test_hooks.yt_dlp_factory


def _yt_probe(url: str) -> YtInfoTD:
    ydl_factory = _get_yt_dlp_factory()
    opts_data: dict[str, JSONValue] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "cachedir": False,
    }
    with ydl_factory(opts_data) as ydl:
        info_raw = ydl.extract_info(url, download=False)
        return _coerce_yt_info(info_raw)


def _select_lang_from_dict(
    subtitle_dict: dict[str, JSONValue],
    preferred_langs: list[str],
) -> str | None:
    """Select best language from subtitle dictionary based on preference order."""
    for lang in preferred_langs:
        if lang in subtitle_dict:
            return lang
    # Fall back to first available
    return next(iter(subtitle_dict.keys()), None)


def _find_subtitle_lang(
    info_raw: dict[str, JSONValue],
    preferred_langs: list[str],
) -> tuple[str | None, bool]:
    """Find best subtitle language and whether it's auto-generated.

    Returns (selected_lang, is_auto) tuple. Returns (None, False) if no subtitles.
    """
    subtitles = info_raw.get("subtitles")
    auto_captions = info_raw.get("automatic_captions")

    # Try manual subtitles first
    if isinstance(subtitles, dict) and subtitles:
        selected = _select_lang_from_dict(subtitles, preferred_langs)
        if selected:
            return selected, False

    # Fall back to automatic captions
    if isinstance(auto_captions, dict) and auto_captions:
        selected = _select_lang_from_dict(auto_captions, preferred_langs)
        if selected:
            return selected, True

    return None, False


def _yt_download(url: str, opts_dict: dict[str, JSONValue]) -> tuple[YtInfoTD, str]:
    ydl_factory = _get_yt_dlp_factory()
    with ydl_factory(opts_dict) as ydl:
        info_raw = ydl.extract_info(url, download=True)
        empty_dict: dict[str, JSONValue] = {}
        info_typed = _coerce_yt_info(info_raw if isinstance(info_raw, dict) else empty_dict)
        path: str | None = None

        # Try to get path from requested_downloads first
        # Note: _coerce_requested_downloads only includes entries with filepath
        if info_typed.get("requested_downloads"):
            first_req = info_typed["requested_downloads"][0]
            path = first_req["filepath"]

        # If not found, prepare filename from info (only if we have valid info with id)
        if not path and isinstance(info_raw, dict) and info_raw.get("id"):
            path = ydl.prepare_filename(info_raw)

        return info_typed, (path or "")


class YtDlpAdapter(ProbeDownloadClient):
    """Adapter over yt_dlp with strict return types without vendor typing."""

    @staticmethod
    def _is_numeric_str(s: str) -> bool:
        if not s:
            return False
        first = s[0]
        rest = s
        if first in "+-":
            rest = s[1:]
            if not rest:
                return False
        dot_seen = False
        digit_seen = True  # A numeric string must have at least one digit
        for ch in rest:
            if ch == ".":
                if dot_seen:
                    return False
                dot_seen = True
            elif ch.isdigit():
                digit_seen = True
            else:
                return False
        return digit_seen

    def probe(self, url: str) -> YtInfoTD:
        return _yt_probe(url)

    def download_audio(self, url: str, *, cookies_path: str | None) -> str:
        tmpdir = _test_hooks.mkdtemp("ytstt_", None)
        outtmpl = os.path.join(tmpdir, "audio.%(ext)s")
        opts: dict[str, JSONValue] = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "restrictfilenames": True,
            "overwrites": True,
            "cachedir": False,
        }
        if cookies_path:
            opts["cookiefile"] = cookies_path
        _info, path = _yt_download(url, opts)
        return path

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None:
        """Download subtitles for a video using yt-dlp.

        Tries manual subtitles first, then automatic captions.
        Returns None if no subtitles are available.
        """
        import glob as glob_module

        tmpdir = _test_hooks.mkdtemp("ytsubs_", None)
        ydl_factory = _get_yt_dlp_factory()

        # First, extract info to check available subtitles
        info_raw = self._probe_subtitle_info(ydl_factory, url, cookies_path)

        # Find best language
        selected_lang, is_auto = _find_subtitle_lang(info_raw, preferred_langs)
        if not selected_lang:
            return None

        # Download subtitles
        outtmpl = os.path.join(tmpdir, "subs.%(ext)s")
        self._download_subtitle_file(
            ydl_factory, url, outtmpl, selected_lang, is_auto, cookies_path
        )

        # Find the downloaded subtitle file
        pattern = os.path.join(tmpdir, "subs.*")
        matches = glob_module.glob(pattern)
        if not matches:
            return None

        return {"path": matches[0], "lang": selected_lang, "is_auto": is_auto}

    def _probe_subtitle_info(
        self,
        ydl_factory: YtDlpFactoryProto,
        url: str,
        cookies_path: str | None,
    ) -> dict[str, JSONValue]:
        """Extract video info to check available subtitles."""
        probe_opts: dict[str, JSONValue] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "cachedir": False,
        }
        if cookies_path:
            probe_opts["cookiefile"] = cookies_path

        with ydl_factory(probe_opts) as ydl:
            return ydl.extract_info(url, download=False)

    def _download_subtitle_file(
        self,
        ydl_factory: YtDlpFactoryProto,
        url: str,
        outtmpl: str,
        selected_lang: str,
        is_auto: bool,
        cookies_path: str | None,
    ) -> None:
        """Download subtitle file with specified options."""
        download_opts: dict[str, JSONValue] = {
            "skip_download": True,
            "writesubtitles": not is_auto,
            "writeautomaticsub": is_auto,
            "subtitleslangs": [selected_lang],
            "subtitlesformat": "vtt/srt/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "cachedir": False,
        }
        if cookies_path:
            download_opts["cookiefile"] = cookies_path

        with ydl_factory(download_opts) as ydl:
            ydl.extract_info(url, download=True)


__all__ = ["YtDlpAdapter"]
