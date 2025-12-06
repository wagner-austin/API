from __future__ import annotations

import re
from typing import Final, TypedDict

from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue
from platform_core.validators import load_json_dict

from .types import ECCLevel, QROptions

_HEX_COLOR_RE: Final[re.Pattern[str]] = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")
_DOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_DOMAIN_RE: Final[re.Pattern[str]] = re.compile(rf"^(?:{_DOMAIN_LABEL}\.)+[A-Za-z]{{2,63}}$")
_IPV4_RE: Final[re.Pattern[str]] = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


class Defaults(TypedDict):
    ecc: ECCLevel
    box_size: int
    border: int
    fill_color: str
    back_color: str


def _split_scheme(url: str) -> tuple[str, str]:
    parts = url.split("://", 1)
    if len(parts) != 2:
        raise AppError(ErrorCode.INVALID_INPUT, "Invalid URL format")
    scheme_raw, rest = parts[0], parts[1]
    scheme = scheme_raw.lower()
    if scheme not in {"http", "https"}:
        raise AppError(ErrorCode.INVALID_INPUT, "URL scheme must be http or https")
    return scheme, rest


def _split_netloc(rest: str) -> tuple[str, str]:
    end = len(rest)
    slash = rest.find("/")
    ques = rest.find("?")
    if slash != -1:
        end = min(end, slash)
    if ques != -1:
        end = min(end, ques)
    return rest[:end], rest[end:]


def _host_from_netloc(netloc: str) -> str:
    host = netloc
    if host.startswith("[") and "]" in host:
        return host.split("]", 1)[0] + "]"
    if ":" in host:
        return host.split(":", 1)[0]
    return host


def _valid_host(host: str) -> bool:
    if not host:
        return False
    h = host.lower().strip(".")
    return (
        h == "localhost"
        or _DOMAIN_RE.match(h) is not None
        or _IPV4_RE.match(h) is not None
        or (h.startswith("[") and h.endswith("]"))
    )


def _normalize_url(raw: str) -> str:
    s = raw.strip()
    if not s:
        raise AppError(ErrorCode.INVALID_INPUT, "Please provide a URL")
    if len(s) > 2000:
        raise AppError(ErrorCode.INVALID_INPUT, "URL is too long (max 2000 characters)")
    candidate = s if "://" in s else f"https://{s}"
    scheme, rest = _split_scheme(candidate)
    netloc, path_query = _split_netloc(rest)
    host = _host_from_netloc(netloc)
    if not _valid_host(host):
        raise AppError(ErrorCode.INVALID_INPUT, "Please check the URL and try again.")
    return f"{scheme}://{netloc}{path_query.split('#', 1)[0]}"


def _validate_hex_color(color: str | None, default: str) -> str:
    if color is None or color.strip() == "":
        return default
    c = color.strip()
    if _HEX_COLOR_RE.match(c) is None:
        raise AppError(
            ErrorCode.INVALID_INPUT,
            "Invalid color format. Use hex codes (e.g., #FF0000 or #F00)",
        )
    return c


def _validate_ecc(level: str | None, default: ECCLevel) -> ECCLevel:
    if level is None or level.strip() == "":
        return default
    up = level.strip().upper()
    # Segno supports the full set of QR ECC levels; enforce L/M/Q/H selection.
    allowed: tuple[ECCLevel, ...] = ("L", "M", "Q", "H")
    if up not in allowed:
        raise AppError(ErrorCode.INVALID_INPUT, "Invalid error correction. Choose L, M, Q, H")
    # Return as ECCLevel without casts by mapping through a typed dict
    mapping: dict[str, ECCLevel] = {"L": "L", "M": "M", "Q": "Q", "H": "H"}
    return mapping[up]


def _validate_box_size(value: str | int | float | None, default: int) -> int:
    if value is None:
        return default
    try_val = int(str(value))
    if not (5 <= try_val <= 20):
        raise AppError(ErrorCode.INVALID_INPUT, "box_size must be between 5 and 20")
    return try_val


def _validate_border(value: str | int | float | None, default: int) -> int:
    if value is None:
        return default
    try_val = int(str(value))
    if not (1 <= try_val <= 10):
        raise AppError(ErrorCode.INVALID_INPUT, "border must be between 1 and 10")
    return try_val


def _decode_qr_payload(obj: JSONValue, defaults: Defaults) -> QROptions:
    d = load_json_dict(obj)
    url_o = d.get("url")
    if not isinstance(url_o, str):
        raise AppError(ErrorCode.INVALID_INPUT, "Field 'url' is required")

    ecc_o = d.get("ecc")
    box_o = d.get("box_size")
    border_o = d.get("border")
    fill_o = d.get("fill_color")
    back_o = d.get("back_color")

    url = _normalize_url(url_o)
    ecc = _validate_ecc(ecc_o if isinstance(ecc_o, str) else None, defaults["ecc"])
    box = _validate_box_size(
        box_o if isinstance(box_o, (str, int, float, type(None))) else None, defaults["box_size"]
    )
    border = _validate_border(
        border_o if isinstance(border_o, (str, int, float, type(None))) else None,
        defaults["border"],
    )
    fill = _validate_hex_color(fill_o if isinstance(fill_o, str) else None, defaults["fill_color"])
    back = _validate_hex_color(back_o if isinstance(back_o, str) else None, defaults["back_color"])

    return {
        "url": url,
        "ecc": ecc,
        "box_size": box,
        "border": border,
        "fill_color": fill,
        "back_color": back,
    }


__all__ = ["Defaults"]
