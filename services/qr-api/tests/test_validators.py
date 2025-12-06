from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode

from qr_api.validators import (
    Defaults,
    _decode_qr_payload,
    _host_from_netloc,
    _normalize_url,
    _split_netloc,
    _split_scheme,
    _valid_host,
)


def test_split_scheme_valid_and_invalid() -> None:
    scheme, rest = _split_scheme("https://example.com/path?x=1")
    assert scheme == "https"
    assert rest.startswith("example.com")
    with pytest.raises(AppError) as ex_missing_scheme:
        _split_scheme("no-scheme-here")
    assert ex_missing_scheme.value.code is ErrorCode.INVALID_INPUT
    assert ex_missing_scheme.value.http_status == 400
    with pytest.raises(AppError):
        _split_scheme("://bad")
    with pytest.raises(AppError):
        _split_scheme("ftp://example.com")


def test_split_netloc_variants() -> None:
    netloc, tail = _split_netloc("example.com/path?x=1")
    assert netloc == "example.com"
    assert tail.startswith("/path")
    netloc2, tail2 = _split_netloc("example.com?x=1")
    assert netloc2 == "example.com"
    assert tail2.startswith("?")


def test_host_from_netloc_and_valid_host() -> None:
    assert _host_from_netloc("example.com:8080") == "example.com"
    assert _host_from_netloc("[::1]:443") == "[::1]"
    assert _valid_host("localhost")
    assert _valid_host("example.com")
    assert _valid_host("127.0.0.1")
    assert _valid_host("[::1]")
    assert not _valid_host("")
    assert not _valid_host("bad host with spaces")


def test_normalize_url_variants() -> None:
    assert _normalize_url("example.com").startswith("https://example.com")
    assert _normalize_url("http://example.com").startswith("http://example.com")
    with pytest.raises(AppError):
        _normalize_url("")
    with pytest.raises(AppError):
        _normalize_url("x" * 2001)
    with pytest.raises(AppError):
        _normalize_url("bad host with spaces")


def test_parse_qr_payload_defaults_and_errors() -> None:
    d: Defaults = {
        "ecc": "M",
        "box_size": 10,
        "border": 2,
        "fill_color": "#000000",
        "back_color": "#FFFFFF",
    }
    out = _decode_qr_payload({"url": "example.com"}, d)
    assert out["url"].startswith("https://")
    assert out["ecc"] == d["ecc"]
    assert out["box_size"] == d["box_size"]
    assert out["border"] == d["border"]
    assert out["fill_color"] == d["fill_color"]
    assert out["back_color"] == d["back_color"]

    with pytest.raises(AppError):
        _decode_qr_payload({}, d)
    with pytest.raises(AppError):
        _decode_qr_payload({"url": "example.com", "ecc": "Z"}, d)
    with pytest.raises(AppError):
        _decode_qr_payload({"url": "example.com", "box_size": 1}, d)
    with pytest.raises(AppError):
        _decode_qr_payload({"url": "example.com", "border": 0}, d)
    with pytest.raises(AppError):
        _decode_qr_payload({"url": "example.com", "fill_color": "red"}, d)
    with pytest.raises(AppError):
        _decode_qr_payload("not a dict", d)


def test_parse_qr_payload_with_explicit_options() -> None:
    d: Defaults = {
        "ecc": "M",
        "box_size": 10,
        "border": 2,
        "fill_color": "#000000",
        "back_color": "#FFFFFF",
    }
    out = _decode_qr_payload(
        {
            "url": "https://example.com/path",
            "ecc": "q",
            "box_size": 12,
            "border": 3,
            "fill_color": "#0f0",
            "back_color": "#fff",
        },
        d,
    )
    assert out["ecc"] == "Q"
    assert out["box_size"] == 12
    assert out["border"] == 3
    assert out["fill_color"] == "#0f0"
    assert out["back_color"] == "#fff"
