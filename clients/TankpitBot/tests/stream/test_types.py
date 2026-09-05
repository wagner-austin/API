"""Codecs and domain validation for :class:`StreamConfigDict`."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.stream.types import (
    StreamConfigDict,
    decode_stream_config,
    encode_stream_config,
)


def _valid_payload() -> JSONObject:
    """One in-domain JSON payload the failure tests mutate.

    Returns:
        A payload :func:`decode_stream_config` accepts unchanged.
    """
    return {
        "display": 9,
        "width": 704,
        "height": 544,
        "scale": 2,
        "fps": 30,
        "bitrate_kbps": 1000,
        "segment_seconds": 2,
        "hls_dir": "runs/bot/demo-1/hls",
    }


class TestRoundTrip:
    """Encode and decode agree on every field."""

    def test_decode_of_encode_is_identity(self) -> None:
        """A config survives the round trip byte-for-byte."""
        config = StreamConfigDict(
            display=27301,
            width=704,
            height=544,
            scale=2,
            fps=24,
            bitrate_kbps=800,
            segment_seconds=2,
            hls_dir="runs/bot/demo-2/hls",
        )
        assert decode_stream_config(encode_stream_config(config)) == config

    def test_decode_validates_a_raw_payload(self) -> None:
        """A plain JSON object in domain decodes to the typed shape."""
        config = decode_stream_config(_valid_payload())
        assert config["display"] == 9
        assert config["hls_dir"] == "runs/bot/demo-1/hls"


class TestDomainRefusals:
    """Every out-of-domain value is refused with its own message."""

    def test_missing_field_is_a_type_error(self) -> None:
        """A payload without a required field fails the require layer."""
        payload = _valid_payload()
        del payload["display"]
        with pytest.raises(JSONTypeError):
            decode_stream_config(payload)

    def test_negative_display_is_refused(self) -> None:
        """Display numbers are non-negative."""
        payload = _valid_payload()
        payload["display"] = -1
        with pytest.raises(ValueError, match="display must be non-negative"):
            decode_stream_config(payload)

    @pytest.mark.parametrize("width", [0, -2, 701])
    def test_bad_width_is_refused(self, width: int) -> None:
        """Width must be positive and even (yuv420p halves chroma)."""
        payload = _valid_payload()
        payload["width"] = width
        with pytest.raises(ValueError, match="width must be positive and even"):
            decode_stream_config(payload)

    @pytest.mark.parametrize("height", [0, -2, 543])
    def test_bad_height_is_refused(self, height: int) -> None:
        """Height must be positive and even."""
        payload = _valid_payload()
        payload["height"] = height
        with pytest.raises(ValueError, match="height must be positive and even"):
            decode_stream_config(payload)

    def test_non_positive_scale_is_refused(self) -> None:
        """A zero device scale factor is not a smaller picture, it is none."""
        payload = _valid_payload()
        payload["scale"] = 0
        with pytest.raises(ValueError, match="scale must be positive"):
            decode_stream_config(payload)

    def test_non_positive_fps_is_refused(self) -> None:
        """Zero frames per second is not a slower stream, it is none."""
        payload = _valid_payload()
        payload["fps"] = 0
        with pytest.raises(ValueError, match="fps must be positive"):
            decode_stream_config(payload)

    def test_non_positive_bitrate_is_refused(self) -> None:
        """Zero bitrate is refused."""
        payload = _valid_payload()
        payload["bitrate_kbps"] = 0
        with pytest.raises(ValueError, match="bitrate_kbps must be positive"):
            decode_stream_config(payload)

    def test_non_positive_segment_length_is_refused(self) -> None:
        """Zero-length segments are refused."""
        payload = _valid_payload()
        payload["segment_seconds"] = 0
        with pytest.raises(ValueError, match="segment_seconds must be positive"):
            decode_stream_config(payload)

    def test_empty_hls_dir_is_refused(self) -> None:
        """A config with nowhere to write is refused."""
        payload = _valid_payload()
        payload["hls_dir"] = ""
        with pytest.raises(ValueError, match="hls_dir must not be empty"):
            decode_stream_config(payload)
