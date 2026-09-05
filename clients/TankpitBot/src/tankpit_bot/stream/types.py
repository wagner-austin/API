"""The display-capture configuration shape and its codecs.

One immutable description of a capture session: which X display to
create, its geometry, and how the encoder should cut the video into
HLS segments. Built once by the config resolver
(:func:`tankpit_bot.bot.config.resolve_stream_config`) and handed down
whole, so the Xvfb command, the ffmpeg command and the HTTP surface can
never disagree about a dimension.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int, require_str
from typing_extensions import TypedDict


class StreamConfigDict(TypedDict):
    """One capture session's fixed parameters.

    Attributes:
        display: X display number Xvfb creates and ffmpeg records.
            Unique per live bot — the fleet derives it from the
            child's own service port, which the manager already
            allocates uniquely.
        width: Screen width in pixels. Even, because yuv420p encodes
            chroma at half resolution and ffmpeg refuses odd sizes.
        height: Screen height in pixels. Even, same law.
        fps: Frames per second ffmpeg samples the display at.
        bitrate_kbps: Target video bitrate in kilobits per second.
        segment_seconds: Length of one HLS segment. Also the keyframe
            cadence — every segment opens on a keyframe so a viewer
            can join at any segment boundary.
        hls_dir: Absolute directory the playlist and segments land in.
    """

    display: int
    width: int
    height: int
    fps: int
    bitrate_kbps: int
    segment_seconds: int
    hls_dir: str


def encode_stream_config(config: StreamConfigDict) -> JSONObject:
    """Encode :class:`StreamConfigDict` to a JSON-serializable dict.

    Args:
        config: Capture configuration to encode.

    Returns:
        JSON object carrying every field.
    """
    return {
        "display": config["display"],
        "width": config["width"],
        "height": config["height"],
        "fps": config["fps"],
        "bitrate_kbps": config["bitrate_kbps"],
        "segment_seconds": config["segment_seconds"],
        "hls_dir": config["hls_dir"],
    }


def decode_stream_config(data: JSONObject) -> StreamConfigDict:
    """Validate and decode a :class:`StreamConfigDict` from JSON.

    Args:
        data: JSON object to validate.

    Returns:
        Validated :class:`StreamConfigDict`.

    Raises:
        JSONTypeError: A field is missing or has the wrong type.
        ValueError: A field is outside its documented domain.
    """
    config = StreamConfigDict(
        display=require_int(data, "display"),
        width=require_int(data, "width"),
        height=require_int(data, "height"),
        fps=require_int(data, "fps"),
        bitrate_kbps=require_int(data, "bitrate_kbps"),
        segment_seconds=require_int(data, "segment_seconds"),
        hls_dir=require_str(data, "hls_dir"),
    )
    if config["display"] < 0:
        raise ValueError(f"display must be non-negative, got {config['display']}")
    if config["width"] <= 0 or config["width"] % 2:
        raise ValueError(f"width must be positive and even, got {config['width']}")
    if config["height"] <= 0 or config["height"] % 2:
        raise ValueError(f"height must be positive and even, got {config['height']}")
    if config["fps"] <= 0:
        raise ValueError(f"fps must be positive, got {config['fps']}")
    if config["bitrate_kbps"] <= 0:
        raise ValueError(f"bitrate_kbps must be positive, got {config['bitrate_kbps']}")
    if config["segment_seconds"] <= 0:
        raise ValueError(f"segment_seconds must be positive, got {config['segment_seconds']}")
    if not config["hls_dir"]:
        raise ValueError("hls_dir must not be empty")
    return config


__all__ = [
    "StreamConfigDict",
    "decode_stream_config",
    "encode_stream_config",
]
