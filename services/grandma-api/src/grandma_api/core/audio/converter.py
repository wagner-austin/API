"""Audio format conversion using ffmpeg.

Converts audio files to 16kHz mono WAV format required by language detection.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Final, Protocol

from platform_core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_RATE = 16000

#: Wall-clock bound on one ffmpeg conversion. Ten minutes: this service
#: converts a single uploaded clip to 16 kHz mono, which takes seconds, so
#: the bound clears any honest conversion by orders of magnitude and is
#: still FINITE. It runs inside a REQUEST, which is the reason it cannot be
#: unbounded: an ffmpeg that never returns holds the worker handling that
#: request forever, and the caller sees no answer and no error (board task
#: 0d891468).
FFMPEG_CONVERT_WALL_SECONDS: Final[int] = 600


class AudioConverterProtocol(Protocol):
    """Protocol for audio conversion function."""

    def __call__(self, audio_bytes: bytes, source_filename: str) -> bytes:
        """Convert audio bytes to WAV format.

        Args:
            audio_bytes: Raw audio bytes in any supported format.
            source_filename: Original filename for format detection.

        Returns:
            WAV audio bytes at 16kHz mono.

        Raises:
            subprocess.CalledProcessError: If ffmpeg conversion fails.
            FileNotFoundError: If ffmpeg is not installed.
        """
        ...


def _run_ffmpeg(input_path: str, output_path: str) -> subprocess.CompletedProcess[bytes]:
    """Run ffmpeg to convert audio to 16kHz mono WAV.

    Args:
        input_path: Path to input audio file.
        output_path: Path to output WAV file.

    Returns:
        CompletedProcess with stdout/stderr captured.

    Raises:
        subprocess.CalledProcessError: If ffmpeg returns non-zero exit code.
        FileNotFoundError: If ffmpeg is not installed.
    """
    return subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            "-ac",
            "1",
            "-f",
            "wav",
            output_path,
        ],
        capture_output=True,
        check=True,
        timeout=FFMPEG_CONVERT_WALL_SECONDS,
    )


def _default_convert_to_wav(audio_bytes: bytes, source_filename: str) -> bytes:
    """Convert audio bytes to WAV format using ffmpeg.

    Args:
        audio_bytes: Raw audio bytes (webm, mp3, etc.).
        source_filename: Original filename for format detection.

    Returns:
        WAV audio bytes at 16kHz mono.

    Raises:
        subprocess.CalledProcessError: If ffmpeg conversion fails.
        FileNotFoundError: If ffmpeg is not installed.
    """
    suffix = Path(source_filename).suffix or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as in_file:
        in_file.write(audio_bytes)
        in_path = in_file.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
        out_path = out_file.name

    in_path_obj = Path(in_path)
    out_path_obj = Path(out_path)

    result = _run_ffmpeg(in_path, out_path)
    logger.debug("ffmpeg conversion complete", extra={"stderr": result.stderr.decode()})

    wav_bytes = out_path_obj.read_bytes()

    in_path_obj.unlink(missing_ok=True)
    out_path_obj.unlink(missing_ok=True)

    return wav_bytes


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "AudioConverterProtocol",
    "_default_convert_to_wav",
]
