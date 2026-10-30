"""Real children consume bounded input, fail, or stop reading without hanging."""

import subprocess
import sys

import pytest

from platform_core.process_input import process_input


@pytest.mark.parametrize("payload", [None, b"\x00private\xff"])
def test_input_is_exact_or_closed(payload: bytes | None) -> None:
    with process_input(payload) as source:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); sys.exit(7)",
            ],
            stdin=source,
            capture_output=True,
            timeout=5,
            check=False,
        )
    assert result.returncode == 7
    assert result.stdout == (b"" if payload is None else payload)
    assert result.stderr == b""
    if not isinstance(source, int):
        assert source.closed


def test_nonreading_child_times_out_and_input_is_closed() -> None:
    with pytest.raises(subprocess.TimeoutExpired), process_input(b"x" * 1000000) as source:
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=source,
            capture_output=True,
            timeout=0.2,
            check=False,
        )
    assert not isinstance(source, int)
    assert source.closed
