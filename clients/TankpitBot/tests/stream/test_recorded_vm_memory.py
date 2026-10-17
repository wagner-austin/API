"""The four-bot memory watch of 2026-10-17, read from its raw samples.

``docs/sources/vm-memory-2026-10-17.txt`` is the unedited output of the
probe that sampled sedona's docker-desktop VM every 10 s while release
v0.1.0-3888f06b2 ran four bots and the four-up TV stream (board task
73d4789e). Each line is ``HH:MM:SS avail_kb=N swapfree_kb=N load=F
new_oom=N``, ``new_oom`` counting oom-kill lines dmesg gained since the
probe began. This test computes the watch's bounds from those samples,
so the claim the task closed on is the file's arithmetic, not prose.
"""

from __future__ import annotations

import re
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[2] / "docs" / "sources" / "vm-memory-2026-10-17.txt"
SAMPLE_LINE = re.compile(
    r"(?P<time>\d{2}:\d{2}:\d{2}) avail_kb=(?P<avail>\d+) swapfree_kb=(?P<swap>\d+)"
    r" load=(?P<load>[\d.]+) new_oom=(?P<oom>\d+)"
)
WATCH_START = "09:40:24"
"""The first sample with all four tiles live and the TV cast."""
WATCH_END = "09:51:07"
"""The last sample before the operator had every bot drained."""
ONE_GIGABYTE_KB = 1024 * 1024


def _watch_samples() -> list[tuple[int, int, int]]:
    """Read the samples inside the watch window.

    Returns:
        ``(avail_kb, swapfree_kb, new_oom)`` per sample, in order.

    Raises:
        AssertionError: If any line of the file is not a sample.
    """
    samples: list[tuple[int, int, int]] = []
    for line in SAMPLES.read_text(encoding="utf-8").splitlines():
        match = SAMPLE_LINE.fullmatch(line)
        if match is None:
            raise AssertionError(f"not a sample line: {line!r}")
        time: str = match.group("time")
        if WATCH_START <= time <= WATCH_END:
            samples.append(
                (int(match.group("avail")), int(match.group("swap")), int(match.group("oom")))
            )
    return samples


def test_the_watch_covers_ten_minutes_of_samples() -> None:
    """Ten minutes at one sample per 10 s is at least 60 samples."""
    assert len(_watch_samples()) == 65


def test_the_vm_kept_over_a_gigabyte_available_and_in_swap() -> None:
    """The bounds the task's fourth criterion sets, computed from the samples."""
    samples = _watch_samples()
    lowest_available = min(avail for avail, _, _ in samples)
    lowest_swap_free = min(swap for _, swap, _ in samples)

    assert lowest_available == 1496620
    assert lowest_available > ONE_GIGABYTE_KB
    assert lowest_swap_free == 2090572
    assert lowest_swap_free > ONE_GIGABYTE_KB


def test_no_oom_kill_was_logged_during_the_watch() -> None:
    """dmesg gained no oom-kill line from the probe's start to the watch's end."""
    assert max(oom for _, _, oom in _watch_samples()) == 0
