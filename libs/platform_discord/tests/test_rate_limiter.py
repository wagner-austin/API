from __future__ import annotations

import time

from platform_discord.rate_limiter import RateLimiter


def test_rate_limiter_allows_then_blocks_and_reports_wait() -> None:
    rl = RateLimiter(2, window_seconds=1)
    ok1, wait1 = rl.allow(1, "cmd")
    ok2, wait2 = rl.allow(1, "cmd")
    ok3, wait3 = rl.allow(1, "cmd")
    assert ok1 and wait1 == 0.0
    assert ok2 and wait2 == 0.0
    assert not ok3 and wait3 >= 0.1


def test_rate_limiter_window_expires() -> None:
    rl = RateLimiter(1, window_seconds=1)
    ok1, _ = rl.allow(2, "x")
    time.sleep(1.05)
    ok2, w2 = rl.allow(2, "x")
    assert ok1 and ok2 and w2 == 0.0
