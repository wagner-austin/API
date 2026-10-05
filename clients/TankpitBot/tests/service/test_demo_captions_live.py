"""Captions from a bot's events artifact to the public ``/demo/fleet`` row.

The artifact is the in-memory stand-in the telemetry tests use, read
through the production incremental reader; the route is the real fleet
app. What is pinned: the fleet reads captions on the same bounded
cadence as its other summaries, a run whose reasons the fleet cannot
name is surfaced rather than half-captioned, and a stranger's row
carries the bot's words.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from datetime import datetime

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.bot.ai.scoring_types import ReasonKind
from tankpit_bot.service.demo_caption_words import REASON_WORDS
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_routes import make_fleet_app
from tankpit_bot.service.fleet_telemetry import TELEMETRY_CACHE_TTL_MS, FleetTelemetry
from tests.service._artifact_fixtures import FakeArtifact
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_account_pool,
)


def _decision(second: int, reason: str) -> str:
    """One executor decision line carrying a typed reason.

    Args:
        second: Seconds past 20:00:00 on the test day.
        reason: The ``behavior_reason_kind`` word.

    Returns:
        One JSONL event line.
    """
    return (
        f'{{"timestamp":"2026-10-05T20:00:{second:02d}","level":"INFO","logger":"l",'
        f'"mode":"bot","channel":"AI","message":"decision {second}",'
        f'"behavior_reason_kind":"{reason}"}}'
    )


def _at(second: int) -> int:
    """Epoch milliseconds of a test-day second.

    Args:
        second: Seconds past 20:00:00 on the test day.

    Returns:
        Epoch milliseconds, read in the local zone as the fold reads them.
    """
    return int(datetime(2026, 10, 5, 20, 0, second).timestamp() * 1000)


class _Clock:
    """A settable ``get_current_time_ms``."""

    def __init__(self, now_ms: int) -> None:
        """Start at ``now_ms``.

        Args:
            now_ms: The first reading.
        """
        self.now_ms = now_ms

    def __call__(self) -> int:
        """Read the clock.

        Returns:
            The current setting.
        """
        return self.now_ms


@pytest.fixture()
def clock() -> _Clock:
    """Install a clock just after the test day's first events.

    Returns:
        The clock; the autouse hook-restore fixture puts the real one back.
    """
    fixed = _Clock(_at(5))
    top_hooks.get_current_time_ms = fixed
    return fixed


def test_an_instance_with_no_events_has_no_captions(artifact: FakeArtifact, clock: _Clock) -> None:
    """A bot still logging in has said nothing yet."""
    _ = (artifact, clock)
    assert FleetTelemetry().captions("demo-1") == []


def test_captions_are_read_on_the_telemetry_cadence(artifact: FakeArtifact, clock: _Clock) -> None:
    """Within the cache window the artifact is not re-read; after it, it is."""
    artifact.start_run([_decision(1, "find_enemies")])
    telemetry = FleetTelemetry()

    first = telemetry.captions("demo-1")
    artifact.append([_decision(3, "teleport_target")])
    within = telemetry.captions("demo-1")
    reads_within = len(artifact.read_offsets)
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    after = telemetry.captions("demo-1")

    assert [caption["doing"] for caption in first] == [
        REASON_WORDS[ReasonKind.FIND_ENEMIES]["doing"]
    ]
    assert within == first
    assert reads_within == 1
    assert [caption["at_ms"] for caption in after] == [_at(1), _at(3)]


def test_forgetting_an_instance_rereads_it_at_once(artifact: FakeArtifact, clock: _Clock) -> None:
    """A forgotten instance starts over, cadence stamp included."""
    _ = clock
    artifact.start_run([_decision(1, "find_enemies")])
    telemetry = FleetTelemetry()
    telemetry.captions("demo-1")

    telemetry.forget("demo-1")
    again = telemetry.captions("demo-1")

    assert len(artifact.read_offsets) == 2
    assert len(again) == 1


def test_a_reason_the_fleet_cannot_name_spoils_the_run(
    artifact: FakeArtifact, clock: _Clock
) -> None:
    """No captions, and the operator's summaries refuse the run too."""
    artifact.start_run([_decision(1, "invented_reason")])
    telemetry = FleetTelemetry()

    assert telemetry.captions("demo-1") == []
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1
    assert telemetry.activity("demo-1") == {"available": False}


def test_manager_captions_require_a_registered_instance() -> None:
    """The manager gate: an unknown name is refused before any read."""
    with pytest.raises(FleetError, match="unknown instance"):
        FleetManager().captions("ghost")


@pytest.fixture()
async def demo_client(
    spawner: _FakeSpawner,
) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """Serve the fleet app on a one-account machine.

    Yields:
        A client bound to the real routes.
    """
    _ = spawner
    client: TestClient[web.Request, web.Application] = TestClient(
        TestServer(make_fleet_app(FleetManager()))
    )
    await client.start_server()
    yield client
    await client.close()


@pytest.fixture()
def one_account() -> Generator[None, None, None]:
    """Configure a one-account machine for one test.

    Yields:
        Nothing; the fixture exists for its effect.
    """
    originals = _with_account_pool("alpha")
    yield
    _restore_account_hooks(originals)


@pytest.mark.asyncio
async def test_the_public_row_carries_the_bots_captions(
    one_account: None,
    artifact: FakeArtifact,
    clock: _Clock,
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """What a stranger reads is the bot's reason, in words, with its moment."""
    _ = one_account
    spawned = await demo_client.post("/demo/spawn")
    assert spawned.status == 201
    artifact.start_run([_decision(2, "shoot_target")])
    # The spawn's own row read the artifact while it was absent; the
    # next read is due once the telemetry window has passed.
    clock.now_ms += TELEMETRY_CACHE_TTL_MS + 1

    response = await demo_client.get("/demo/fleet")
    assert response.status == 200
    body = narrow_json_to_dict(load_json_str(await response.text()))
    row = narrow_json_to_dict(require_list(body, "bots")[0])
    captions = [narrow_json_to_dict(item) for item in require_list(row, "captions")]

    words = REASON_WORDS[ReasonKind.SHOOT_TARGET]
    assert len(captions) == 1
    assert require_int(captions[0], "at_ms") == _at(2)
    assert require_str(captions[0], "doing") == words["doing"]
    assert require_str(captions[0], "why") == words["why"]
    assert require_int(captions[0], "fuel") == -1
