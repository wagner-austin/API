"""Tests for the public demo surface.

Two things are being pinned here, and only one of them is "the feature
works". The other is that the surface stays NARROW, because it is the
one part of this service a stranger can reach: a demo row must not
carry an account username, a pid, or a port; a demo spawn must not be
steerable by its request body; and a demo read must not resolve to an
instance the operator spawned. Each of those is asserted against the
real routes rather than by reading the source, so a future field added
to the operator's report row cannot leak through by inheritance.

The account pool is stated per test rather than taken from the
developer's ``accounts.json``, because the demo's capacity is derived
from the pool size — a suite that read the real file would assert a
different ceiling on every machine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator

import pytest
from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionError
from aiohttp.test_utils import TestClient, TestServer
from platform_core.json_utils import (
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_bool,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.demo import (
    DEMO_MAX_BOTS,
    DEMO_SESSION_SECONDS,
    demo_capacity,
    demo_fleet,
    demo_slot_or_refuse,
    demo_slots,
    demo_spawn,
)
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_routes import make_fleet_app
from tankpit_bot.service.video_relay import CHILD_WARMUP_RETRY_SECONDS
from tests.service._fleet_fixtures import (
    _FakeSpawner,
    _restore_account_hooks,
    _with_account_pool,
)

#: One operator-surface spawn, named the way the operator surface names
#: things: after its account. It is the exact name a stranger would
#: guess if slot names were not the only ones the demo will resolve.
_OPERATOR_SPAWN: dict[str, str | int] = {"instance": "artax", "account": "alpha"}


@pytest.fixture()
def two_accounts() -> Generator[None, None, None]:
    """Configure a two-account machine for the duration of one test.

    Yields:
        Nothing — the fixture exists for its effect on the account
        config seams.
    """
    originals = _with_account_pool("alpha", "bravo")
    yield
    _restore_account_hooks(originals)


@pytest.fixture()
async def demo_client(
    spawner: _FakeSpawner,
    two_accounts: None,
) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """Serve the fleet app on a two-account machine.

    Yields:
        A client bound to the real routes, demo surface included.
    """
    _ = spawner
    _ = two_accounts
    manager = FleetManager()
    test_client: TestClient[web.Request, web.Application] = TestClient(
        TestServer(make_fleet_app(manager))
    )
    await test_client.start_server()
    yield test_client
    await test_client.close()


async def _fleet_body(
    client: TestClient[web.Request, web.Application],
) -> dict[str, JSONValue]:
    """Read ``GET /demo/fleet`` as a validated object.

    Args:
        client: The demo client.

    Returns:
        The decoded payload.
    """
    response = await client.get("/demo/fleet")
    assert response.status == 200
    return narrow_json_to_dict(load_json_str(await response.text()))


@pytest.mark.asyncio
async def test_an_idle_demo_reports_its_capacity_and_no_bots(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """Before anyone presses the button the demo is empty, not absent."""
    body = await _fleet_body(demo_client)

    assert require_int(body, "running") == 0
    assert require_int(body, "capacity") == 2
    assert require_bool(body, "draining") is False
    assert require_list(body, "bots") == []


@pytest.mark.asyncio
async def test_spawn_starts_one_bounded_practice_bot_in_the_first_slot(
    demo_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """The button starts a Practice bot on the demo's own clock.

    The environment is the contract with the child, so it is what gets
    asserted: the room it joins and the seconds it may play are stated
    by the demo rather than inherited from whatever the operator's
    environment happens to hold.
    """
    response = await demo_client.post("/demo/spawn")

    assert response.status == 201
    body = narrow_json_to_dict(load_json_str(await response.text()))
    assert require_str(body, "slot") == "demo-1"
    assert require_bool(body, "alive") is True

    env = spawner.envs[0]
    assert env["TANKPIT_ROOM"] == "Practice"
    assert env["TANKPIT_BOT_SESSION_SECONDS"] == str(DEMO_SESSION_SECONDS)
    assert env["TANKPIT_BOT_INSTANCE"] == "demo-1"


@pytest.mark.asyncio
async def test_a_demo_row_carries_nothing_but_slot_liveness_and_uptime(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """The public projection is a whitelist, asserted as its whole shape.

    Instance names on the operator surface are derived from account
    usernames, so a row that leaked one would publish the login. The
    key set is asserted rather than a handful of absences, so a field
    added to the report row later cannot arrive here unnoticed.
    """
    spawned = await demo_client.post("/demo/spawn")
    assert spawned.status == 201
    row = narrow_json_to_dict(load_json_str(await spawned.text()))

    assert sorted(row) == ["alive", "slot", "uptime_seconds"]

    body = await _fleet_body(demo_client)
    listed = narrow_json_to_dict(require_list(body, "bots")[0])
    assert sorted(listed) == ["alive", "slot", "uptime_seconds"]
    assert require_str(listed, "slot") == "demo-1"


@pytest.mark.asyncio
async def test_spawn_ignores_whatever_the_caller_puts_in_the_body(
    demo_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """A steering attempt is not rejected — there is nothing to steer.

    The demo reads no body at all, so a caller naming the World room,
    an unbounded session and someone else's instance gets exactly the
    bot the button gives everybody else.
    """
    steering: dict[str, str | int] = {
        "instance": "artax",
        "room": "World",
        "seconds": 0,
        "account": "bravo",
    }
    response = await demo_client.post("/demo/spawn", json=steering)

    assert response.status == 201
    body = narrow_json_to_dict(load_json_str(await response.text()))
    assert require_str(body, "slot") == "demo-1"
    env = spawner.envs[0]
    assert env["TANKPIT_ROOM"] == "Practice"
    assert env["TANKPIT_BOT_SESSION_SECONDS"] == str(DEMO_SESSION_SECONDS)
    assert env["TANKPIT_BOT_INSTANCE"] == "demo-1"


@pytest.mark.asyncio
async def test_each_spawn_takes_the_next_slot_and_a_free_account(
    demo_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """Two presses give two bots on two accounts, not two logins on one.

    The demo PICKS an untaken account where the operator surface would
    refuse a taken one; that is the whole reason both read the same
    liveness map.
    """
    first = await demo_client.post("/demo/spawn")
    second = await demo_client.post("/demo/spawn")

    assert (first.status, second.status) == (201, 201)
    assert require_str(narrow_json_to_dict(load_json_str(await second.text())), "slot") == "demo-2"
    assert [env["TANKPIT_ACCOUNT"] for env in spawner.envs] == ["alpha", "bravo"]


@pytest.mark.asyncio
async def test_spawning_past_the_account_pool_is_refused(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """A third press on a two-account machine is refused, not queued."""
    assert (await demo_client.post("/demo/spawn")).status == 201
    assert (await demo_client.post("/demo/spawn")).status == 201

    third = await demo_client.post("/demo/spawn")

    assert third.status == 409
    assert "every configured account" in await third.text()


@pytest.mark.asyncio
async def test_a_finished_bot_returns_its_slot_and_account(
    demo_client: TestClient[web.Request, web.Application],
    spawner: _FakeSpawner,
) -> None:
    """The demo recovers on its own when a session ends.

    Demo bots are bounded, so this is the ordinary path rather than a
    recovery case: the slot frees, the account frees, and the next
    visitor gets a bot without anybody restarting anything.
    """
    assert (await demo_client.post("/demo/spawn")).status == 201
    assert (await demo_client.post("/demo/spawn")).status == 201
    spawner.processes[0].returncode = 0

    body = await _fleet_body(demo_client)
    assert require_int(body, "running") == 1
    assert [
        narrow_json_to_str(narrow_json_to_dict(row)["slot"]) for row in require_list(body, "bots")
    ] == ["demo-2"]

    again = await demo_client.post("/demo/spawn")
    assert again.status == 201
    assert require_str(narrow_json_to_dict(load_json_str(await again.text())), "slot") == "demo-1"


@pytest.mark.asyncio
async def test_the_demo_never_lists_the_operators_own_bots(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """An operator bot is invisible to the public surface.

    The two surfaces share one registry, so this is the property that
    keeps them apart: the operator's instance is named for its account
    and the demo neither counts it nor names it.
    """
    assert (await demo_client.post("/bots", json=_OPERATOR_SPAWN)).status == 201

    body = await _fleet_body(demo_client)

    assert require_int(body, "running") == 0
    assert require_list(body, "bots") == []


@pytest.mark.asyncio
async def test_the_demo_video_route_refuses_an_operator_instance(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """A live operator bot cannot be watched by guessing its name.

    404 before the registry is consulted at all: the slot grammar is
    the gate, so a correct guess of a real instance name buys nothing.
    """
    assert (await demo_client.post("/bots", json=_OPERATOR_SPAWN)).status == 201

    response = await demo_client.get("/demo/video/artax")

    assert response.status == 404
    assert "is not a demo slot" in await response.text()


@pytest.mark.asyncio
async def test_the_demo_video_route_relays_a_demo_slot(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """A demo slot streams through the same relay the operator uses."""
    assert (await demo_client.post("/demo/spawn")).status == 201
    stream = _FakeChildVideoStream([b"one", b"two"])
    original = service_hooks.open_child_video
    service_hooks.open_child_video = _Opener(stream)
    try:
        response = await demo_client.get("/demo/video/demo-1")
        body = await response.read()
    finally:
        service_hooks.open_child_video = original

    assert response.status == 200
    assert body == b"onetwo"
    assert stream.closes == 1


@pytest.mark.asyncio
async def test_a_child_that_has_not_bound_its_port_yet_is_503_not_500(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """Watching a bot the moment it starts says "not yet", not "broken".

    A child is ``alive`` from the instant it is forked and its video
    port opens seconds later, so the demo page — which draws a tile as
    soon as a bot appears — asks early every single time. Uncaught, the
    connection refusal reached the boundary as a 500 and reported the
    server as broken while the bot was merely still booting.
    """
    assert (await demo_client.post("/demo/spawn")).status == 201
    original = service_hooks.open_child_video
    service_hooks.open_child_video = _RefusingOpener()
    try:
        response = await demo_client.get("/demo/video/demo-1")
        body = await response.text()
    finally:
        service_hooks.open_child_video = original

    assert response.status == 503
    assert response.headers["Retry-After"] == str(CHILD_WARMUP_RETRY_SECONDS)
    # The cause travels: a refused connection that is NOT a warming
    # child reads identically from outside, so the body has to say
    # which one this was.
    assert "is not serving video yet" in body
    assert "connection refused by the child" in body


@pytest.mark.asyncio
async def test_the_demo_video_route_refuses_a_slot_nothing_is_playing_in(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """A well-formed but empty slot is a 404, not an empty stream."""
    response = await demo_client.get("/demo/video/demo-4")

    assert response.status == 404
    assert "unknown instance" in await response.text()


@pytest.mark.asyncio
async def test_a_draining_fleet_says_so_and_starts_nothing(
    demo_client: TestClient[web.Request, web.Application],
) -> None:
    """Shutdown is visible to the page and closed to new bots.

    A page that could not see the drain would show a working button
    that refused every press without explaining itself.
    """
    assert (await demo_client.post("/demo/spawn")).status == 201
    assert (await demo_client.post("/shutdown")).status == 202

    body = await _fleet_body(demo_client)
    assert require_bool(body, "draining") is True

    refused = await demo_client.post("/demo/spawn")
    assert refused.status == 409
    assert "shutting down" in await refused.text()


def test_a_machine_with_no_accounts_has_no_demo(spawner: _FakeSpawner) -> None:
    """Capacity zero is reported honestly and refuses with a reason."""
    _ = spawner
    originals = _with_account_pool()
    try:
        manager = FleetManager()
        assert demo_capacity() == 0
        assert demo_slots() == []
        with pytest.raises(FleetError, match="no accounts are configured"):
            demo_spawn(manager)
    finally:
        _restore_account_hooks(originals)


def test_the_slot_ceiling_holds_when_accounts_outnumber_it(spawner: _FakeSpawner) -> None:
    """More accounts than slots fills the demo without spilling past it.

    The ceiling and the account pool are separate limits and the
    smaller wins. With seven accounts behind a five-slot demo, the
    refusal is the SLOT one — the account branch cannot be what stops
    a sixth press.
    """
    originals = _with_account_pool("a1", "a2", "a3", "a4", "a5", "a6", "a7")
    try:
        manager = FleetManager()
        assert demo_capacity() == DEMO_MAX_BOTS
        slots = [demo_spawn(manager)["slot"] for _ in range(DEMO_MAX_BOTS)]
        assert slots == ["demo-1", "demo-2", "demo-3", "demo-4", "demo-5"]
        with pytest.raises(FleetError, match="the demo is full"):
            demo_spawn(manager)
    finally:
        _restore_account_hooks(originals)
    _ = spawner


def test_uptime_counts_whole_seconds_since_the_spawn(spawner: _FakeSpawner) -> None:
    """A row's uptime is derived from the clock, not carried from spawn."""
    _ = spawner
    originals = _with_account_pool("alpha")
    original_clock = top_hooks.get_current_time_ms
    clock = _SteppingClock(start_ms=1_000_000)
    top_hooks.get_current_time_ms = clock
    try:
        manager = FleetManager()
        assert demo_spawn(manager)["uptime_seconds"] == 0
        clock.now_ms += 12_400
        assert demo_fleet(manager)["bots"][0]["uptime_seconds"] == 12
    finally:
        top_hooks.get_current_time_ms = original_clock
        _restore_account_hooks(originals)


def test_every_slot_name_the_demo_can_hand_out_resolves(spawner: _FakeSpawner) -> None:
    """The grammar admits the full ceiling, not just the current capacity.

    Shrinking ``accounts.json`` must not make a running bot unwatchable
    — capacity governs the next SPAWN, never an existing read — so the
    gate is checked against the ceiling and refuses everything else.
    """
    _ = spawner
    for index in range(DEMO_MAX_BOTS):
        name = f"demo-{index + 1}"
        assert demo_slot_or_refuse(name) == name
    for rejected in ("demo-0", f"demo-{DEMO_MAX_BOTS + 1}", "demo-", "demo-1x", "artax", ""):
        with pytest.raises(FleetError, match="is not a demo slot"):
            demo_slot_or_refuse(rejected)


class _SteppingClock:
    """A millisecond clock a test moves by hand."""

    def __init__(self, start_ms: int) -> None:
        """Start the clock.

        Args:
            start_ms: Initial epoch milliseconds.
        """
        self.now_ms = start_ms

    def __call__(self) -> int:
        """Read the clock.

        Returns:
            The current epoch milliseconds.
        """
        return self.now_ms


class _FakeChildVideoStream:
    """A child video stream over a fixed chunk list."""

    def __init__(self, chunks: list[bytes]) -> None:
        """Bind the stream to the bytes it will yield.

        Args:
            chunks: Body chunks to yield in order.
        """
        self._chunks = chunks
        self.closes = 0

    @property
    def content_type(self) -> str:
        """The upstream content type.

        Returns:
            A multipart type carrying the child's own boundary.
        """
        return "multipart/x-mixed-replace; boundary=demoframe"

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield the bound chunks.

        Yields:
            Each chunk in order.
        """
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        """Record one release."""
        self.closes += 1


class _RefusingOpener:
    """An opener that refuses, as a child mid-boot does."""

    async def __call__(self, url: str) -> _FakeChildVideoStream:
        """Refuse the connection.

        Args:
            url: Upstream URL the route asked for.

        Returns:
            Never returns.

        Raises:
            ClientConnectionError: Always.
        """
        raise ClientConnectionError(f"connection refused by the child at {url}")


class _Opener:
    """An opener handing back one bound stream."""

    def __init__(self, stream: _FakeChildVideoStream) -> None:
        """Bind the opener to its stream.

        Args:
            stream: Stream returned for every call.
        """
        self._stream = stream

    async def __call__(self, url: str) -> _FakeChildVideoStream:
        """Return the bound stream.

        Args:
            url: Upstream URL the route asked for.

        Returns:
            The bound stream.
        """
        _ = url
        return self._stream
