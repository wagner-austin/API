"""The dashboard: the queue drawn as HTML, through the door's own route.

The renderer composes the same queue reads the NDJSON surface uses, so
the tests drive it the same way that surface is tested: submit through
the router, play jobs through the real claim and finish, then read the
page and assert the numbers landed in the right cells.
"""

from __future__ import annotations

import pytest

from rw_bot.service.dashboard import render_dashboard
from rw_bot.service.http import route_service_request
from rw_bot.service.queue import claim, finish, fleet_batches, running_matches
from rw_bot.service.queue_rows import MatchServiceError
from rw_bot.wire.ndjson import render_json
from tests.service_fakes import FakeConnection

_JOBS_TEXT = (
    "control|12345|doctrines/flame-nocover.doctrine|400\n"
    "navy2|12345|doctrines/flame-navy2.doctrine|400"
)


def _submission(name: str) -> bytes:
    return render_json(
        {
            "name": name,
            "jobs": _JOBS_TEXT,
            "lockstep": 75,
            "map_path": "maps/skirmish/[p2]duel_lake.tmx",
            "difficulty": 2,
            "pin_delta": 3,
            "fast_forward": 10,
        }
    ).encode("utf-8")


def test_the_root_route_serves_the_page_as_html() -> None:
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission("demo"))
    status, kind, payload = route_service_request(conn, "GET", "/", b"")
    assert status == 200
    assert kind == "text/html; charset=utf-8"
    page = payload.decode("utf-8")
    assert "<h2>demo</h2>" in page
    assert "<td class='arm'>control</td>" in page
    assert "<td class='arm'>navy2</td>" in page


def test_a_running_match_appears_as_its_lane() -> None:
    """A claimed match shows its engine slot, batch, arm and worker."""
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission("demo"))
    held = claim(conn, "w1", (3,))
    if held is None:
        raise AssertionError("expected a claim and the queue declined")
    page = render_dashboard(conn)
    lane = (
        "<tr><td class='running'>3</td><td class='arm'>demo</td>"
        "<td class='arm'>control</td><td>12345</td><td class='arm'>w1</td></tr>"
    )
    assert lane in page
    assert "<span class='running'>1 running</span>" in page


def test_an_idle_fleet_says_so() -> None:
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission("demo"))
    page = render_dashboard(conn)
    assert "<p class='sub'>all lanes idle</p>" in page


def test_a_finished_win_lands_in_the_won_cell() -> None:
    """One claim, one win: the arm's row shows done 1 and won 1."""
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission("demo"))
    held = claim(conn, "w1", (1,))
    if held is None:
        raise AssertionError("expected a claim and the queue declined")
    finish(conn, held["job_id"], True, "### control-s12345\nverdict        won (won)")
    page = render_dashboard(conn)
    row = (
        "<tr><td class='arm'>control</td>"
        "<td class='muted'>0</td><td class='muted'>0</td><td class='done'>1</td>"
        "<td class='won'>1</td><td class='muted'>0</td><td class='muted'>0</td>"
        "<td class='muted'>0</td><td class='muted'>0</td></tr>"
    )
    assert row in page
    assert "<span class='running'>0 running</span>" in page
    assert "<span class='queued'>1 queued</span>" in page


def test_batches_lead_newest_first() -> None:
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission("older"))
    route_service_request(conn, "POST", "/batches", _submission("newer"))
    page = render_dashboard(conn)
    assert page.index("<h2>newer</h2>") < page.index("<h2>older</h2>")


def test_a_corrupt_running_row_is_a_loud_stop() -> None:
    """A running row with a non-int seed is refused, never drawn."""
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission("demo"))
    held = claim(conn, "w1", (1,))
    if held is None:
        raise AssertionError("expected a claim and the queue declined")
    conn.store.jobs[0].seed = "not-a-seed"
    with pytest.raises(MatchServiceError) as caught:
        running_matches(conn)
    assert caught.value.code == "RW-SERVICE-001"


def test_an_unknown_state_is_counted_nowhere() -> None:
    """A planted state outside the lifecycle tallies no cell at all."""
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission("demo"))
    conn.store.jobs[0].state = "limbo"
    page = render_dashboard(conn)
    row = (
        "<tr><td class='arm'>control</td>"
        "<td class='muted'>0</td><td class='muted'>0</td><td class='muted'>0</td>"
        "<td class='muted'>0</td><td class='muted'>0</td><td class='muted'>0</td>"
        "<td class='muted'>0</td><td class='muted'>0</td></tr>"
    )
    assert row in page


def test_a_corrupt_batch_row_is_a_loud_stop() -> None:
    """A non-string batch name is refused, never drawn."""
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission("demo"))
    conn.store.jobs[0].batch = 7
    with pytest.raises(MatchServiceError) as caught:
        fleet_batches(conn)
    assert caught.value.code == "RW-SERVICE-001"
    status, _kind, payload = route_service_request(conn, "GET", "/", b"")
    assert status == 400
    assert b"RW-SERVICE-001" in payload
