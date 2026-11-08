"""Native observation identity, state, pagination, and failure contracts."""

from __future__ import annotations

from collections.abc import Generator, Sequence

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from fleet.core import _test_hooks, _test_hooks_native, native_observe, observe
from fleet.core.ssh_options import SSH_OPTIONS
from tests._queue_fakes import FakeQueue
from tests.conftest import FakeRun, ok
from tests.test_observe import BOARD, EVIDENCE, IDENTITY, document, node

SESSION = "11111111-1111-4111-8111-111111111111"
MACHINE = "linux:ziggurat"


def inventory(*rows: JSONObject, offset: int = 0, more: bool = False) -> str:
    """Encode a ledger page using the existing listing contract."""
    return dump_json_str(
        {
            "rows": list(rows),
            "pagination": {
                "offset": offset,
                "limit": 200,
                "hasMore": more,
                "nextOffset": offset + len(rows),
            },
        }
    )


def identity_row(**changes: JSONValue) -> JSONObject:
    """One registered target; deliberately includes no inferred runtime state."""
    return {
        "sessionId": SESSION,
        "machine": MACHINE,
        "harness": "codex",
        "selfRegistered": True,
        **changes,
    }


class Runtime:
    """Transport seam double paired with test_native_rpc's real process tests."""

    def __init__(self, threads: tuple[JSONObject | str, ...], failure: str | None = None) -> None:
        self.threads = threads
        self.failure = failure
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...], int]] = []

    def __call__(
        self, argv: Sequence[str], thread_ids: tuple[str, ...], *, timeout_seconds: int
    ) -> tuple[tuple[JSONObject | str, ...], str | None]:
        self.calls.append((tuple(argv), thread_ids, timeout_seconds))
        return self.threads, self.failure


@pytest.fixture(autouse=True)
def reset_native_hook() -> Generator[None, None, None]:
    saved = _test_hooks_native.read
    yield
    _test_hooks_native.read = saved


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"type": "idle"}, ("idle", None)),
        ({"type": "notLoaded"}, ("not-loaded", None)),
        ({"type": "systemError"}, ("system-error", "Codex runtime reports systemError")),
        ({"type": "active", "activeFlags": []}, ("active", None)),
        ({"type": "active", "activeFlags": ["waitingOnApproval"]}, ("waiting-approval", None)),
        ({"type": "active", "activeFlags": ["waitingOnUserInput"]}, ("waiting-input", None)),
        (
            {"type": "active", "activeFlags": ["waitingOnApproval", "waitingOnUserInput"]},
            ("waiting-approval", None),
        ),
    ],
)
def test_native_status(status: JSONObject, expected: tuple[str, str | None]) -> None:
    assert native_observe.thread_status({"id": SESSION, "status": status}, SESSION) == expected


@pytest.mark.parametrize(
    ("thread", "message"),
    [
        ({"id": "other", "status": {"type": "idle"}}, "IDENTITY"),
        ({"id": SESSION, "status": {"type": "invented"}}, "STATUS"),
        ({"id": SESSION, "status": {"type": "active", "activeFlags": ["invented"]}}, "STATUS"),
    ],
)
def test_refuses_unknown_identity_or_status(thread: JSONObject, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        native_observe.thread_status(thread, SESSION)


def test_pages_inventory_filters_and_deduplicates_registered_targets() -> None:
    board = FakeQueue(
        [
            inventory(identity_row(), identity_row(harness="claude-code"), more=True),
            inventory(identity_row(), identity_row(selfRegistered=False), offset=2),
        ]
    )
    _test_hooks.http_post = board
    assert native_observe.registered_codex(BOARD, MACHINE, IDENTITY) == (SESSION,)
    assert board.tools == ["task_sessions_list", "task_sessions_list"]
    assert board.arguments == [
        {**IDENTITY, "machine": MACHINE, "format": "json", "limit": 200, "offset": offset}
        for offset in (0, 2)
    ]


def sized(*rows: JSONObject, offset: int, total: int) -> str:
    """A page the board served short of the 200 asked, sized to its list budget.

    Its limit is the rows served and its nextOffset the first row left out,
    as MCPs task_sessions_list answers a page past its budget (board task
    d2c66a85).
    """
    return dump_json_str(
        {
            "rows": list(rows),
            "pagination": {
                "total": total,
                "offset": offset,
                "limit": len(rows),
                "hasMore": True,
                "nextOffset": offset + len(rows),
            },
        }
    )


def test_a_machine_whose_rows_exceed_one_budgeted_page_is_read_whole() -> None:
    # 2026-11-08: one machine's 200 json rows answered 141,366 characters
    # against the 48,000-character list budget, so the board now serves the
    # rows that fit and the arm reads on from where the page stopped.
    other = "22222222-2222-4222-8222-222222222222"
    claude = [
        identity_row(sessionId=f"00000000-0000-4000-8000-{n:012x}", harness="claude-code")
        for n in range(3)
    ]
    board = FakeQueue(
        [
            sized(identity_row(), *claude[:2], offset=0, total=5),
            sized(claude[2], offset=3, total=5),
            inventory(identity_row(sessionId=other), offset=4),
        ]
    )
    _test_hooks.http_post = board
    assert native_observe.registered_codex(BOARD, MACHINE, IDENTITY) == (SESSION, other)
    assert board.arguments == [
        {**IDENTITY, "machine": MACHINE, "format": "json", "limit": 200, "offset": offset}
        for offset in (0, 3, 4)
    ]


def test_a_full_page_then_a_sized_one_reach_offsets_past_two_hundred() -> None:
    # Offsets past 256 are new int objects on every parse, so the window is
    # compared by value: a full 200-row page, a sized 100-row page, then the
    # last page at offset 300.
    rows = [
        identity_row(sessionId=f"00000000-0000-4000-8000-{n:012x}", harness="claude-code")
        for n in range(300)
    ]
    board = FakeQueue(
        [
            inventory(*rows[:200], more=True),
            sized(*rows[200:], offset=200, total=301),
            inventory(identity_row(), offset=300),
        ]
    )
    _test_hooks.http_post = board
    assert native_observe.registered_codex(BOARD, MACHINE, IDENTITY) == (SESSION,)
    assert [arguments["offset"] for arguments in board.arguments] == [0, 200, 300]


@pytest.mark.parametrize(
    "reply",
    [
        inventory(identity_row(machine="linux:another")),
        inventory(offset=1),
        inventory(offset=-1),
        dump_json_str({"rows": [], "pagination": {"offset": 0, "limit": 1, "hasMore": False}}),
        # A page wider than the 200 asked is not the window asked for.
        dump_json_str(
            {
                "rows": [identity_row()],
                "pagination": {"offset": 0, "limit": 201, "hasMore": True, "nextOffset": 1},
            }
        ),
        # A short page whose limit is not the rows it carries is not a
        # budgeted page: the window it names and the rows disagree.
        dump_json_str(
            {
                "rows": [identity_row()],
                "pagination": {"offset": 0, "limit": 2, "hasMore": True, "nextOffset": 1},
            }
        ),
        # A budgeted page can never be empty while rows remain.
        dump_json_str(
            {
                "rows": [],
                "pagination": {"offset": 0, "limit": 0, "hasMore": True, "nextOffset": 0},
            }
        ),
        inventory(more=True),
        dump_json_str(
            {
                "rows": [identity_row()],
                "pagination": {
                    "offset": 0,
                    "limit": 200,
                    "hasMore": True,
                    "nextOffset": 0,
                },
            }
        ),
    ],
)
def test_refuses_mismatched_or_nonadvancing_pages(reply: str) -> None:
    _test_hooks.http_post = FakeQueue([reply])
    with pytest.raises(ValueError, match="CODEX_OBSERVE_"):
        native_observe.registered_codex(BOARD, MACHINE, IDENTITY)


def test_empty_inventory_never_opens_proxy() -> None:
    _test_hooks.http_post = FakeQueue([inventory()])
    runtime = Runtime(())
    _test_hooks_native.read = runtime
    assert (
        native_observe.observe_codex(BOARD, "ziggurat", "corvis", MACHINE, IDENTITY, EVIDENCE) == []
    )
    assert runtime.calls == []


@pytest.mark.parametrize("failure", [None, "CODEX_OBSERVE_UNAVAILABLE: TimeoutError"])
def test_existing_observer_submits_native_evidence_with_bounded_expiry(failure: str | None) -> None:
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="ziggurat"))])
    _test_hooks.now = lambda: 1000
    board = FakeQueue([inventory(identity_row()), "observed native evidence"])
    _test_hooks.http_post = board
    runtime = Runtime(({"id": SESSION, "status": {"type": "active", "activeFlags": []}},), failure)
    _test_hooks_native.read = runtime
    result = observe.observe_node(BOARD, node("ziggurat", "session-host"), IDENTITY, EVIDENCE)
    assert result == {
        "node": "ziggurat",
        "recorded": True,
        "detail": "native evidence for 1 Codex session(s) read at "
        "1970-01-01T00:16:40+00:00 lasts 450 s; observed native evidence",
    }
    assert runtime.calls == [
        (
            (
                "ssh",
                *SSH_OPTIONS,
                "ziggurat",
                "/home/corvis/.local/bin/codex",
                "app-server",
                "proxy",
            ),
            (SESSION,),
            20,
        )
    ]
    assert board.arguments[1] == {
        **IDENTITY,
        "machine": MACHINE,
        "observations": [],
        "nativeObservations": [
            {
                "sessionId": SESSION,
                "harness": "codex",
                "status": "active" if failure is None else "unavailable",
                "detail": failure,
                "observedAt": "1970-01-01T00:16:40+00:00",
                "expiresAt": "1970-01-01T00:24:10+00:00",
            }
        ],
    }


def test_a_machine_past_one_budgeted_page_still_has_its_pass_recorded() -> None:
    # MCPs board task d2c66a85: from 02:12Z on 2026-11-08 every hub tick was
    # refused at ziggurat's oversized ledger page and no pass was recorded.
    # The board now serves the rows that fit; the pass reads on and records.
    claude = identity_row(sessionId="33333333-3333-4333-8333-333333333333", harness="claude-code")
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="ziggurat"))])
    _test_hooks.now = lambda: 1000
    board = FakeQueue(
        [
            sized(claude, offset=0, total=2),
            inventory(identity_row(), offset=1),
            "observed native evidence",
        ]
    )
    _test_hooks.http_post = board
    _test_hooks_native.read = Runtime(({"id": SESSION, "status": {"type": "notLoaded"}},))
    result = observe.observe_node(BOARD, node("ziggurat", "session-host"), IDENTITY, EVIDENCE)
    assert result == {
        "node": "ziggurat",
        "recorded": True,
        "detail": "native evidence for 1 Codex session(s) read at "
        "1970-01-01T00:16:40+00:00 lasts 450 s; observed native evidence",
    }
    assert board.tools == ["task_sessions_list", "task_sessions_list", "task_session_observe"]
    observed = board.arguments[2]
    assert observed["machine"] == MACHINE
    assert observed["nativeObservations"] == [
        {
            "sessionId": SESSION,
            "harness": "codex",
            "status": "not-loaded",
            "detail": None,
            "observedAt": "1970-01-01T00:16:40+00:00",
            "expiresAt": "1970-01-01T00:24:10+00:00",
        }
    ]


def test_incomplete_runtime_batch_does_not_guess_idle() -> None:
    _test_hooks.http_post = FakeQueue([inventory(identity_row())])
    _test_hooks_native.read = Runtime(())
    with pytest.raises(ValueError, match="incomplete runtime response"):
        native_observe.observe_codex(BOARD, "ziggurat", "corvis", MACHINE, IDENTITY, EVIDENCE)


def test_missing_runtime_thread_is_unavailable_not_departed() -> None:
    assert native_observe.thread_status(
        "CODEX_OBSERVE_RPC: thread/read returned an error", SESSION
    ) == (
        "unavailable",
        "CODEX_OBSERVE_RPC: thread/read returned an error",
    )


def test_excessive_inventory_fails_explicitly() -> None:
    _test_hooks.http_post = FakeQueue(
        [
            inventory(
                *[
                    identity_row(sessionId=f"00000000-0000-4000-8000-{index:012d}")
                    for index in range(201)
                ]
            )
        ]
    )
    with pytest.raises(ValueError, match="CODEX_OBSERVE_LIMIT"):
        native_observe.observe_codex(BOARD, "ziggurat", "corvis", MACHINE, IDENTITY, EVIDENCE)
