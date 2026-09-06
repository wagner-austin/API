"""Posting to the board as a service: identity, argument shape, and refusal.

WHY THE IDENTITY TESTS PIN LITERALS. The board binds a session id to an agent
label on first write and never releases it, so a change to the derivation is
not a refactor -- it is a new writer that the board refuses for the rest of the
service's life, with no way to unbind the old one. A test that re-derived the
id the way the implementation does would pass through exactly that change. The
two live services' ids are therefore written out as strings, and a change to
:func:`~platform_core.board.mint_service_session_id` fails here rather than in
production.
"""

from __future__ import annotations

import pytest

from platform_core.board import (
    BOARD_ROOM,
    BoardIdentity,
    encode_board_post,
    mint_service_session_id,
    post_to_task,
    require_task_id,
    service_identity,
)
from platform_core.error_codes import BoardBridgeErrorCode, McpClientErrorCode
from platform_core.errors import AppError
from platform_core.mcp_client import McpCredentials, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments

CREDENTIALS = McpCredentials(
    url="http://127.0.0.1:8033/mcp",
    api_key="test-key",
    tenant_id="2e137b5f-0000-4000-8000-000000000000",
)

TASK_ID = "df6f1dc8-cd6b-4314-b28a-eb3625390ae0"

IDENTITY = BoardIdentity(
    agent="bridge-example-0906",
    session_id="00000000-0000-5000-8000-000000000000",
    cwd="service://example",
)


class TestServiceSessionIdIsPermanent:
    def test_the_hpc_wake_bridge_keeps_the_id_the_board_already_bound(self) -> None:
        """GOLDEN VALUE FOR A LIVE SERVICE. bridge-hpc-wake-0906 has been
        posting under this id since 2026-09-06; the board holds the binding
        permanently. This literal is what proved the lift out of hpc-wake
        changed no behaviour, and it is what stops a later edit to the
        namespace or the name from locking that bridge out."""
        assert (
            mint_service_session_id("corvis:hpc-wake:bridge")
            == "b6048b2e-2e32-5247-a488-7b4ccc35f2cc"
        )

    def test_the_fleet_wake_bridge_has_its_own_id(self) -> None:
        assert (
            mint_service_session_id("corvis:fleet-wake:bridge")
            == "0a6cb261-eaa4-5330-84b9-079a1afe268a"
        )

    def test_the_same_name_yields_the_same_id_every_call(self) -> None:
        """The property the whole design rests on: a restart is not a new
        identity."""
        assert mint_service_session_id("corvis:x:bridge") == mint_service_session_id(
            "corvis:x:bridge"
        )

    def test_different_names_yield_different_ids(self) -> None:
        """Two bridges must not collide on one board identity, or the second
        to write is refused under the first's label."""
        assert mint_service_session_id("corvis:a:bridge") != mint_service_session_id(
            "corvis:b:bridge"
        )


class TestServiceIdentity:
    def test_it_carries_the_label_the_derived_id_and_the_stated_cwd(self) -> None:
        identity = service_identity(
            agent="bridge-example-0906",
            service="corvis:example:bridge",
            cwd="service://example",
        )

        assert identity == BoardIdentity(
            agent="bridge-example-0906",
            session_id=mint_service_session_id("corvis:example:bridge"),
            cwd="service://example",
        )

    def test_cwd_is_taken_as_given_and_never_built_from_the_label(self) -> None:
        """The reason it is a parameter. hpc-wake records
        ``service://hpc-wake`` and its label is ``bridge-hpc-wake-0906``; a
        derivation would have rewritten a live audit trail during a lift that
        was supposed to change nothing."""
        identity = service_identity(
            agent="bridge-hpc-wake-0906",
            service="corvis:hpc-wake:bridge",
            cwd="service://hpc-wake",
        )

        assert identity["cwd"] == "service://hpc-wake"
        assert identity["cwd"] != f"service://{identity['agent']}"


class TestRequireTaskId:
    def test_a_configured_id_is_returned(self) -> None:
        assert require_task_id(TASK_ID, variable="ANY_TASK_ID") == TASK_ID

    def test_an_absent_id_refuses_and_names_the_variable(self) -> None:
        """Named because the reader's next action is to export it, and a
        refusal that does not say which variable sends them to grep."""
        with pytest.raises(AppError) as caught:
            require_task_id(None, variable="FLEET_WAKE_TASK_ID")

        assert caught.value.code is BoardBridgeErrorCode.TASK_ID_MISSING
        assert "FLEET_WAKE_TASK_ID" in caught.value.message


class TestEncodeBoardPost:
    def test_every_key_is_the_boards_spelling(self) -> None:
        """THE POINT OF THE FUNCTION. ``taskId`` and ``sessionId`` are
        camelCase because the board validates them that way; a bridge that
        spelled either in snake_case would be refused at the endpoint rather
        than at review, and holding it here means two bridges cannot spell
        them differently."""
        arguments = encode_board_post(IDENTITY, task_id=TASK_ID, kind="note", body="hello")

        assert arguments == {
            "room": BOARD_ROOM,
            "taskId": TASK_ID,
            "kind": "note",
            "body": "hello",
            "agent": "bridge-example-0906",
            "sessionId": "00000000-0000-5000-8000-000000000000",
            "cwd": "service://example",
        }

    def test_the_kind_is_carried_rather_than_fixed(self) -> None:
        arguments = encode_board_post(IDENTITY, task_id=TASK_ID, kind="checkin", body="x")

        assert arguments["kind"] == "checkin"


class TestPostToTask:
    def test_the_request_carries_the_credentials_and_the_encoded_arguments(self) -> None:
        fake = FakeHttpPost([posted_ok()])

        post_to_task(fake, CREDENTIALS, IDENTITY, task_id=TASK_ID, kind="note", body="hello")

        assert fake.urls == ["http://127.0.0.1:8033/mcp"]
        assert fake.headers[0]["x-api-key"] == "test-key"
        assert fake.headers[0]["X-Tenant-Id"] == "2e137b5f-0000-4000-8000-000000000000"
        assert sent_arguments(fake.bodies[0]) == encode_board_post(
            IDENTITY, task_id=TASK_ID, kind="note", body="hello"
        )

    def test_a_refused_post_raises_rather_than_reporting_delivery(self) -> None:
        """A rotated key is the ORDINARY failure here, and it must end the
        cycle loudly. A bridge that swallowed this would write its position
        record and never announce the work again -- reproducing the exact
        silence it exists to remove."""
        fake = FakeHttpPost(
            [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
        )

        with pytest.raises(AppError) as caught:
            post_to_task(fake, CREDENTIALS, IDENTITY, task_id=TASK_ID, kind="note", body="hello")

        assert caught.value.code is McpClientErrorCode.HTTP_STATUS
        assert "401" in caught.value.message
