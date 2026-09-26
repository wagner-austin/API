"""Who this bridge is, and the values that must never change.

THE SESSION ID IS PINNED AS A LITERAL, NOT RE-DERIVED, for the reason
lock-wake's identity test gives: re-deriving it would keep agreeing with
``identity.py`` after somebody edited ``_SESSION_NAME``, and that edit mints
a second identity the board refuses with ``TASK_IDENTITY_MISMATCH``.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.journal_cursor import cursor_path

from fleet_health_wake.identity import BRIDGE_AGENT, CURSOR_READER, IDENTITY, load_task_id
from tests.conftest import CONFIGURED_ENV, TASK_ID, pin_env


class TestIdentity:
    def test_the_session_id_is_pinned(self) -> None:
        assert IDENTITY["session_id"] == "615d5c7b-a22c-5ab0-8855-46b1eb84d688"

    def test_the_label_is_kebab_case_and_service_shaped(self) -> None:
        assert IDENTITY["agent"] == BRIDGE_AGENT == "bridge-fleet-health-0926"

    def test_the_cwd_declares_a_service(self) -> None:
        """The ``service://`` prefix is what the session audits key on."""
        assert IDENTITY["cwd"] == "service://fleet-health-wake"

    def test_the_cursor_file_sits_beside_the_journal(self) -> None:
        journal = pathlib.Path("C:/Users/Test/PROJECTS/MCPs/fleet-mcp/state/health-events.jsonl")
        assert cursor_path(journal, CURSOR_READER) == pathlib.Path(
            "C:/Users/Test/PROJECTS/MCPs/fleet-mcp/state/"
            "health-events.jsonl.fleet-health-wake-offset.json"
        )


class TestLoadTaskId:
    def test_it_reads_the_standing_task_from_the_environment(self) -> None:
        pin_env(CONFIGURED_ENV)
        assert load_task_id() == TASK_ID

    def test_an_unset_variable_refuses_with_its_own_name(self) -> None:
        pin_env({})
        with pytest.raises(AppError, match="FLEET_HEALTH_WAKE_TASK_ID"):
            load_task_id()
