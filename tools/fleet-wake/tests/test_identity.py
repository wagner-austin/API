"""The bridge's identity is a service contract: same pair, every run."""

from __future__ import annotations

import re

import pytest
from platform_core.error_codes import BoardBridgeErrorCode
from platform_core.errors import AppError

from fleet_wake.identity import BRIDGE_AGENT, IDENTITY, TASK_ID_VARIABLE, load_task_id
from tests.conftest import CONFIGURED_ENV, TASK_ID, pin_env


class TestIdentityConstants:
    def test_the_session_id_is_this_exact_uuid_forever(self) -> None:
        """GOLDEN VALUE, not a recomputation. A test deriving the id the way
        the implementation does would pass through any change to the
        derivation -- and a changed id is a NEW board identity, which the
        one-session-one-label rule turns into refused posts with no way to
        unbind the old label."""
        assert IDENTITY["session_id"] == "0a6cb261-eaa4-5330-84b9-079a1afe268a"

    def test_it_is_not_the_hpc_wake_bridges_identity(self) -> None:
        """Two bridges sharing one session id means the second to write is
        refused under the first's label. Asserted against the literal the
        other package pins, so a copy-paste of its ``_SESSION_NAME`` fails
        here rather than on the board.

        The LABEL is not compared here: mypy reads both sides as literals and
        rejects the check as statically non-overlapping, which is a fair
        complaint -- the label is pinned exactly above instead.
        """
        assert IDENTITY["session_id"] != "b6048b2e-2e32-5247-a488-7b4ccc35f2cc"

    def test_the_label_satisfies_the_board_grammar(self) -> None:
        """Kebab-case, 3-64 chars: the same regex taskboard-mcp validates."""
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}[a-z0-9]", BRIDGE_AGENT)
        assert BRIDGE_AGENT == "bridge-fleet-wake-0906"

    def test_the_recorded_cwd_names_the_service_not_a_directory(self) -> None:
        """A service has no directory a person could open, and recording one
        that does not exist would send a reader looking for it."""
        assert IDENTITY["cwd"] == "service://fleet-wake"


class TestLoadTaskId:
    def test_the_configured_id_is_read_back(self) -> None:
        pin_env(CONFIGURED_ENV)

        assert load_task_id() == TASK_ID

    def test_an_unset_variable_refuses_and_names_itself(self) -> None:
        """The reader's next action is to export it, and a refusal that does
        not say which variable sends them to grep."""
        pin_env({})

        with pytest.raises(AppError) as caught:
            load_task_id()

        assert caught.value.code is BoardBridgeErrorCode.TASK_ID_MISSING
        assert TASK_ID_VARIABLE in caught.value.message

    def test_a_blank_variable_is_the_unset_case(self) -> None:
        """The shared reader trims; an exported blank must not become a task
        id the board then rejects one layer later, where the message would
        blame the board for a shell mistake."""
        pin_env({TASK_ID_VARIABLE: "   "})

        with pytest.raises(AppError) as caught:
            load_task_id()

        assert caught.value.code is BoardBridgeErrorCode.TASK_ID_MISSING
