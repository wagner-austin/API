"""The bridge's identity is a service contract: same pair, every run."""

from __future__ import annotations

import re

import pytest
from platform_core.error_codes import HpcWakeErrorCode
from platform_core.errors import AppError

from hpc_wake.identity import (
    BRIDGE_AGENT,
    BRIDGE_SESSION_ID,
    TASK_ID_VARIABLE,
    load_task_id,
)
from tests.conftest import CONFIGURED_ENV, TASK_ID, pin_env


class TestIdentityConstants:
    def test_the_session_id_is_this_exact_uuid_forever(self) -> None:
        """Golden value, not a recomputation. A test deriving the id the way
        the implementation does would pass through any change to the
        derivation -- and a changed id is a NEW board identity, which the
        one-session-one-label rule turns into refused posts."""
        assert BRIDGE_SESSION_ID == "b6048b2e-2e32-5247-a488-7b4ccc35f2cc"

    def test_the_label_satisfies_the_board_grammar(self) -> None:
        """Kebab-case, 3-64 chars: the same regex taskboard-mcp validates."""
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}[a-z0-9]", BRIDGE_AGENT)
        assert BRIDGE_AGENT == "bridge-hpc-wake-0906"


class TestLoadTaskId:
    def test_the_configured_id_is_read_back(self) -> None:
        pin_env(CONFIGURED_ENV)
        assert load_task_id() == TASK_ID

    def test_an_unset_variable_refuses_and_names_itself(self) -> None:
        pin_env({})
        with pytest.raises(AppError) as caught:
            load_task_id()
        assert caught.value.code is HpcWakeErrorCode.TASK_ID_MISSING
        assert TASK_ID_VARIABLE in caught.value.message

    def test_a_blank_variable_is_the_unset_case(self) -> None:
        """The shared reader trims; an exported blank must not become a
        task id the board then rejects one layer later."""
        pin_env({TASK_ID_VARIABLE: "   "})
        with pytest.raises(AppError) as caught:
            load_task_id()
        assert caught.value.code is HpcWakeErrorCode.TASK_ID_MISSING
