"""Who this bridge is, and the one value that must never change.

THE SESSION ID IS PINNED AS A LITERAL, NOT RE-DERIVED. Re-deriving it with
``uuid5`` here would assert that this file and ``identity.py`` agree with
each other, which they would continue to do after somebody edited
``_SESSION_NAME`` -- and that edit is the one thing this test exists to
stop. The board binds a session id to an agent label on first write and
never releases it, so a changed name mints a second identity, every post
after it is refused with ``TASK_IDENTITY_MISMATCH``, and the old label
cannot be unbound by anyone.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError

from lock_wake.identity import BRIDGE_AGENT, IDENTITY, load_task_id
from tests.conftest import CONFIGURED_ENV, TASK_ID, pin_env


class TestIdentity:
    def test_the_session_id_is_the_one_the_board_already_knows(self) -> None:
        """Pinned. See this module's docstring on why it is not derived."""
        assert IDENTITY["session_id"] == "1d211a4b-aef9-5d26-925a-ff4543550dd4"

    def test_the_label_is_kebab_case_and_service_shaped(self) -> None:
        assert IDENTITY["agent"] == BRIDGE_AGENT == "bridge-lock-wake-0909"

    def test_the_cwd_is_a_service_uri_rather_than_a_path(self) -> None:
        """A service has no directory a person could open, and recording one
        would put a path on the board that resolves nowhere."""
        assert IDENTITY["cwd"] == "service://lock-wake"


class TestLoadTaskId:
    def test_it_reads_the_standing_task_from_the_environment(self) -> None:
        pin_env(CONFIGURED_ENV)
        assert load_task_id() == TASK_ID

    def test_an_unset_variable_refuses_with_its_own_name(self) -> None:
        pin_env({})
        with pytest.raises(AppError, match="LOCK_WAKE_TASK_ID"):
            load_task_id()
