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
from platform_core.error_codes_tooling import BoardBridgeErrorCode
from platform_core.errors import AppError

from ci_wake.identity import BRIDGE_AGENT, IDENTITY, TASK_ID_VARIABLE, load_task_id
from tests.conftest import CONFIGURED_ENV, TASK_ID, pin_env


class TestIdentity:
    def test_the_session_id_is_the_one_the_board_already_knows(self) -> None:
        """Pinned. See this module's docstring on why it is not derived."""
        assert IDENTITY["session_id"] == "f9b0de52-c89a-50a9-8230-c911afe86b58"

    def test_the_label_is_kebab_case_and_service_shaped(self) -> None:
        assert IDENTITY["agent"] == BRIDGE_AGENT == "bridge-ci-wake-0909"

    def test_the_cwd_is_a_service_uri_rather_than_a_path(self) -> None:
        """A service has no directory a person could open, and recording one
        would put a path on the board that resolves nowhere."""
        assert IDENTITY["cwd"] == "service://ci-wake"


class TestLoadTaskId:
    def test_it_reads_the_standing_task_from_the_environment(self) -> None:
        pin_env(CONFIGURED_ENV)

        assert load_task_id() == TASK_ID

    def test_an_unset_task_id_refuses_rather_than_being_guessed(self) -> None:
        """An announcement posted to a guessed task is an announcement
        nobody is subscribed to, which reads exactly like the bridge
        working."""
        pin_env({})

        with pytest.raises(AppError) as caught:
            load_task_id()

        assert caught.value.code is BoardBridgeErrorCode.TASK_ID_MISSING
        assert TASK_ID_VARIABLE in caught.value.message
