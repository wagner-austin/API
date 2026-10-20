"""The adopt pass at the tick level (MCPs board task 49d366ac): every netboot
tick, whether or not it claimed an arm, runs the installed release's
``fleet-netboot adopt`` with the key and logs what it printed, and nothing
the pass meets fails the tick.

The queue speaks the real wire shape and the commands go through the same
hook every dispatch test uses; the release and the env file are real files.
"""

from __future__ import annotations

import pathlib
from typing import Final

import pytest
from platform_core.json_utils import dump_json_str

from fleet.cli import agent
from fleet.core import _test_hooks, netboot_adopt, sealed_release
from tests._agent_cli import hub_host, netboot_argv
from tests._netboot_tick import (
    API_KEY,
    NOTHING_PENDING,
    RELEASE_ID,
    REVISION,
    adopt_calls,
    adopt_replies,
    place,
    verify_reply,
)
from tests._queue_fakes import FakeQueue, queue_env
from tests.conftest import FakeRun, failed

#: How every line the pass logs opens.
OPENING: Final = f"fleet-netboot adopt from sealed release {RELEASE_ID} (revision {REVISION})"

#: What ``adopt`` prints for pendragon's reinstall.
ADOPTED: Final = (
    "NETBOOT ADOPTED host=pendragon arm=ded2eb05 device=22 address=100.74.0.62 "
    "readdressed=true retired=11"
)


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """The queue's variables, and the machine the tick runs on."""
    _test_hooks.env = queue_env()
    _test_hooks.hostname = hub_host


def empty_tick(
    config_path: pathlib.Path,
    repo: pathlib.Path,
    paths: dict[str, pathlib.Path],
    runner: FakeRun,
    caplog: pytest.LogCaptureFixture,
) -> list[str]:
    """Run one netboot tick whose lane is empty.

    Args:
        config_path: The workspace document.
        repo: The monorepo root.
        paths: What :func:`tests._netboot_tick.place` laid out.
        runner: The scripted commands.
        caplog: The test's log capture.

    Returns:
        Every message the tick logged, after asserting it claimed once and
        exited 0.
    """
    _test_hooks.run = runner
    endpoint = FakeQueue([dump_json_str({"claimed": None})])
    _test_hooks.http_post = endpoint
    with caplog.at_level("INFO"):
        assert agent.main(netboot_argv(config_path, repo, paths)) == 0
    assert endpoint.tools == ["dispatch_claim"]
    return [record.getMessage() for record in caplog.records]


class TestTheAdoptPass:
    def test_an_empty_lane_still_adopts_and_runs_no_hub_pass(
        self,
        config_path: pathlib.Path,
        repo: pathlib.Path,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        paths = place(tmp_path)
        runner = FakeRun(adopt_replies(paths))

        logged = empty_tick(config_path, repo, paths, runner, caplog)

        assert runner.calls == adopt_calls(paths, config_path)
        assert runner.timeouts == [300, netboot_adopt.ADOPT_TIMEOUT_SECONDS]
        assert runner.set_env == [(), (("TAILSCALE_API_KEY", API_KEY),)]
        assert logged == ["netboot lane empty", f"{OPENING}: {NOTHING_PENDING}"]

    def test_each_adoption_is_logged_with_the_release_it_ran_from(
        self,
        config_path: pathlib.Path,
        repo: pathlib.Path,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        paths = place(tmp_path)
        printed = f"{ADOPTED}\n{ADOPTED.replace('pendragon', 'lenovoold')}"

        runner = FakeRun(adopt_replies(paths, printed))

        logged = empty_tick(config_path, repo, paths, runner, caplog)

        assert logged[1:] == [
            f"{OPENING}: {ADOPTED}",
            f"{OPENING}: {ADOPTED.replace('pendragon', 'lenovoold')}",
        ]

    def test_a_refused_adoption_is_logged_and_the_tick_still_exits_0(
        self,
        config_path: pathlib.Path,
        repo: pathlib.Path,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        paths = place(tmp_path)
        refusal = "fleet-netboot: NETBOOT_ADOPT_UNSEEN: no device of 9 holds 100.88.1.2"
        runner = FakeRun(
            [verify_reply(paths[agent.NETBOOT_RELEASE_FLAG]), failed(1, f"{refusal}\n")]
        )

        logged = empty_tick(config_path, repo, paths, runner, caplog)

        assert logged[1:] == [f"{OPENING} exited 1: {refusal}"]

    def test_a_release_failing_its_seal_runs_no_adopt_and_says_why(
        self,
        config_path: pathlib.Path,
        repo: pathlib.Path,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        paths = place(tmp_path)
        runner = FakeRun([failed(1, "RELEASE_DIGEST_MISMATCH: the payload changed")])

        logged = empty_tick(config_path, repo, paths, runner, caplog)

        assert runner.calls == adopt_calls(paths, config_path)[:1]
        assert len(logged) == 2
        assert logged[1].startswith(
            f"fleet-netboot adopt did not run: {sealed_release.UNSEALED_CODE}: "
        )
        assert "RELEASE_DIGEST_MISMATCH: the payload changed" in logged[1]
