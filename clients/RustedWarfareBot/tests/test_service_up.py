"""Starting the match service's door and workers, detached from any terminal.

Driven against :class:`tests.harness_fakes.FakeHost` for the commands, the
port and the clock, with :class:`tests.test_fleet.FakeSpawner` standing in for
the detached spawn whose real implementation ``test_fleet_http`` runs.
"""

from __future__ import annotations

import runpy
from collections.abc import Generator, Sequence
from pathlib import Path

import pytest
from scripts.match_service import SERVICE_PORT_DEFAULT
from scripts.service_up import (
    CLONE_POOL,
    DOOR_LOG,
    DOOR_READY_SECONDS,
    FLEET_SETTLE_SECONDS,
    PASSWORD_COMMAND,
    PROCESS_TABLE_COMMAND,
    ServiceLaunchError,
    database_dsn,
    main,
    running_workers,
    start_door,
    start_fleet,
)

from rw_bot.harness import _test_hooks
from rw_bot.harness.play_match import EXIT_OK
from tests.harness_fakes import FakeHost
from tests.test_fleet import FakeMatch, FakeSpawner

DSN = "host=127.0.0.1 port=55432 user=covenant password=s3cret dbname=covenant connect_timeout=10"

#: Two workers as ``ps -eo args=`` shows them, and an unrelated python.
PROCESS_TABLE = (
    f"/venv/bin/python -u -m scripts.match_worker {DSN} creepw-5 {CLONE_POOL}",
    f"/venv/bin/python -u -m scripts.match_worker {DSN} creepw-2 {CLONE_POOL}",
    "/usr/bin/python3 -m http.server",
)


class DyingSpawner(FakeSpawner):
    """A spawner whose every child has already exited with status 1."""

    def __call__(self, argv: Sequence[str], transcript: Path) -> FakeMatch:
        """Record one spawn of a child that died at once.

        Args:
            argv: Argument vector, program first.
            transcript: Transcript file the child would write.

        Returns:
            A process double reporting exit status 1.
        """
        match = super().__call__(argv, transcript)
        match.returncode = 1
        return match


def _install(fake: FakeSpawner) -> Generator[FakeSpawner, None, None]:
    """Install a spawner for one test and restore the real one after.

    Args:
        fake: The spawner to install.

    Yields:
        The spawner.
    """
    original = _test_hooks.spawn_match
    _test_hooks.spawn_match = fake
    yield fake
    _test_hooks.spawn_match = original


@pytest.fixture()
def spawner() -> Generator[FakeSpawner, None, None]:
    """Install a recording detached spawn whose children live.

    Yields:
        The spawner.
    """
    yield from _install(FakeSpawner())


@pytest.fixture()
def dying_spawner() -> Generator[FakeSpawner, None, None]:
    """Install a recording detached spawn whose children die at once.

    Yields:
        The spawner.
    """
    yield from _install(DyingSpawner())


def _host() -> FakeHost:
    """Build a host whose container answers with a password.

    Returns:
        The host, with nothing yet listening on the door's port.
    """
    host = FakeHost()
    host.command_results["docker"] = (0, ("s3cret\n",))
    host.channel_opens = False
    return host


class TestTheDsn:
    def test_the_password_is_read_from_the_container(self) -> None:
        with _host() as host:
            assert database_dsn() == DSN
            assert host.commands == [PASSWORD_COMMAND]

    @pytest.mark.parametrize("answer", [(1, ("Error: No such container",)), (0, ("  ",))])
    def test_a_container_that_will_not_say_is_refused(
        self, answer: tuple[int, tuple[str, ...]]
    ) -> None:
        with _host() as host:
            host.command_results["docker"] = answer
            with pytest.raises(ServiceLaunchError) as caught:
                database_dsn()
        assert caught.value.code == "RW-SERVICE-001"
        assert caught.value.message.startswith(f"{' '.join(PASSWORD_COMMAND)} exited {answer[0]}")


class TestTheDoor:
    def test_it_starts_detached_and_waits_until_it_listens(self, spawner: FakeSpawner) -> None:
        with _host() as host:
            host.channel_opens_after = 3
            assert start_door() == EXIT_OK
            assert spawner.argvs == [["/venv/bin/python", "-u", "-m", "scripts.match_service", DSN]]
            assert spawner.transcripts == [Path(DOOR_LOG)]
            assert {port for port, _ in host.probed} == {SERVICE_PORT_DEFAULT}
            assert (
                f"door up: pid 4001, http://127.0.0.1:{SERVICE_PORT_DEFAULT}/, log {DOOR_LOG}"
                in host.printed
            )

    def test_a_door_already_up_is_refused_before_anything_starts(
        self, spawner: FakeSpawner
    ) -> None:
        with _host() as host:
            host.channel_opens = True
            with pytest.raises(ServiceLaunchError) as caught:
                start_door()
        assert caught.value.code == "RW-SERVICE-002"
        assert spawner.argvs == []

    def test_a_door_that_exits_is_reported_with_its_status(
        self, dying_spawner: FakeSpawner
    ) -> None:
        with _host() as host, pytest.raises(ServiceLaunchError) as caught:
            start_door()
        assert caught.value.code == "RW-SERVICE-003"
        assert "it exited 1" in caught.value.message
        assert host.slept == []

    def test_a_door_that_never_listens_is_reported_at_the_deadline(
        self, spawner: FakeSpawner
    ) -> None:
        with _host() as host, pytest.raises(ServiceLaunchError) as caught:
            start_door()
        assert caught.value.code == "RW-SERVICE-003"
        assert f"still not up after {DOOR_READY_SECONDS}s; see {DOOR_LOG}" in (caught.value.message)
        assert sum(host.slept) >= DOOR_READY_SECONDS


class TestTheFleet:
    def test_the_running_workers_are_read_off_their_command_lines(self) -> None:
        with _host() as host:
            host.command_results["ps"] = (0, PROCESS_TABLE)
            assert running_workers() == ("creepw-2", "creepw-5")
            assert host.commands == [PROCESS_TABLE_COMMAND]

    def test_only_the_missing_workers_are_started(self, spawner: FakeSpawner) -> None:
        """A running worker is left alone, so its match is not doubled."""
        with _host() as host:
            host.command_results["ps"] = (0, PROCESS_TABLE)
            assert start_fleet() == EXIT_OK
            names = [argv[-2] for argv in spawner.argvs]
            assert names == ["creepw-1", "creepw-3", "creepw-4", "creepw-6", "creepw-7", "creepw-8"]
            assert spawner.argvs[0] == [
                "/venv/bin/python",
                "-u",
                "-m",
                "scripts.match_worker",
                DSN,
                "creepw-1",
                CLONE_POOL,
            ]
            assert spawner.transcripts[0] == Path("runs/creepw-1.log")
            assert "creepw-2 already running; leaving it alone" in host.printed
            assert host.slept == [FLEET_SETTLE_SECONDS]
            assert "fleet up: started 6, running now: creepw-2, creepw-5" in host.printed

    def test_a_fleet_with_no_password_starts_nothing(self, spawner: FakeSpawner) -> None:
        with _host() as host:
            host.command_results["docker"] = (1, ())
            with pytest.raises(ServiceLaunchError):
                start_fleet()
        assert spawner.argvs == []


class TestTheCommandLine:
    def test_door_starts_the_door(self, spawner: FakeSpawner) -> None:
        with _host() as host:
            host.channel_opens_after = 2
            assert main(["door"]) == EXIT_OK
            assert len(spawner.argvs) == 1

    def test_fleet_up_starts_the_fleet(self, spawner: FakeSpawner) -> None:
        with _host() as host:
            host.command_results["ps"] = (0, ())
            assert main(["fleet-up"]) == EXIT_OK
            assert len(spawner.argvs) == 8

    @pytest.mark.parametrize("argv", [[], ["up"], ["door", "fleet-up"]])
    def test_anything_else_is_refused(self, argv: list[str]) -> None:
        with _host(), pytest.raises(ServiceLaunchError) as caught:
            main(argv)
        assert caught.value.code == "RW-SERVICE-004"

    def test_the_module_guard_runs_main_on_the_process_arguments(
        self, spawner: FakeSpawner
    ) -> None:
        with _host() as host:
            host.channel_opens_after = 2
            host.argv = ["door"]
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.service_up", run_name="__main__")
            assert caught.value.code == EXIT_OK
