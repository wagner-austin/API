"""Tests for the fleet manager's control verbs, domain and HTTP.

``FleetManager.control`` writes a live bot's ``CONTROL`` file and
``POST /bots/{instance}/control`` is the operator's way to it. The
``CONTROL`` path is held in memory by :class:`_ControlFiles`; every
other path passes through to whatever seam the fixtures installed, so
the spawn records still land in the ``records`` store.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.bot.control import CONTROL_FILE_NAME, ControlVerb, make_control_command
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_page import FLEET_PAGE_HTML
from tankpit_bot.service.fleet_record import decode_process_record, process_record_path
from tests.service._fleet_fixtures import FakeRecordStore, _FakeSpawner


class _ControlFiles:
    """The ``CONTROL`` files, in memory; every other path passes through.

    Attributes:
        files: ``CONTROL`` path (as a string) to its content.
    """

    def __init__(self) -> None:
        """Wrap whatever ``replace_text`` and ``path_exists`` are installed now."""
        self.files: dict[str, str] = {}
        self._replace_text = top_hooks.replace_text
        self._path_exists = top_hooks.path_exists

    def replace_text(self, path: Path, content: str) -> None:
        """Hold a ``CONTROL`` write, pass any other through.

        Args:
            path: File path.
            content: File content.
        """
        if path.name != CONTROL_FILE_NAME:
            self._replace_text(path, content)
            return
        self.files[str(path)] = content

    def path_exists(self, path: Path) -> bool:
        """Answer for a ``CONTROL`` file, pass any other through.

        Args:
            path: Candidate path.

        Returns:
            Whether the file exists.
        """
        if path.name != CONTROL_FILE_NAME:
            return self._path_exists(path)
        return str(path) in self.files

    def install(self) -> None:
        """Point the two seams at this store."""
        top_hooks.replace_text = self.replace_text
        top_hooks.path_exists = self.path_exists

    def restore(self) -> None:
        """Put the two seams back."""
        top_hooks.replace_text = self._replace_text
        top_hooks.path_exists = self._path_exists


@pytest.fixture()
def control_files(spawner: _FakeSpawner) -> Generator[_ControlFiles, None, None]:
    """Install the in-memory ``CONTROL`` files over the spawn-record store.

    Yields:
        The store.
    """
    _ = spawner
    files = _ControlFiles()
    files.install()
    yield files
    files.restore()


_ALPHA_CONTROL = str(Path("runs/bot/alpha/CONTROL"))
_SPAWN_ALPHA: dict[str, str] = {"instance": "alpha"}


class TestManagerControl:
    """``FleetManager.control``."""

    def test_a_verb_is_written_beside_stop(
        self, spawner: _FakeSpawner, control_files: _ControlFiles
    ) -> None:
        """The child reads exactly what the manager wrote, and the row is returned."""
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)

        row = manager.control("alpha", make_control_command(ControlVerb.DISENGAGE, ""))

        assert control_files.files == {_ALPHA_CONTROL: '{"verb":"disengage","argument":""}'}
        assert row["instance"] == "alpha"
        assert row["alive"] is True
        assert spawner.envs[0]["TANKPIT_BOT_INSTANCE"] == "alpha"

    def test_a_second_verb_before_the_first_is_taken_is_refused(
        self, control_files: _ControlFiles
    ) -> None:
        """One verb at a time, so none is lost between two ticks."""
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        manager.control("alpha", make_control_command(ControlVerb.HOLD, "HUNT"))

        with pytest.raises(FleetError) as raised:
            manager.control("alpha", make_control_command(ControlVerb.RELEASE, ""))

        assert str(raised.value) == "instance 'alpha' has not taken its previous control verb yet"
        assert control_files.files == {_ALPHA_CONTROL: '{"verb":"hold","argument":"HUNT"}'}

    def test_an_unknown_or_finished_instance_is_refused(
        self, spawner: _FakeSpawner, control_files: _ControlFiles
    ) -> None:
        """A verb for a bot that cannot take it is an error, never a file."""
        manager = FleetManager()
        with pytest.raises(FleetError, match="unknown instance 'ghost'"):
            manager.control("ghost", make_control_command(ControlVerb.RELEASE, ""))

        manager.spawn(instance="alpha", account="", kills=0, seconds=0)
        spawner.processes[0].returncode = 0
        with pytest.raises(FleetError, match="instance 'alpha' is not running"):
            manager.control("alpha", make_control_command(ControlVerb.RELEASE, ""))
        assert control_files.files == {}

    def test_a_doctrine_verb_becomes_the_row_and_the_record(
        self, records: FakeRecordStore, control_files: _ControlFiles
    ) -> None:
        """The page, an adopting manager and a restart all see what now runs."""
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0)

        row = manager.control("alpha", make_control_command(ControlVerb.DOCTRINE, "passive"))

        assert row["doctrine"] == "passive"
        stored = records.files[str(process_record_path("alpha"))]
        assert (
            decode_process_record(narrow_json_to_dict(load_json_str(stored)))["doctrine"]
            == "passive"
        )
        assert control_files.files == {_ALPHA_CONTROL: '{"verb":"doctrine","argument":"passive"}'}

    def test_a_mode_verb_leaves_the_doctrine_alone(self, control_files: _ControlFiles) -> None:
        """Only the doctrine verb changes what the row reports."""
        manager = FleetManager()
        manager.spawn(instance="alpha", account="", kills=0, seconds=0, doctrine="duelist")

        row = manager.control("alpha", make_control_command(ControlVerb.WIND_DOWN, ""))

        assert row["doctrine"] == "duelist"
        assert control_files.files == {_ALPHA_CONTROL: '{"verb":"wind_down","argument":""}'}


class TestControlRoute:
    """``POST /bots/{instance}/control``."""

    @pytest.mark.asyncio
    async def test_a_valid_verb_answers_the_row(
        self,
        fleet_client: TestClient[web.Request, web.Application],
        control_files: _ControlFiles,
    ) -> None:
        """200 with the bot's row, and the file written."""
        created = await fleet_client.post("/bots", json=_SPAWN_ALPHA)
        assert created.status == 201

        answered = await fleet_client.post("/bots/alpha/control?verb=hold&argument=COLLECT")

        assert answered.status == 200
        assert narrow_json_to_dict(load_json_str(await answered.text()))["instance"] == "alpha"
        assert control_files.files == {_ALPHA_CONTROL: '{"verb":"hold","argument":"COLLECT"}'}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("query", "reason"),
        [
            ("verb=self_destruct", "verb"),
            ("", "verb"),
            ("verb=hold&argument=UNSET", "hold needs a mode to hold (HUNT, COLLECT), got 'UNSET'"),
            ("verb=release&argument=HUNT", "release takes no argument, got 'HUNT'"),
        ],
    )
    async def test_a_bad_request_is_a_400(
        self,
        fleet_client: TestClient[web.Request, web.Application],
        control_files: _ControlFiles,
        query: str,
        reason: str,
    ) -> None:
        """The decoder's refusal is the operator's answer, before any bot is asked."""
        answered = await fleet_client.post(f"/bots/alpha/control?{query}")

        assert answered.status == 400
        text = await answered.text()
        assert text.startswith("bad control request: ")
        assert reason in text
        assert control_files.files == {}

    @pytest.mark.asyncio
    async def test_an_unknown_instance_is_a_404(
        self,
        fleet_client: TestClient[web.Request, web.Application],
        control_files: _ControlFiles,
    ) -> None:
        """No such bot."""
        answered = await fleet_client.post("/bots/ghost/control?verb=release")

        assert answered.status == 404
        assert await answered.text() == "unknown instance 'ghost'"
        assert control_files.files == {}

    @pytest.mark.asyncio
    async def test_a_verb_not_yet_taken_is_a_409(
        self,
        fleet_client: TestClient[web.Request, web.Application],
        control_files: _ControlFiles,
    ) -> None:
        """The bot exists but cannot take another verb yet."""
        await fleet_client.post("/bots", json=_SPAWN_ALPHA)
        first = await fleet_client.post("/bots/alpha/control?verb=disengage")
        assert first.status == 200

        second = await fleet_client.post("/bots/alpha/control?verb=release")

        assert second.status == 409
        assert await second.text() == "instance 'alpha' has not taken its previous control verb yet"
        assert control_files.files == {_ALPHA_CONTROL: '{"verb":"disengage","argument":""}'}


def test_the_page_offers_every_verb() -> None:
    """Each verb has a button, and each button's path is one the route decodes."""
    for label, path in [
        ('"hold hunt"', '"/control?verb=hold&argument=HUNT"'),
        ('"hold collect"', '"/control?verb=hold&argument=COLLECT"'),
        ('"disengage"', '"/control?verb=disengage"'),
        ('"wind down"', '"/control?verb=wind_down"'),
        ('"release"', '"/control?verb=release"'),
    ]:
        assert f'[{label}, "POST", () => {path}, !bot.alive]' in FLEET_PAGE_HTML
    assert '["doctrine", "POST", chosenDoctrine, !bot.alive]' in FLEET_PAGE_HTML
    assert (
        '"/control?verb=doctrine&argument=" +\n'
        '    encodeURIComponent(document.getElementById("doctrine").value)'
    ) in FLEET_PAGE_HTML
