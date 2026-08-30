"""The service hooks' real implementations, executed rather than trusted.

The delegates run against the harness's own in-memory host -- the same
FakeHost the sweep suite drives prepare/play through -- and the connector
proves its body by failing honestly against an address nothing listens on.
A hook whose real half is never executed is a seam that can rot unseen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.harness.match import decode_match_config
from rw_bot.harness.runner import decode_sweep_config
from rw_bot.harness.sweep import parse_job_line
from rw_bot.service._test_hooks import (
    _connect_impl,
    _play_job_impl,
    _prepare_clone_impl,
    _prepare_tree_impl,
    _read_card_impl,
)
from tests.harness_fakes import FakeHost

_CONFIG = decode_sweep_config(
    {
        "out_dir": "runs/sweeps/demo",
        "traces": "runs/traces",
        "workers": 1,
        "lockstep": 75,
        "clone_prefix": ".game-w",
        "source_game_dir": ".game",
        "tree": "runs/sweeps/demo/.tree",
        "pin_delta": 3,
        "fast_forward": 10,
    },
    decode_match_config(
        {"map_path": "maps/skirmish/[p2]duel_lake.tmx", "opponents": 1, "difficulty": 2}
    ),
)

_JOB = parse_job_line("alpha|12345|doctrines/flame-nocover.doctrine|400")


def test_the_real_delegates_drive_the_harness() -> None:
    """Tree, clone and match run through runner exactly as a sweep's do."""
    with FakeHost() as host:
        host.plant_source(".game")
        _prepare_tree_impl(_CONFIG)
        game_dir = _prepare_clone_impl(0, _CONFIG)
        assert game_dir == ".game-w1"
        played = _play_job_impl(_JOB, game_dir, _CONFIG)
        assert played is True
        assert any("alpha-s12345" in key for key in host.files)


def test_the_real_card_reader_reads_from_disk(tmp_path: Path) -> None:
    """The read body runs against a real filed card, encoding stated."""
    card = tmp_path / "alpha-s12345.txt"
    card.write_text("### alpha-s12345\nverdict        won (won)\n", encoding="utf-8")
    assert _read_card_impl(str(card)) == "### alpha-s12345\nverdict        won (won)\n"


def test_the_real_connector_fails_honestly_when_nothing_listens() -> None:
    """The connect body runs -- import, dial, and the driver's own refusal."""
    operational_error: type[Exception] = __import__("psycopg").OperationalError
    with pytest.raises(operational_error):
        _connect_impl("host=127.0.0.1 port=1 dbname=nothing connect_timeout=1")
