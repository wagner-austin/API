"""The entry point loads each shipped head when its doctrine asks.

Split from the play tests when the second head arrived: both tests here
drive ``main`` end to end against the REAL committed model artifacts, so
each doubles as proof the shipped file still decodes
([[policy-exact-timing]] for the doom head,
[[impossible-step-three-design]] for the razing head).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.play import EXIT_INCOMPLETE, EXIT_OK, main

from rw_bot.policy.doctrine import DEFAULT_DOCTRINE, Doctrine
from rw_bot.policy.doctrine_file import format_doctrine
from tests.play_fixtures import (
    BUILDER as _BUILDER,
)
from tests.play_fixtures import (
    CATALOGUE_PATH as _CATALOGUE_PATH,
)
from tests.play_fixtures import (
    PLACEMENT_PATH as _PLACEMENT_PATH,
)
from tests.play_fixtures import (
    sample_lines as _sample_lines,
)
from tests.wire_fixtures import ScriptedPeer, StubbedConnect


def _preset(
    tmp_path: Path,
    name: str,
    *,
    counter: bool = False,
    navtilt: int = 0,
    brace: bool = False,
) -> Path:
    """Write a doctrine file: the default style plus the head under test."""
    doctrine = Doctrine(
        {
            **DEFAULT_DOCTRINE,
            "name": name,
            "goals": ("c_tank", "c_tank"),
            "mass": 7,
            "max_workers": 4,
            "reserve": -1,
            "counter": counter,
            "navtilt": navtilt,
            "brace": brace,
        }
    )
    path = tmp_path / f"{name}.doctrine"
    path.write_text("\n".join(format_doctrine(doctrine)) + "\n", encoding="utf-8")
    return path


def _run(preset: Path) -> int:
    peer = ScriptedPeer(_sample_lines(1, 9000, _BUILDER, (300, "landFactory")) * 3)
    with StubbedConnect(peer):
        return main(["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), "1", str(preset)])


def test_the_predicted_mode_loads_the_shipped_doom_model_and_plays(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """navtilt 3 end to end: the doctrine asks for prediction, main loads
    models/fleetdoom.ndjson -- the real shipped artifact, which this test
    also proves decodes -- and the loop runs with the latch fed every
    sample (log 2026-08-09, the replication verdict)."""
    code = _run(_preset(tmp_path, "seer", counter=True, navtilt=3))
    assert code in (EXIT_OK, EXIT_INCOMPLETE)
    assert capsys.readouterr().out.splitlines()[0] == "doctrine: seer"


def test_the_brace_flag_loads_the_shipped_razing_head_and_plays(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """brace 1 end to end: main loads models/razebrace.ndjson -- the
    fitted artifact, proven to decode here -- and the loop runs with the
    sliding latch fed every sample."""
    code = _run(_preset(tmp_path, "braced", brace=True))
    assert code in (EXIT_OK, EXIT_INCOMPLETE)
    assert capsys.readouterr().out.splitlines()[0] == "doctrine: braced"
