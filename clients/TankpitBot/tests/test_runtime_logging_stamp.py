"""The session_build provenance stamp (board task 7e766d65).

Every bot events artifact opens with one record stating what produced
the run — build ref, distribution version, instance, doctrine, room,
and deliberately never the account name. 539 archived runs cannot say
which build played them; every artifact written since 2026-09-04 can.
"""

from __future__ import annotations

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tests.conftest import FakeEnv, FakeFileSystem


def test_the_events_artifact_opens_with_its_own_provenance(
    fake_fs: FakeFileSystem,
) -> None:
    """The first record states what produced the run (board task 7e766d65).

    539 archived runs cannot say which build, doctrine or room played
    them; every artifact written since 2026-09-04 opens with a
    ``session_build`` stamp so the feature corpus can join rows to a
    real build instead of only a digest. The ACCOUNT NAME is
    deliberately absent — the tank_registry username-exposure decision
    is open, and a stamp here would widen it into every artifact.
    """
    artifacts = configure_bot_runtime_logging("20260331-230405")

    files = fake_fs.get_written_files()
    first = files[artifacts["latest_events_path"]].strip().splitlines()[0]
    decoded = narrow_json_to_dict(load_json_str(first))

    assert decoded["diagnostic_kind"] == "session_build"
    assert decoded["build_ref"] == "test-build-ref"
    version = decoded["distribution_version"]
    assert isinstance(version, str) and version != ""
    assert decoded["instance"] == ""
    assert decoded["doctrine"] == ""
    assert decoded["room"] == ""
    assert "account" not in decoded


def test_session_build_carries_the_spawn_environment(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """The fleet child's spawn axes ride the stamp.

    ``TANKPIT_BOT_INSTANCE``, ``TANKPIT_DOCTRINE`` and ``TANKPIT_ROOM``
    are exactly what the fleet manager states in a child's environment
    (``fleet_bot.py``), so a container bot's artifact names its own
    spawn configuration.
    """
    fake_env.set("TANKPIT_BOT_INSTANCE", "demo-1")
    fake_env.set("TANKPIT_DOCTRINE", "swarm")
    fake_env.set("TANKPIT_ROOM", "Practice")
    artifacts = configure_bot_runtime_logging("20260331-230405")

    files = fake_fs.get_written_files()
    first = files[artifacts["latest_events_path"]].strip().splitlines()[0]
    decoded = narrow_json_to_dict(load_json_str(first))

    assert decoded["diagnostic_kind"] == "session_build"
    assert decoded["instance"] == "demo-1"
    assert decoded["doctrine"] == "swarm"
    assert decoded["room"] == "Practice"
