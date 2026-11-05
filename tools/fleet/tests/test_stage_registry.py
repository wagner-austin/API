"""The fleet registry staged beside a run, in as few round trips as it takes (fleet.core.staging).

Every launch stages the registry between its job's claim and its first
phase. Its two directories were made by a script apiece, two more ssh
connections of the 18 one launch on pendragon opened on 2026-11-05 (MCPs
board task 5280a25d), and are now made by one. The node is scripted
through the command hook, so what is asserted is each call in order and
what each script sends.
"""

from __future__ import annotations

import pathlib

from fleet.core import _test_hooks, dialect, names, staging
from tests.conftest import FakeRun, ok

TARGET = "/home/corvis/fleet/stage/libs-demo-lavender-1/libs-demo-lavender-1"


def _registry(tmp_path: pathlib.Path) -> pathlib.Path:
    """A registry of one settings file.

    Args:
        tmp_path: The case's directory.

    Returns:
        The registry directory.
    """
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "settings.json").write_text('{"node_poll_seconds": 5}\n', encoding="utf-8")
    return registry


def test_both_directories_are_made_by_one_script_before_the_archive_is_sent(
    tmp_path: pathlib.Path,
) -> None:
    registry = _registry(tmp_path)
    staged = names.staging_directory(TARGET)
    carried = f"{staged}/{names.REGISTRY_ARCHIVE_DIRECTORY}"
    landed = f"{staged}/{names.REGISTRY_DIRECTORY}"
    runner = FakeRun(
        [
            ok(""),  # send the mkdir script of both directories
            ok(""),  # run it
            ok(""),  # scp the archive
            ok(""),  # send the digest script
            ok(staging.digest(staging.registry_payload(registry))),  # run it
            ok(""),  # send the extract script
            ok(""),  # run it
        ]
    )
    _test_hooks.run = runner
    local = tmp_path / "archives" / "libs-demo-lavender-1-registry.tar.gz"

    extracted = staging.stage_registry("lavender", target=TARGET, registry=registry, local=local)

    assert extracted == landed
    script = dialect.script_path(staged, names.make_directory_stem(names.REGISTRY_DIRECTORY))
    assert runner.stdin[0] == dialect.checked_script((("mkdir", "-p", carried, landed),)).encode()
    assert runner.calls[0][-1] == dialect.write_command(script)
    assert runner.calls[1][-2:] == (*dialect.SH_INVOCATION, script)
    assert runner.calls[2][0] == "scp"
    assert runner.calls[2][-1] == f"lavender:{carried}/{names.ARCHIVE_NAME}"
    assert len(runner.calls) == 7
    assert not local.exists()
