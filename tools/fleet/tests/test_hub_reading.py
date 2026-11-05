"""The hub cases' workspace when the rolled commit is not the one under test.

MCPs board task 6b6caf74. Fleet job e62ad33d failed both hub cases at API
37f52927b because they read the registry refs/fleet/rolled named (0234136ea),
whose MCPs/fleet-nixos row declared worker_cores 8 without the
worker_cores_measurement 37f52927b's decoder requires. This case builds that
window for real: a hub checkout whose rolled commit carries this commit's
registry with that one field dropped, which this decoder refuses, and
:func:`tests._hub_reading.hub_workspace` must still load the registry beside
the code under test, with the checkout's records.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError

from fleet.cli import _config
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import _test_hooks, rolled
from tests._git_fixtures import rolled_at_home
from tests._hub_reading import hub_workspace
from tests._registry import SHIPPED_REGISTRY, shipped_document
from tests.conftest import FakeTempRoot

#: The project whose row the rolled commit carries without its measurement.
FLAKE = "MCPs/fleet-nixos"

#: The key the rolled row drops, one line of its file, as written.
MEASUREMENT_KEY = '"worker_cores_measurement":'

#: The sources ``git archive`` extracts beside the registry; any bytes do.
ROLLED_SOURCES: dict[str, str] = {
    "tools/fleet/src/fleet/cli/agent.py": "",
    "libs/platform_core/src/platform_core/__init__.py": "",
    "libs/monorepo_guards/src/monorepo_guards/__init__.py": "",
    "tools/board-watch/src/board_watch/__init__.py": "",
}


def _rolled_registry_without_the_measurement() -> dict[str, str]:
    """This commit's registry files, with the flake's measurement dropped.

    Returns:
        Repository-relative paths and their contents.
    """
    files: dict[str, str] = {}
    for path in sorted(SHIPPED_REGISTRY.rglob("*.json")):
        relative = f"{rolled.REGISTRY_DIRECTORY}/{path.relative_to(SHIPPED_REGISTRY).as_posix()}"
        files[relative] = path.read_text(encoding="utf-8")
    flake = f"{rolled.REGISTRY_DIRECTORY}/projects/{FLAKE}.json"
    lines = files[flake].splitlines(keepends=True)
    files[flake] = "".join(line for line in lines if MEASUREMENT_KEY not in line)
    assert len(files[flake].splitlines()) == len(lines) - 1
    return files


def test_a_rolled_registry_this_decoder_refuses_does_not_stop_the_hub_cases(
    tmp_path: pathlib.Path,
) -> None:
    api = tmp_path / "api"
    rolled_at_home(api, {**ROLLED_SOURCES, **_rolled_registry_without_the_measurement()})
    _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
    tree = rolled.extract_rolled_tree(api)
    assert not isinstance(tree, str), tree

    # API ea7e52c7b's hub_workspace, call for call: the rolled commit's
    # registry with the checkout's records, which this commit's decoder refuses.
    with pytest.raises(JSONTypeError, match="must carry worker_cores_measurement"):
        _config.load_workspace(
            {
                _config.CONFIG_FLAG: tree["config"],
                _config.RECORDS_FLAG: str(api / rolled.RECORDS_DIR),
            }
        )

    loaded = hub_workspace(api)

    assert loaded.workspace == decode_fleet_workspace(shipped_document())
    assert loaded.workspace["projects"][FLAKE]["worker_cores"] == 8
    assert loaded.ledger.is_relative_to((api / rolled.RECORDS_DIR).resolve())
