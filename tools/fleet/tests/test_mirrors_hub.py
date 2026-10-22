"""The hub runs the mirror-ref fetch, and its mirrors show it (MCPs board task 86f70457).

On 2026-10-22 diphtheria's project mirrors held no ref, so every export
received its repository's whole history, six at once for one MCPs commit,
and ten ``tmp_pack`` files (1005496 KiB) were left by fetches that died. These cases read the
hub itself: the tree its ``refs/fleet/rolled`` names, which is the code its
ticks run (:mod:`fleet.core.rolled`), and every mirror under its records.
Each mirror must hold the ref its fetches write, so the next fetch offers
that ref's history, and no ``tmp_pack`` older than the fetch turn's bound,
which :func:`fleet.core.mirror_fetch.remove_dead_packs` removes inside the
turn.

``tools/fleet-execution-hub`` runs them, pinned to the hub, with
``--fleet-hub`` naming the checkout the hub's ticks run with.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from fleet.cli import _config
from fleet.core import _test_hooks, export, mirror_fetch, rolled
from tests._git_fixtures import git
from tests._host import READING_PROPERTY


def _ref_of(mirror: pathlib.Path) -> str:
    """The ref a mirror's fetches write, by the kind of mirror its name says.

    Args:
        mirror: The mirror's path.

    Returns:
        :data:`~fleet.core.mirror_fetch.COMPANION_REF` for a companion's
        mirror, :data:`~fleet.core.mirror_fetch.FETCHED_REF` for a project's.
    """
    # export.companion_mirror_key's prefix, which the case below pins.
    companion = mirror.name.startswith("companion-")
    return mirror_fetch.COMPANION_REF if companion else mirror_fetch.FETCHED_REF


def test_a_mirror_s_ref_is_told_by_its_name(tmp_path: pathlib.Path) -> None:
    project = export.mirror_path(tmp_path, export.remote_mirror_key("git@github.com:o/MCPs.git"))
    companion = export.mirror_path(
        tmp_path, export.companion_mirror_key("https://github.com/o/MCPs.git")
    )

    assert (_ref_of(project), _ref_of(companion)) == (
        mirror_fetch.FETCHED_REF,
        mirror_fetch.COMPANION_REF,
    )


@pytest.mark.host_fleet_hub
def test_the_rolled_code_fetches_into_the_mirror_ref(
    fleet_hub: pathlib.Path, record_property: Callable[[str, str], None]
) -> None:
    rolled_sha = git(fleet_hub, "rev-parse", rolled.ROLLED_REF)
    module = git(
        fleet_hub, "show", f"{rolled.ROLLED_REF}:{rolled.FLEET_SRC}/fleet/core/mirror_fetch.py"
    )

    record_property(READING_PROPERTY, f"{rolled.ROLLED_REF} {rolled_sha}")
    assert f'FETCHED_REF: Final = "{mirror_fetch.FETCHED_REF}"' in module
    assert "file_turn.held(turn_lock(mirror)" in module


@pytest.mark.host_fleet_hub
def test_every_mirror_holds_its_ref_and_no_dead_fetch_s_pack(
    fleet_hub: pathlib.Path, record_property: Callable[[str, str], None]
) -> None:
    loaded = _config.load_workspace(
        {
            _config.CONFIG_FLAG: str(fleet_hub / rolled.REGISTRY_DIRECTORY),
            _config.RECORDS_FLAG: str(fleet_hub / rolled.RECORDS_DIR),
        }
    )
    readings: list[str] = []
    unreferenced: list[str] = []
    dead: list[str] = []
    mirrors = sorted(loaded.mirrors.glob("*.git"))
    for mirror in mirrors:
        ref = _ref_of(mirror)
        resolved = _test_hooks.run(
            ("git", "-C", str(mirror), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"),
            timeout_seconds=mirror_fetch.INIT_TIMEOUT_SECONDS,
        )
        held = resolved["stdout"].strip() if resolved["returncode"] == 0 else ""
        if not held:
            unreferenced.append(f"{mirror.name} holds no {ref}")
        packs = sorted((mirror / "objects" / "pack").glob("tmp_pack_*"))
        ages = {path.name: mirror_fetch._age(path) for path in packs}
        old = [
            name
            for name, age in ages.items()
            if age is not None and age >= mirror_fetch.FETCH_TURN["stale_seconds"]
        ]
        dead.extend(f"{mirror.name}: {name}" for name in old)
        readings.append(
            f"{mirror.name} {ref}={held[:12] or 'none'} tmp_pack={len(packs)} dead={len(old)}"
        )

    record_property(
        READING_PROPERTY, f"{len(mirrors)} mirror(s) in {loaded.mirrors}: " + "; ".join(readings)
    )
    assert mirrors != [], f"no mirror under {loaded.mirrors}"
    assert unreferenced == [], f"mirrors whose next fetch would offer no history: {unreferenced}"
    assert dead == [], f"tmp_pack files no live fetch is writing: {dead}"
