"""lavender is the Linux worker it was reimaged to, and its WSL guest is declared nowhere.

MCPs board task 1e9ab738. lavender was reimaged from Windows to Ubuntu
24.04.4 from a CORVIS stick on 2026-10-19. Its Windows lane and the WSL
guest the fleet called lavender-wsl went with the Windows install. A check
review of the task's closure
(8a49ff03) asked for a check that fails when the roster keeps either: a
lavender-wsl machine file left beside the others, a field of lavender's that
still describes the Windows machine, or a registry, runner or workflow value
that still sends work to the guest. Each case reads the shipped tree:

- the shipped registry decodes lavender as an enabled Linux machine run as
  corvis, and its own file holds the tailnet address the live policy names it
  at and the wired MAC a netboot arm boots it by, which only MCPs' decoder
  reads (``fleet.contracts.machine`` says why);
- no file under ``registry/machines`` is the guest's, and the decoded
  workspace names no such machine;
- every occurrence of the guest's name in a registry file, ``runners.json``
  or a workflow sits in prose (a ``notes`` or ``_comment`` value, a YAML
  comment), read with the retired-name scan's own lexer
  (``_retired_name_lexing.py``).

The last reading is first handed texts held in memory, so it is seen to find
a value and a ``runs-on`` line naming the guest and to pass the same name in
prose: a guard seen only on clean code has never been seen to work.
"""

from __future__ import annotations

import pathlib
from typing import Final

from platform_core.json_utils import require_dict, require_str

from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core.workspace_files import JSON_SUFFIX, MACHINES_DIRECTORY
from tests._host import ROSTER
from tests._registry import SHIPPED_REGISTRY, shipped_document
from tests._retired_name_lexing import lex, line_starts_of, within

#: The repository root: tools/fleet/tests is three levels below it.
ROOT: Final = pathlib.Path(__file__).resolve().parents[3]

#: The WSL guest lavender carried until its reimage.
GUEST: Final = "lavender-wsl"

#: lavender's tailnet address since the reimage, as the live policy names it.
TAILNET_IP: Final = "100.85.53.111"

#: The MAC of lavender's wired Realtek port, enp2s0, measured 2026-10-19.
BOOT_MAC: Final = "1c:69:7a:fe:78:9a"


def lines_naming(path: str, text: str, name: str) -> tuple[int, ...]:
    """The lines of a file that name a machine outside its prose.

    Args:
        path: The file's repository-relative path, which picks its lexer.
        text: Its text.
        name: The machine's name, matched in any case.

    Returns:
        The 1-based lines holding an occurrence no comment or prose span
        holds, in order, each once.
    """
    comments = lex(path, text).comments
    starts = line_starts_of(text)
    lowered = text.lower()
    needle = name.lower()
    found: list[int] = []
    at = lowered.find(needle)
    while at != -1:
        if not within(comments, at):
            line = sum(1 for start in starts if start <= at)
            if line not in found:
                found.append(line)
        at = lowered.find(needle, at + 1)
    return tuple(found)


def shipped_files() -> tuple[pathlib.Path, ...]:
    """The files that declare which machines exist and what runs on them.

    Returns:
        Every registry file, the runner roster and every workflow, sorted.
    """
    registry = SHIPPED_REGISTRY.rglob(f"*{JSON_SUFFIX}")
    workflows = (ROOT / ".github" / "workflows").glob("*.yml")
    return tuple(sorted((*registry, ROSTER, *workflows)))


def test_lavender_decodes_as_the_linux_worker_it_was_reimaged_to() -> None:
    document = shipped_document()
    lavender = decode_fleet_workspace(document)["machines"]["lavender"]
    declared = require_dict(require_dict(document, "machines"), "lavender")

    assert require_str(declared, "platform") == "linux"
    assert lavender["user"] == "corvis"
    assert lavender["enabled"] is True
    assert declared["guest"] is None
    assert require_str(declared, "tailnetIp") == TAILNET_IP
    assert require_str(declared, "bootMac") == BOOT_MAC


def test_no_machine_file_or_decoded_machine_is_the_guest() -> None:
    files = sorted(path.name for path in (SHIPPED_REGISTRY / MACHINES_DIRECTORY).iterdir())
    machines = decode_fleet_workspace(shipped_document())["machines"]

    assert f"lavender{JSON_SUFFIX}" in files
    assert f"{GUEST}{JSON_SUFFIX}" not in files
    assert sorted(machines) == sorted(name.removesuffix(JSON_SUFFIX) for name in files)
    assert GUEST not in machines


def test_the_registry_runners_and_workflows_name_the_guest_only_in_prose() -> None:
    files = shipped_files()
    texts = {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8") for path in files}
    naming = {
        path: lines for path, text in texts.items() if (lines := lines_naming(path, text, GUEST))
    }

    assert ROSTER in files
    assert SHIPPED_REGISTRY / MACHINES_DIRECTORY / f"lavender{JSON_SUFFIX}" in files
    assert ROOT / ".github" / "workflows" / "packages.yml" in files
    assert naming == {}


def test_a_value_naming_the_guest_is_found_and_its_notes_are_not() -> None:
    text = (
        '{\n  "notes": "replacing lavender-wsl, which no longer exists",\n'
        '  "guest": {"host": "lavender"},\n  "runner": "Lavender-WSL-2"\n}\n'
    )

    assert lines_naming("tools/fleet/registry/machines/x.json", text, GUEST) == (4,)


def test_a_runs_on_line_naming_the_guest_is_found_and_its_comment_is_not() -> None:
    text = (
        "jobs:\n  check:\n    # ran on lavender-wsl until 2026-10-19\n"
        "    runs-on: [self-hosted, lavender-wsl]  # lavender-wsl\n"
        "    steps:\n      - run: echo '# lavender-wsl'\n"
    )

    clean = "runs-on: [self-hosted, lavender]\n"

    assert lines_naming(".github/workflows/x.yml", text, GUEST) == (4, 6)
    assert lines_naming(".github/workflows/x.yml", clean, GUEST) == ()
