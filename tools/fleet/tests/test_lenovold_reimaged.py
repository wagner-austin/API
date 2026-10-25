"""lenovold is the Linux worker it was reimaged to, and its Windows name is gone from the roster.

MCPs board task 447efce2. The operator's Lenovo Yoga 3 14 was enrolled on
Windows 10 as lenovoold, the print host, run as austi. It was reimaged to
Ubuntu 24.04.4 from a CORVIS stick on 2026-10-20 and enrolled again as
lenovold at a new tailnet address. The task's criteria ask that the roster
hold lenovold's file as a Linux machine run as corvis at that address, and
that no file in the roster name lenovoold afterwards. Each case reads the
shipped tree:

- the shipped registry decodes lenovold as an enabled Linux machine run as
  corvis, and its own file holds the tailnet address the live policy names
  it at;
- no file under ``registry/machines`` is lenovoold's, and the decoded
  workspace names no such machine;
- no line of any registry file names lenovoold in any case, its notes
  included: unlike lavender's guest (``test_lavender_reimaged.py``), the
  criterion admits no prose.

The last reading is first handed a text held in memory, so it is seen to
find the name in a value and in notes: a guard seen only on clean code has
never been seen to work.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import require_dict, require_str

from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core.workspace_files import JSON_SUFFIX, MACHINES_DIRECTORY
from tests._registry import SHIPPED_REGISTRY, shipped_document

#: The name the laptop carried on Windows 10.
RETIRED: Final = "lenovoold"

#: lenovold's tailnet address since the reimage, as the live policy names it.
TAILNET_IP: Final = "100.127.247.38"


def lines_naming(text: str, name: str) -> tuple[int, ...]:
    """The lines of a text that name a machine, in any case.

    Args:
        text: The text.
        name: The machine's name.

    Returns:
        The 1-based lines holding the name, in order.
    """
    needle = name.lower()
    lines = enumerate(text.splitlines(), start=1)
    return tuple(number for number, line in lines if needle in line.lower())


def test_lenovold_decodes_as_the_linux_worker_it_was_reimaged_to() -> None:
    document = shipped_document()
    lenovold = decode_fleet_workspace(document)["machines"]["lenovold"]
    declared = require_dict(require_dict(document, "machines"), "lenovold")

    assert require_str(declared, "platform") == "linux"
    assert lenovold["user"] == "corvis"
    assert lenovold["enabled"] is True
    assert require_str(declared, "tailnetIp") == TAILNET_IP


def test_no_machine_file_or_decoded_machine_is_lenovoold() -> None:
    files = sorted(path.name for path in (SHIPPED_REGISTRY / MACHINES_DIRECTORY).iterdir())
    machines = decode_fleet_workspace(shipped_document())["machines"]

    assert f"lenovold{JSON_SUFFIX}" in files
    assert f"{RETIRED}{JSON_SUFFIX}" not in files
    assert RETIRED not in machines


def test_no_registry_file_names_lenovoold_even_in_its_notes() -> None:
    files = sorted(SHIPPED_REGISTRY.rglob(f"*{JSON_SUFFIX}"))
    naming = {
        path.relative_to(SHIPPED_REGISTRY).as_posix(): lines
        for path in files
        if (lines := lines_naming(path.read_text(encoding="utf-8"), RETIRED))
    }

    assert SHIPPED_REGISTRY / MACHINES_DIRECTORY / f"lenovold{JSON_SUFFIX}" in files
    assert naming == {}


def test_the_name_is_found_in_a_value_and_in_notes_in_any_case() -> None:
    text = (
        '{\n  "notes": "enrolled as LenovoOld on Windows",\n'
        '  "platform": "linux",\n  "host": "lenovoold"\n}\n'
    )

    assert lines_naming(text, RETIRED) == (2, 4)
    assert lines_naming('{\n  "host": "lenovold"\n}\n', RETIRED) == ()
