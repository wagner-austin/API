"""The toolchain pins' committed JSON: current and the contract's (MCPs board task ce698bfb).

MCPs' host provisioning installs the Poetry this copy names on the session
host and on diphtheria, so a pin moved in the contract without
``make render-documents`` would install the old version: the first class
fails until the copy is the render.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import load_json_str

from fleet.contracts.toolchain import PINNED_POETRY
from fleet.core.rendered_powershell import RENDERED_DIRECTORY
from fleet.core.toolchain_document import TOOLCHAIN_DOCUMENT, render_toolchain_document

#: tools/fleet.
_PACKAGE = pathlib.Path(__file__).resolve().parent.parent

#: The committed copy.
_COMMITTED = _PACKAGE / RENDERED_DIRECTORY / TOOLCHAIN_DOCUMENT


class TestTheCommittedCopy:
    def test_the_committed_copy_is_the_render(self) -> None:
        assert _COMMITTED.read_text(encoding="utf-8") == render_toolchain_document(), (
            "run `make render-documents` in tools/fleet"
        )


class TestTheDocument:
    def test_it_names_poetry_at_the_version_every_committed_lock_was_written_by(self) -> None:
        assert load_json_str(render_toolchain_document()) == {"poetry": {"version": "2.4.2"}}
        assert PINNED_POETRY == "2.4.2"

    def test_it_ends_in_one_newline(self) -> None:
        text = render_toolchain_document()
        assert text.endswith("}\n")
        assert not text.endswith("\n\n")
