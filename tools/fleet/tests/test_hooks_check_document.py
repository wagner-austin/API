"""The hooks check pins' committed JSON: current and the contract's.

MCPs' session-host provision.sh and the fleet audit read the committed copy
(MCPs board task 88ca2402), so a pin moved in the contract without
``make render-documents`` would install and measure the old versions:
the first class fails until the copy is the render. The command itself is
``tests/test_render_documents.py``'s.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import load_json_str

from fleet.cli.render_documents import RENDERED_DIRECTORY
from fleet.contracts.tagged_tools import HOOKS_CHECK_PACKAGES
from fleet.core.hooks_check_document import HOOKS_CHECK_DOCUMENT, render_hooks_check_document

#: tools/fleet.
_PACKAGE = pathlib.Path(__file__).resolve().parent.parent

#: The committed copy.
_COMMITTED = _PACKAGE / RENDERED_DIRECTORY / HOOKS_CHECK_DOCUMENT


class TestTheCommittedCopy:
    def test_the_committed_copy_is_the_render(self) -> None:
        assert _COMMITTED.read_text(encoding="utf-8") == render_hooks_check_document(), (
            "run `make render-documents` in tools/fleet"
        )


class TestTheDocument:
    def test_it_names_every_pin_in_the_contracts_order_split_at_its_separator(self) -> None:
        pins = [
            {"name": "ruff", "version": "0.17.1"},
            {"name": "mypy", "version": "2.4.0"},
            {"name": "pytest", "version": "9.1.1"},
            {"name": "pytest-xdist", "version": "3.8.0"},
            {"name": "pytest-cov", "version": "7.1.0"},
            {"name": "coverage", "version": "7.16.2"},
            {"name": "pytest-timeout", "version": "2.4.0"},
        ]
        assert load_json_str(render_hooks_check_document()) == {"packages": pins}
        joined = tuple(f"{pin['name']}=={pin['version']}" for pin in pins)
        assert joined == HOOKS_CHECK_PACKAGES

    def test_it_ends_in_one_newline(self) -> None:
        text = render_hooks_check_document()
        assert text.endswith("}\n")
        assert not text.endswith("\n\n")
