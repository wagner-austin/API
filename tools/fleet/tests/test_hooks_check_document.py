"""The hooks check pins' committed JSON: current, the contract's, and written by one command.

MCPs' session-host provision.sh and the fleet audit read the committed copy
(MCPs board task 88ca2402), so a pin moved in the contract without
``make render-hooks-check`` would install and measure the old versions:
the first class fails until the copy is the render.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.json_utils import load_json_str

from fleet.cli import render_hooks_check
from fleet.contracts.tagged_tools import HOOKS_CHECK_PACKAGES
from fleet.core.hooks_check_document import HOOKS_CHECK_DOCUMENT, render_hooks_check_document
from fleet.core.rendered_powershell import RENDERED_DIRECTORY

#: tools/fleet.
_PACKAGE = pathlib.Path(__file__).resolve().parent.parent

#: The committed copy.
_COMMITTED = _PACKAGE / RENDERED_DIRECTORY / HOOKS_CHECK_DOCUMENT


class TestTheCommittedCopy:
    def test_the_committed_copy_is_the_render(self) -> None:
        assert _COMMITTED.read_text(encoding="utf-8") == render_hooks_check_document(), (
            "run `make render-hooks-check` in tools/fleet"
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


class TestTheCommand:
    def test_it_writes_the_document_into_the_directory_named(self, tmp_path: pathlib.Path) -> None:
        assert render_hooks_check.main(["--out", str(tmp_path)]) == 0
        assert [path.name for path in tmp_path.iterdir()] == [HOOKS_CHECK_DOCUMENT]
        written = (tmp_path / HOOKS_CHECK_DOCUMENT).read_text(encoding="utf-8")
        assert written == render_hooks_check_document()

    def test_a_missing_out_flag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            render_hooks_check.main([])

    def test_the_entrypoint_exits_with_mains_status(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["x", "--out", str(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                render_hooks_check.entrypoint()
        finally:
            sys.argv = saved
        assert raised.value.code == 0
        assert (tmp_path / HOOKS_CHECK_DOCUMENT).is_file()

    def test_running_as_a_module_actually_runs(self, tmp_path: pathlib.Path) -> None:
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.render_hooks_check", None)
        sys.argv = ["x", "--out", str(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(
                    "fleet.cli.render_hooks_check", run_name="__main__", alter_sys=False
                )
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.render_hooks_check"] = saved_module
        assert raised.value.code == 0
        assert (tmp_path / HOOKS_CHECK_DOCUMENT).is_file()
