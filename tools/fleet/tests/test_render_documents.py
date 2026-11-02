"""``make render-documents``: one command writes every rendered JSON document.

The documents' content is each module's own suite
(``tests/test_hooks_check_document.py``, ``tests/test_toolchain_document.py``,
``tests/test_runner_host_document.py``); this holds the command to writing
all of them, exactly their renders, and to running when invoked as a module.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest

from fleet.cli import render_documents
from fleet.cli.runners import load_reserved_cores, load_runner_spec
from fleet.core.hooks_check_document import HOOKS_CHECK_DOCUMENT, render_hooks_check_document
from fleet.core.runner_host_document import RUNNER_HOSTS_DOCUMENT, render_runner_hosts_document
from fleet.core.toolchain_document import TOOLCHAIN_DOCUMENT, render_toolchain_document

#: The module the command runs as.
_MODULE = "fleet.cli.render_documents"

#: tools/fleet, whose roster and registry ``make render-documents`` names.
_PACKAGE = pathlib.Path(__file__).resolve().parent.parent

#: The runner roster and the registry, as ``make render-documents`` names them.
_SPEC = str(_PACKAGE / "runners.json")
_REGISTRY = str(_PACKAGE / "registry")


def _argv(out: pathlib.Path) -> list[str]:
    """The command's flags, writing into ``out``."""
    return ["--out", str(out), "--spec", _SPEC, "--config", _REGISTRY]


class TestTheCommand:
    def test_it_writes_every_document_into_the_directory_named(
        self, tmp_path: pathlib.Path
    ) -> None:
        assert render_documents.main(_argv(tmp_path)) == 0
        assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
            (HOOKS_CHECK_DOCUMENT, TOOLCHAIN_DOCUMENT, RUNNER_HOSTS_DOCUMENT)
        )
        hooks = (tmp_path / HOOKS_CHECK_DOCUMENT).read_text(encoding="utf-8")
        assert hooks == render_hooks_check_document()
        toolchain = (tmp_path / TOOLCHAIN_DOCUMENT).read_text(encoding="utf-8")
        assert toolchain == render_toolchain_document()
        spec = load_runner_spec(_SPEC)
        reserved = {
            host["name"]: load_reserved_cores(_REGISTRY, host["name"]) for host in spec["hosts"]
        }
        runners = (tmp_path / RUNNER_HOSTS_DOCUMENT).read_text(encoding="utf-8")
        assert runners == render_runner_hosts_document(spec, reserved)

    @pytest.mark.parametrize("missing", ["--out", "--spec", "--config"])
    def test_a_missing_flag_is_refused_naming_it(
        self, tmp_path: pathlib.Path, missing: str
    ) -> None:
        argv = _argv(tmp_path)
        at = argv.index(missing)
        with pytest.raises(ValueError, match=missing):
            render_documents.main([*argv[:at], *argv[at + 2 :]])
        assert list(tmp_path.iterdir()) == []

    def test_the_entrypoint_exits_with_mains_status(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["x", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                render_documents.entrypoint()
        finally:
            sys.argv = saved
        assert raised.value.code == 0
        assert (tmp_path / TOOLCHAIN_DOCUMENT).is_file()

    def test_running_as_a_module_actually_runs(self, tmp_path: pathlib.Path) -> None:
        saved_argv = sys.argv
        saved_module = sys.modules.pop(_MODULE, None)
        sys.argv = ["x", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(_MODULE, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[_MODULE] = saved_module
        assert raised.value.code == 0
        assert (tmp_path / RUNNER_HOSTS_DOCUMENT).is_file()
