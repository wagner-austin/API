"""The Windows retire's text (fleet.core.windows_retire, MCPs board task 74b13c20).

``tests/pester/rendered-dialect-state.Tests.ps1`` executes the committed
render, a tree past MAX_PATH and an inert ``cmd`` included; this pins what
that suite cannot see from outside: the directories go through ``rd`` on
the verbatim path through the parameter it can replace, and nothing in the
script calls ``Remove-Item -Recurse`` again.
"""

from __future__ import annotations

import pytest

from fleet.core.windows_retire import RETIRE_INCOMPLETE, VERBATIM_PREFIX_LITERAL, retire_script


def _script() -> str:
    """The retire for one example run."""
    return retire_script(
        target="C:/fleet/stage/run-1",
        retained="C:/fleet/stage/logs/run-1.log",
        scripts=("C:/fleet/stage/stop-run-1.ps1", "C:/fleet/stage/retire-run-1.ps1"),
        task="fleet-run-1",
    )


def test_each_directory_goes_through_rd_on_the_verbatim_path_and_is_looked_for_after() -> None:
    lines = _script().splitlines()

    verbatim = lines.index(
        f"        $verbatim = {VERBATIM_PREFIX_LITERAL} + [IO.Path]::GetFullPath($directory)"
    )
    assert VERBATIM_PREFIX_LITERAL == "'\\\\?\\'"
    assert lines[verbatim + 1] == "        & $Cmd /d /c rd /s /q $verbatim"
    assert lines[verbatim + 2] == "        if (Test-Path -LiteralPath $directory) {"
    assert lines[verbatim + 3] == (
        f'            throw "{RETIRE_INCOMPLETE}: rd exited $LASTEXITCODE and left $directory"'
    )
    assert "Remove-Item -Recurse" not in _script()


def test_cmd_is_a_parameter_naming_system32_s_by_default() -> None:
    assert '    [string]$Cmd = "$env:SystemRoot\\System32\\cmd.exe"' in _script().splitlines()


def test_a_path_that_cannot_be_embedded_is_refused() -> None:
    with pytest.raises(ValueError, match="target"):
        retire_script(target="C:/it's", retained="C:/r.log", scripts=(), task="t")
