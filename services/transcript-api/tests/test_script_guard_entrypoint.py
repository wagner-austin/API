from __future__ import annotations

import logging
import runpy

import pytest


@pytest.mark.asyncio
async def test_guard_entrypoint_runs_as_main(monkeypatch: pytest.MonkeyPatch) -> None:
    # Running as a module should exit with 0 or 2 depending on checks
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    err = exc.value
    code: int = err.code if isinstance(err.code, int) else 0
    assert code in (0, 2)


logger = logging.getLogger(__name__)
