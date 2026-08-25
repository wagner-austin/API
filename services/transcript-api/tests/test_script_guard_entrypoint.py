from __future__ import annotations

import logging
import runpy

import pytest


@pytest.mark.asyncio
async def test_guard_entrypoint_runs_as_main(monkeypatch: pytest.MonkeyPatch) -> None:
    # Running as a module must exit 0: the guard found no violations.
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.guard", run_name="__main__")
    assert exc.value.code == 0


logger = logging.getLogger(__name__)
