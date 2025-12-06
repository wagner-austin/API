from __future__ import annotations

import runpy
import sys
from types import ModuleType

from pytest import MonkeyPatch


def test_worker_entry_runs_as_main(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://unit-test")
    called: dict[str, str] = {}

    # Provide a stub rq_harness so the script import binds our stub
    stub = ModuleType("platform_workers.rq_harness")

    def _run(cfg: dict[str, str]) -> None:
        called["queue"] = cfg["queue_name"]

    object.__setattr__(stub, "run_rq_worker", _run)
    # Provide a placeholder WorkerConfig type for the import
    object.__setattr__(stub, "WorkerConfig", dict)
    sys.modules["platform_workers.rq_harness"] = stub

    # Now run the module as a script
    runpy.run_module("music_wrapped_api.worker_entry", run_name="__main__")

    assert called.get("queue") == "music_wrapped"
