"""Real local WRDS attempt fixtures and their metadata-only transport replies."""

from __future__ import annotations

import hashlib
import pathlib

from platform_core.json_utils import JSONObject, dump_json_str

from hpc3.contracts.wrds_run import (
    WrdsRun,
    decode_wrds_run,
    encode_wrds_runtime,
    execution_document,
)
from hpc3.wrds import _test_hooks

PAYLOAD = (
    "import os,json,pathlib,random\n"
    "rng=random.Random(int(os.environ['WRDS_SEED']))\n"
    "value=sum(rng.randrange(1000) for _ in range(100))\n"
    "root=pathlib.Path(os.environ['WRDS_OUTPUT_DIR'])\n"
    "(root/'metrics.json').write_text(json.dumps([{'name':'sum','value':float(value)}]))\n"
)


def example(directory: pathlib.Path, source: str = PAYLOAD) -> WrdsRun:
    """Build a real attempt directory whose sources and runtime are measured.

    Args:
        directory: Existing temporary directory.
        source: Payload program to execute.

    Returns:
        Validated run ready for the worker.
    """
    (directory / "code").mkdir()
    (directory / "outputs").mkdir()
    payload = source.encode()
    (directory / "code" / "main.py").write_bytes(payload)
    (directory / "worker.pyz").write_bytes(b"fixture worker identity")
    value: JSONObject = {
        "project": "wrds-system-probe",
        "name": "seeded-sum",
        "host": "wrds",
        "root": "/home/uci/awagner14/research-runs",
        "runtime": encode_wrds_runtime(_test_hooks._default_runtime()),
        "slots": 1,
        "memory_gb_per_slot": 2,
        "seconds": 30,
        "seed": 42,
        "code": [
            {
                "path": "code/main.py",
                "name": "main.py",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "inputs": [],
        "entrypoint": "main.py",
        "arguments": [],
        "outputs": ["metrics.json"],
    }
    run = decode_wrds_run(value)
    (directory / "run.json").write_text(dump_json_str(execution_document(run)))
    return run


__all__ = ["PAYLOAD", "example"]
