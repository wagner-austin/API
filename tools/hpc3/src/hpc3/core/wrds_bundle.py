"""Ship the same typed worker and shared provenance implementation to WRDS."""

from __future__ import annotations

import io
import pathlib
import zipfile

import typing_extensions
from platform_core import json_utils

from hpc3.contracts import wrds_run
from hpc3.core import _test_hooks


def worker_bundle() -> bytes:
    """Build a deterministic zipapp from the maintained source files.

    Returns:
        Worker bytes, including the existing serializers and RunRecord code.
        No installed hpc3 package is required on the compute node.
    """
    package = pathlib.Path(wrds_run.__file__).parents[1]
    shared = pathlib.Path(json_utils.__file__).parent
    members: dict[str, pathlib.Path] = {
        "typing_extensions.py": pathlib.Path(typing_extensions.__file__),
    }
    for name in (
        "json_utils",
        "run_record",
        "comparability",
        "environment_record",
        "determinism_record",
        "toml_utils",
        "config/_test_hooks",
    ):
        members[f"platform_core/{name}.py"] = shared / f"{name}.py"
    for name in ("contracts/pins", "contracts/wrds_run", "wrds/_test_hooks", "wrds/worker"):
        members[f"hpc3/{name}.py"] = package / f"{name}.py"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in (
            "platform_core",
            "platform_core/config",
            "hpc3",
            "hpc3/contracts",
            "hpc3/wrds",
        ):
            archive.writestr(zipfile.ZipInfo(f"{name}/__init__.py"), b"")
        for name, source in sorted(members.items()):
            archive.writestr(zipfile.ZipInfo(name), _test_hooks.read_bytes(source))
        archive.writestr(
            zipfile.ZipInfo("__main__.py"),
            b"from hpc3.wrds.worker import main\nraise SystemExit(main())\n",
        )
    return buffer.getvalue()


__all__ = ["worker_bundle"]
