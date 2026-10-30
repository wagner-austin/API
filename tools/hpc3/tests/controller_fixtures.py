"""Frozen wheel and installed controller fixtures shared by evidence and CLI cases."""

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

from platform_core.json_utils import JSONObject

from hpc3.core.digest import sha256_hex
from tests.conftest import FakeRun

WHEEL = "example-1.0-py3-none-any.whl"
RECORD = "example-1.0.dist-info/RECORD"
CONTENT = b"value = 3\n"
SELFCHECK = b"print('image self-check OK')\n"


def write_wheel(root: Path, rows: list[list[str]] | None = None) -> str:
    """Write a wheel with a module, metadata and their original RECORD hashes."""
    members = {"example.py": CONTENT, "example-1.0.dist-info/METADATA": b"Name: example\n"}
    if rows is None:
        rows = []
        for name, data in members.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            rows.append([name, "sha256=" + digest.decode(), str(len(data))])
        rows.append([RECORD, "", ""])
    text = io.StringIO()
    csv.writer(text).writerows(rows)
    with zipfile.ZipFile(root / WHEEL, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr(RECORD, text.getvalue())
    return sha256_hex((root / WHEEL).read_bytes())


def prepare(root: Path, fake_run: FakeRun) -> JSONObject:
    """Prepare a frozen wheel, installed application and explicit command receipts."""
    (root / "bin").mkdir()
    (root / "bin/python").write_bytes(b"interpreter")
    (root / "site").mkdir()
    (root / "site/example.py").write_bytes(CONTENT)
    (root / "selfcheck.py").write_bytes(SELFCHECK)
    fake_run.add("sysconfig", stdout=str(root / "site"))
    fake_run.add("-", stdout="image self-check OK")
    fake_run.add("check", stdout="No broken requirements found.")
    return {
        "source_commit": "a" * 40, "controller_python": str(root / "bin/python"),
        "controller_python_sha256": sha256_hex(b"interpreter"),
        "first_party_wheel_sha256": {WHEEL: write_wheel(root)},
    }
