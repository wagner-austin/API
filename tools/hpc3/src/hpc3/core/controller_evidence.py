"""Measure an installed controller against frozen wheels and its generated selfcheck."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

from platform_core.json_utils import JSONObject, require_dict, require_str

from hpc3.core import _test_hooks
from hpc3.core.digest import sha256_hex


def _verified_bytes(path: Path, expected: str) -> bytes:
    """Read one pinned file, refusing any digest difference."""
    payload = _test_hooks.read_bytes(path)
    if sha256_hex(payload) != expected:
        raise ValueError(f"CONTROLLER_DIGEST_MISMATCH: {path}")
    return payload


def _checked_process(python: Path, arguments: list[str], payload: bytes | None = None) -> str:
    """Run the isolated interpreter with a finite deadline and propagate failure."""
    result = _test_hooks.run([str(python), "-I", "-B", *arguments], stdin_bytes=payload)
    if result["returncode"] != 0:
        raise ValueError(
            f"CONTROLLER_CHECK_FAILED: {arguments}: exit {result['returncode']}: "
            + result["stderr"]
        )
    return result["stdout"].strip()


def _member_path(row: list[str], seen: set[str], names: list[str]) -> PurePosixPath:
    """Require a unique archive member with a safe, complete RECORD row."""
    if len(row) != 3:
        raise ValueError("CONTROLLER_WHEEL_RECORD: require path, hash and size")
    name = row[0]
    path = PurePosixPath(name)
    if (
        not name or path.is_absolute() or ".." in path.parts
        or "\\" in name or name in seen or name not in names
    ):
        raise ValueError("CONTROLLER_WHEEL_RECORD: unsafe, duplicate or absent member")
    return path


def _wheel_files(wheel: bytes, site: Path) -> int:
    """Compare every application member with its frozen RECORD and installed bytes."""
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        names = archive.namelist()
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1 or len(set(names)) != len(names):
            raise ValueError("CONTROLLER_WHEEL_RECORD: require one RECORD and unique members")
        seen: set[str] = set()
        count = 0
        for row in csv.reader(io.StringIO(archive.read(records[0]).decode("utf-8"))):
            path = _member_path(row, seen, names)
            name, digest, size = row
            seen.add(name)
            if name == records[0]:
                continue
            content = archive.read(name)
            actual = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
            if digest != "sha256=" + actual.decode("ascii") or size != str(len(content)):
                raise ValueError("CONTROLLER_WHEEL_RECORD: member hash or size mismatch")
            if path.parts[0].endswith(".dist-info"):
                continue
            if path.parts[0].endswith(".data"):
                raise ValueError("CONTROLLER_WHEEL_LAYOUT: relocated wheel data is unsupported")
            installed = site.joinpath(*path.parts)
            if not installed.resolve().is_relative_to(site.resolve()):
                raise ValueError("CONTROLLER_WHEEL_LAYOUT: installed member escapes site directory")
            _verified_bytes(installed, sha256_hex(content))
            count += 1
        if seen != set(names) or count == 0:
            raise ValueError("CONTROLLER_WHEEL_RECORD: incomplete or empty application RECORD")
        return count


def measure_controller(
    provenance: JSONObject, wheels_dir: Path, selfcheck: Path, selfcheck_sha256: str
) -> JSONObject:
    """Verify frozen application files and run checks with the installed interpreter.

    Args:
        provenance: Frozen runtime record naming source, interpreter and wheel hashes.
        wheels_dir: Local directory containing every wheel named in that record.
        selfcheck: Generated symbol/version check from the frozen image build.
        selfcheck_sha256: Independently pinned digest of that generated check.

    Returns:
        Measured counts, identities and successful check output, without data or keys.

    Raises:
        ValueError: Named CONTROLLER refusal for drift, malformed wheels or failed checks.
        OSError: Missing installed file, interpreter or wheel.
    """
    source = require_str(provenance, "source_commit")
    if re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("CONTROLLER_SOURCE: require a full frozen source commit")
    python = Path(require_str(provenance, "controller_python"))
    python_sha256 = require_str(provenance, "controller_python_sha256")
    _verified_bytes(python, python_sha256)
    check_bytes = _verified_bytes(selfcheck, selfcheck_sha256)
    wheels = require_dict(provenance, "first_party_wheel_sha256")
    if not wheels:
        raise ValueError("CONTROLLER_WHEELS: require frozen first-party wheels")
    site = Path(_checked_process(
        python, ["-c", "import sysconfig; print(sysconfig.get_path('purelib'))"]
    ))
    if not site.is_absolute() or not site.resolve().is_relative_to(python.parent.parent.resolve()):
        raise ValueError("CONTROLLER_SITE: interpreter site directory is outside the controller")
    verified: JSONObject = {}
    total = 0
    for name in wheels:
        if Path(name).name != name or not name.endswith(".whl"):
            raise ValueError("CONTROLLER_WHEELS: require wheel basenames")
        expected = require_str(wheels, name)
        wheel = _verified_bytes(wheels_dir / name, expected)
        count = _wheel_files(wheel, site)
        verified[name] = {"sha256": expected, "installed_files": count}
        total += count
    symbols = _checked_process(python, ["-"], check_bytes)
    dependencies = _checked_process(python, ["-m", "pip", "check"])
    if symbols != "image self-check OK" or dependencies != "No broken requirements found.":
        raise ValueError("CONTROLLER_CHECK_RECEIPT: missing symbol or dependency success")
    _verified_bytes(python, python_sha256)
    return {
        "kind": "HPC3_CONTROLLER_EVIDENCE", "source_commit": source,
        "controller_python": str(python), "controller_python_sha256": python_sha256,
        "site_packages": str(site), "verified_wheels": verified,
        "installed_files": total, "selfcheck_sha256": selfcheck_sha256,
        "selfcheck": symbols, "dependency_check": dependencies,
    }


__all__ = ["measure_controller"]
