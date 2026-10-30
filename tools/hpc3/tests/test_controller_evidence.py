"""Installed controller measurements reject drift before claiming runtime success."""

import sys
import zipfile
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, require_dict

from hpc3.core import _test_hooks
from hpc3.core.controller_evidence import measure_controller
from hpc3.core.digest import sha256_hex
from tests.conftest import FakeRun
from tests.controller_fixtures import CONTENT, RECORD, SELFCHECK, WHEEL, prepare, write_wheel


def measure(root: Path, provenance: JSONObject) -> JSONObject:
    """Exercise the public measurement with frozen selfcheck identity."""
    return measure_controller(provenance, root, root / "selfcheck.py", sha256_hex(SELFCHECK))


def test_installed_application_and_command_receipts(tmp_path: Path, fake_run: FakeRun) -> None:
    provenance = prepare(tmp_path, fake_run)
    report = measure(tmp_path, provenance)
    assert report["installed_files"] == 1
    assert report["source_commit"] == "a" * 40
    assert report["controller_python"] == str(tmp_path / "bin/python")
    assert report["selfcheck"] == "image self-check OK"
    assert report["dependency_check"] == "No broken requirements found."
    assert report["verified_wheels"] == {
        WHEEL: {"sha256": require_dict(provenance, "first_party_wheel_sha256")[WHEEL],
                "installed_files": 1}
    }
    assert fake_run.calls[1].stdin_bytes == SELFCHECK
    assert all(call.argv[1:3] == ("-I", "-B") for call in fake_run.calls)


@pytest.mark.parametrize("path", ["bin/python", "selfcheck.py", WHEEL, "site/example.py"])
def test_changed_frozen_or_installed_file(tmp_path: Path, fake_run: FakeRun, path: str) -> None:
    provenance = prepare(tmp_path, fake_run)
    (tmp_path / path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="CONTROLLER_DIGEST_MISMATCH"):
        measure(tmp_path, provenance)


@pytest.mark.parametrize("value", ["", "../outside.whl", "not-a-wheel"])
def test_missing_or_unsafe_wheel_set(tmp_path: Path, fake_run: FakeRun, value: str) -> None:
    provenance = prepare(tmp_path, fake_run)
    provenance["first_party_wheel_sha256"] = {} if not value else {value: "a" * 64}
    with pytest.raises(ValueError, match="CONTROLLER_WHEELS"):
        measure(tmp_path, provenance)


def test_invalid_source(tmp_path: Path, fake_run: FakeRun) -> None:
    provenance = prepare(tmp_path, fake_run)
    provenance["source_commit"] = "main"
    with pytest.raises(ValueError, match="CONTROLLER_SOURCE"):
        measure(tmp_path, provenance)


@pytest.mark.parametrize("site", ["", "/outside"])
def test_interpreter_cannot_redirect_subject(tmp_path: Path, fake_run: FakeRun, site: str) -> None:
    fake_run.add("sysconfig", stdout=site)
    provenance = prepare(tmp_path, fake_run)
    with pytest.raises(ValueError, match="CONTROLLER_SITE"):
        measure(tmp_path, provenance)


@pytest.mark.parametrize("command", ["sysconfig", "-", "check"])
def test_nonzero_checks_propagate(tmp_path: Path, fake_run: FakeRun, command: str) -> None:
    fake_run.add(command, returncode=7, stderr="deliberate failure")
    provenance = prepare(tmp_path, fake_run)
    with pytest.raises(ValueError, match="CONTROLLER_CHECK_FAILED.*exit 7: deliberate failure"):
        measure(tmp_path, provenance)


@pytest.mark.parametrize("command", ["-", "check"])
def test_zero_exit_without_receipt_refused(tmp_path: Path, fake_run: FakeRun, command: str) -> None:
    fake_run.add(command, stdout="incomplete")
    provenance = prepare(tmp_path, fake_run)
    with pytest.raises(ValueError, match="CONTROLLER_CHECK_RECEIPT"):
        measure(tmp_path, provenance)


@pytest.mark.parametrize("rows", [
    [["wrong-width"]], [["", "", ""]], [["/absolute", "", ""]],
    [["../escape", "", ""]], [["back\\slash", "", ""]], [["absent", "", ""]],
    [[RECORD, "", ""], [RECORD, "", ""]],
    [["example.py", "sha256=incorrect", "10"]],
    [[RECORD, "", ""]], [],
])
def test_invalid_record_refused(tmp_path: Path, fake_run: FakeRun, rows: list[list[str]]) -> None:
    provenance = prepare(tmp_path, fake_run)
    provenance["first_party_wheel_sha256"] = {WHEEL: write_wheel(tmp_path, rows)}
    with pytest.raises(ValueError, match="CONTROLLER_WHEEL_RECORD"):
        measure(tmp_path, provenance)


@pytest.mark.parametrize("name", ["example.data/payload", "example.py"])
def test_unsupported_relocation_or_installed_symlink(
    tmp_path: Path, fake_run: FakeRun, name: str
) -> None:
    provenance = prepare(tmp_path, fake_run)
    if name.endswith("payload"):
        with zipfile.ZipFile(tmp_path / WHEEL) as archive:
            record = archive.read(RECORD).decode().replace("example.py", name)
            metadata = archive.read("example-1.0.dist-info/METADATA")
        with zipfile.ZipFile(tmp_path / WHEEL, "w") as archive:
            archive.writestr(name, CONTENT)
            archive.writestr("example-1.0.dist-info/METADATA", metadata)
            archive.writestr(RECORD, record)
        provenance["first_party_wheel_sha256"] = {
            WHEEL: sha256_hex((tmp_path / WHEEL).read_bytes())
        }
    else:
        (tmp_path / "site/example.py").unlink()
        (tmp_path / "outside.py").write_bytes(CONTENT)
        (tmp_path / "site/example.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(ValueError, match="CONTROLLER_WHEEL_LAYOUT"):
        measure(tmp_path, provenance)


def test_missing_record(tmp_path: Path, fake_run: FakeRun) -> None:
    provenance = prepare(tmp_path, fake_run)
    with zipfile.ZipFile(tmp_path / WHEEL, "w") as archive:
        archive.writestr("example.py", CONTENT)
    provenance["first_party_wheel_sha256"] = {WHEEL: sha256_hex((tmp_path / WHEEL).read_bytes())}
    with pytest.raises(ValueError, match="require one RECORD"):
        measure(tmp_path, provenance)


@pytest.mark.parametrize("shape", ["duplicate", "empty", "size"])
def test_duplicate_members_empty_application_or_wrong_size(
    tmp_path: Path, fake_run: FakeRun, shape: str
) -> None:
    provenance = prepare(tmp_path, fake_run)
    if shape == "duplicate":
        with zipfile.ZipFile(tmp_path / WHEEL, "a") as archive:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("example.py", CONTENT)
    elif shape == "empty":
        with zipfile.ZipFile(tmp_path / WHEEL, "w") as archive:
            archive.writestr(RECORD, RECORD + ",,\n")
    else:
        with zipfile.ZipFile(tmp_path / WHEEL) as archive:
            rows = archive.read(RECORD).decode().replace(",10\r\n", ",99\r\n")
            metadata = archive.read("example-1.0.dist-info/METADATA")
        with zipfile.ZipFile(tmp_path / WHEEL, "w") as archive:
            archive.writestr("example.py", CONTENT)
            archive.writestr("example-1.0.dist-info/METADATA", metadata)
            archive.writestr(RECORD, rows)
    provenance["first_party_wheel_sha256"] = {WHEEL: sha256_hex((tmp_path / WHEEL).read_bytes())}
    with pytest.raises(ValueError, match="CONTROLLER_WHEEL_RECORD"):
        measure(tmp_path, provenance)


def test_real_interpreter_success_and_failed_generated_check(tmp_path: Path) -> None:
    """Run a real venv and its real pip; failed selfcheck must prevent a report."""
    python = tmp_path / "bin/python"
    result = _test_hooks.run([sys.executable, "-m", "venv", str(tmp_path)])
    assert result["returncode"] == 0, result["stderr"]
    site_result = _test_hooks.run([
        str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"
    ])
    assert site_result["returncode"] == 0
    site = Path(site_result["stdout"].strip())
    (site / "example.py").write_bytes(CONTENT)
    check = tmp_path / "selfcheck.py"
    check.write_bytes(SELFCHECK)
    provenance: JSONObject = {
        "source_commit": "a" * 40, "controller_python": str(python),
        "controller_python_sha256": sha256_hex(python.read_bytes()),
        "first_party_wheel_sha256": {WHEEL: write_wheel(tmp_path)},
    }
    assert measure(tmp_path, provenance)["installed_files"] == 1
    failure = b"import sys; sys.stderr.write('real-selfcheck-failure'); sys.exit(7)\n"
    check.write_bytes(failure)
    with pytest.raises(ValueError, match="exit 7: real-selfcheck-failure"):
        measure_controller(provenance, tmp_path, check, sha256_hex(failure))
