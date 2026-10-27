"""Exercise the actual metadata scanner against filesystem objects, never data rows."""

from __future__ import annotations

import errno
import os
import pathlib
import runpy
import sys

import pytest

from hpc3.core import _test_hooks, wrds_scan
from hpc3.core.wrds_catalog import probe_archive, read_catalog
from hpc3.core.wrds_scan import main, permission_error, scan


def test_real_tree_records_readability_sizes_aliases_and_dangling_links(
    tmp_path: pathlib.Path,
) -> None:
    current = tmp_path / "sasdata"
    current.mkdir()
    data = current / "annual.sas7bdat"
    data.write_bytes(b"not a real dataset")
    (tmp_path / "z_alias").symlink_to(current, target_is_directory=True)
    (current / "cycle").symlink_to(tmp_path, target_is_directory=True)
    (current / "missing").symlink_to(tmp_path / "absent")
    records = list(scan(tmp_path))
    assert ("file", str(data), 18, True, str(data)) in records
    assert ("alias", str(tmp_path / "z_alias"), 0, False, str(current)) in records
    assert ("alias", str(current / "cycle"), 0, False, str(tmp_path)) in records
    assert (
        "unresolved_link",
        str(current / "missing"),
        0,
        False,
        str(tmp_path / "absent"),
    ) in records
    assert sum(record[0] == "file" for record in records) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFO and effective permissions")
def test_real_denied_directory_file_and_special_entry(tmp_path: pathlib.Path) -> None:
    denied = tmp_path / "denied"
    denied.mkdir()
    denied.chmod(0)
    unreadable = tmp_path / "secret.sas7bdat"
    unreadable.write_bytes(b"do not open")
    unreadable.chmod(0)
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    records = list(scan(tmp_path))
    denied.chmod(0o700)
    unreadable.chmod(0o600)
    assert ("denied_directory", str(denied), 0, False, str(denied)) in records
    assert ("file", str(unreadable), 11, False, str(unreadable)) in records
    assert ("special", str(pipe), 0, False, str(pipe)) in records


def test_unexpected_traversal_errors_propagate() -> None:
    with pytest.raises(OSError, match="broken storage"):
        permission_error(OSError(errno.EIO, "broken storage", "/wrds/vendor"), [])
    with pytest.raises(OSError, match="no path"):
        permission_error(OSError(errno.EACCES, "no path"), [])


def test_missing_root_is_not_an_empty_success(tmp_path: pathlib.Path) -> None:
    with pytest.raises(NotADirectoryError):
        list(scan(tmp_path / "absent"))


def test_output_is_exclusive_and_completed(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "catalog.jsonl"
    main(["--root", str(root), "--output", str(output)])
    assert read_catalog(output.read_text())[0]["path"] == str(root)
    with pytest.raises(FileExistsError):
        main(["--root", str(root), "--output", str(output)])


def test_standalone_script_runs_without_package_dependencies(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / "catalog.jsonl"
    archive = tmp_path / "scan.pyz"
    archive.write_bytes(probe_archive())
    result = _test_hooks._default_run(
        [
            sys.executable,
            "-I",
            "-S",
            str(archive),
            "--root",
            str(root),
            "--output",
            str(output),
        ]
    )
    assert result["returncode"] == 0
    assert read_catalog(output.read_text())[0]["kind"] == "directory"
    failure = _test_hooks._default_run(
        [
            sys.executable,
            "-I",
            "-S",
            str(archive),
            "--root",
            str(root / "missing"),
            "--output",
            str(tmp_path / "failed.jsonl"),
        ]
    )
    assert failure["returncode"] != 0
    assert "NotADirectoryError" in failure["stderr"]
    assert (tmp_path / "failed.jsonl").read_text() == ""


def test_script_entrypoint_is_executed_in_coverage(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    original = sys.argv
    sys.argv = [wrds_scan.__file__, "--root", str(root), "--output", str(tmp_path / "out")]
    runpy.run_path(wrds_scan.__file__, run_name="__main__")
    sys.argv = original
    assert read_catalog((tmp_path / "out").read_text())[0]["path"] == str(root)
