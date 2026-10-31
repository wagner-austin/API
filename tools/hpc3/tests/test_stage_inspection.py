"""Complete staged-tree readbacks execute real find and sha256sum commands."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.mcp_testing import sent_arguments
from platform_core.stage_manifest import StageManifest, encode_stage_manifest

from hpc3.cli import stage
from hpc3.core import _test_hooks
from hpc3.core._test_hooks import CommandResult
from hpc3.core.digest import sha256_hex
from hpc3.core.stage import certification_text, inspect_staged_tree
from tests.conftest import FakeRun, workspace_document, write_workspace
from tests.wrds_board import TASK, accepting_board


class LocalShell:
    """Execute the actual remote shell command against isolated local files."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def __call__(self, argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
        assert argv[0] == "ssh"
        self.commands.append(argv[-1])
        return _test_hooks._default_run(["sh", "-eu", "-c", argv[-1]], stdin_bytes=stdin_bytes)


def staged(root: Path, relative: str = "train-validation") -> StageManifest:
    """Write a real private tree and the existing stager's certification."""
    destination = root / relative
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for directory in [destination, *destination.parents]:
        if directory == root or root in directory.parents:
            directory.chmod(0o700)
    payload = b"encrypted fixture bytes"
    (destination / "sample.age").write_bytes(payload)
    manifest = StageManifest(
        destination=str(destination),
        files=[{"name": "sample.age", "size_bytes": len(payload), "sha256": sha256_hex(payload)}],
        provenance={"partition": "train-validation"},
    )
    (destination / "stage-digests.txt").write_text(certification_text(manifest))
    return manifest


@pytest.mark.parametrize("relative", [".", "train-validation", "nested/train-validation"])
def test_complete_private_tree_records_every_file(tmp_path: Path, relative: str) -> None:
    root = tmp_path / "study 'quoted'"
    manifest = staged(root, relative)
    transport = LocalShell()
    _test_hooks.run = transport
    report = inspect_staged_tree("hpc3", str(root), manifest, record_name="stage")
    assert report == {
        "root": str(root),
        "directories": {
            str(path.relative_to(root)) if path != root else "": 0o700
            for path in [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
        },
        "files": [
            {
                "name": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_hex(path.read_bytes()),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ],
    }
    assert len(transport.commands) == 4
    assert transport.commands[0].startswith("find -P ")
    assert transport.commands[0] == transport.commands[-1]


@pytest.mark.parametrize(
    "damage", ["extra", "missing", "link", "size", "content", "public", "root-link", "certificate"]
)
def test_real_tree_corruption_and_unexpected_keys_are_refused(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "study"
    manifest = staged(root)
    destination = Path(manifest["destination"])
    target = destination / "sample.age"
    if damage == "extra":
        (root / "final.agekey").write_text("not a real key")
    elif damage == "missing":
        target.unlink()
    elif damage == "link":
        target.unlink()
        target.symlink_to(tmp_path / "absent")
    elif damage == "size":
        target.write_bytes(b"truncated")
    elif damage == "content":
        target.write_bytes(b"X" * target.stat().st_size)
    elif damage == "public":
        destination.chmod(0o755)
    elif damage == "root-link":
        root.rename(tmp_path / "moved")
        root.symlink_to(tmp_path / "moved")
    else:
        certificate = destination / "stage-digests.txt"
        certificate.write_bytes(b"X" * certificate.stat().st_size)
    _test_hooks.run = LocalShell()
    with pytest.raises((ValueError, AppError), match=r"STAGE_INVENTORY|digests to"):
        inspect_staged_tree("hpc3", str(root), manifest, record_name="stage")


def test_real_traversal_failure_is_not_an_empty_success(tmp_path: Path) -> None:
    manifest = staged(tmp_path / "study")
    root = tmp_path / "absent"
    manifest["destination"] = str(root / "train-validation")
    _test_hooks.run = LocalShell()
    with pytest.raises(AppError, match="No such file or directory"):
        inspect_staged_tree("hpc3", str(root), manifest, record_name="stage")


@pytest.mark.parametrize("root", ["relative", "/study/../other", "/elsewhere"])
def test_scope_is_explicit_and_contains_destination(tmp_path: Path, root: str) -> None:
    with pytest.raises(ValueError, match="absolute study root"):
        inspect_staged_tree("hpc3", root, staged(tmp_path / "study"), record_name="stage")


@pytest.mark.parametrize("reply", ["", "f\t700\t1\tname", "bad\0", "d\t700\t1\t\0d\t700\t1\t\0"])
def test_incomplete_or_duplicate_traversal_is_refused(
    tmp_path: Path, fake_run: FakeRun, reply: str
) -> None:
    fake_run.add("find -P", stdout=reply)
    with pytest.raises(ValueError, match="STAGE_INVENTORY"):
        inspect_staged_tree("hpc3", str(tmp_path), staged(tmp_path), record_name="stage")


def test_certificate_cannot_replace_a_manifest_member(tmp_path: Path) -> None:
    manifest = staged(tmp_path)
    manifest["files"][0]["name"] = "stage-digests.txt"
    with pytest.raises(ValueError, match="certificate collides"):
        inspect_staged_tree("hpc3", str(tmp_path), manifest, record_name="stage")


def test_tree_change_during_hashing_is_not_published(tmp_path: Path) -> None:
    manifest = staged(tmp_path)
    transport = LocalShell()

    def changing(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
        result = transport(argv, stdin_bytes=stdin_bytes)
        if len(transport.commands) == 3:
            (tmp_path / "unexpected.agekey").write_text("fixture only")
        return result

    _test_hooks.run = changing
    with pytest.raises(ValueError, match="tree changed"):
        inspect_staged_tree("hpc3", str(tmp_path), manifest, record_name="stage")


def test_cli_readback_publishes_inventory_without_transferring(
    tmp_path: Path, emitted: list[str]
) -> None:
    root = tmp_path / "study"
    manifest = staged(root)
    manifest_path = tmp_path / "stage.json"
    manifest_path.write_text(dump_json_str(encode_stage_manifest(manifest)))
    expected = tmp_path / "expected.txt"
    expected.write_text(manifest["files"][0]["sha256"])
    transport = LocalShell()
    _test_hooks.run = transport
    board = accepting_board(2)
    flags = [
        "--config",
        write_workspace(tmp_path / "workspace.json", workspace_document()),
        "--manifest",
        str(manifest_path),
        "--expect-from",
        str(expected),
        "--inspect-root",
        str(root),
        "--report-task",
        TASK,
    ]
    assert stage.main(flags) == 0
    report = narrow_json_to_dict(load_json_str(emitted[-1]))
    assert report["kind"] == "HPC3_STAGE_INSPECTION"
    assert report["verified_files"] == 1
    assert report["inventory"] == inspect_staged_tree(
        "hpc3", str(root), manifest, record_name="stage"
    )
    assert report["manifest"] == encode_stage_manifest(manifest)
    assert sent_arguments(board.bodies[-1])["body"] == "HPC3 STAGE LOAD-REPORT\n" + emitted[-1]
    assert not any("cat >" in command or "mkdir" in command for command in transport.commands)
    with pytest.raises(ValueError, match="STAGE_INSPECTION"):
        stage.main([*flags, "--source-dir", str(root)])
    with pytest.raises(ValueError, match="STAGE_INSPECTION"):
        stage.main(flags[:-2])
