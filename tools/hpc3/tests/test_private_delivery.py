"""Private delivery checks real receiver effects and refuses secret-bearing replies."""

from __future__ import annotations

import io
import shlex
import stat
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.stage_manifest import StagedFile, StageManifest

from hpc3.core import _test_hooks
from hpc3.core._private_receive import receive_private, verify_private_file
from hpc3.core._test_hooks import CommandResult
from hpc3.core.private_delivery import deliver_private_bundle, private_delivery_command
from hpc3.core.remote import MAX_COMMAND_CHARS, SSH_OPTIONS
from tests.conftest import FakeRun


def _manifest(root: Path, payload: bytes = b"synthetic-private-key\n") -> StageManifest:
    root.chmod(0o700)
    return StageManifest(
        destination=str(root / "delivery"),
        files=[StagedFile(name="key", size_bytes=len(payload), sha256=sha256(payload).hexdigest())],
        provenance={"purpose": "synthetic-private-delivery"},
    )


def _entries(manifest: StageManifest) -> tuple[tuple[str, int, str], ...]:
    return tuple((item["name"], item["size_bytes"], item["sha256"]) for item in manifest["files"])


class LocalReceiver:
    """Run the exact SSH payload in a real subprocess; native SSH has a host twin."""

    def __init__(self) -> None:
        self.run = _test_hooks.run
        self.arguments: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
        self.arguments.append(tuple(argv))
        command = shlex.split(argv[-1])
        assert command[:2] == ["python3", "-c"]
        return self.run([sys.executable, *command[1:]], stdin_bytes=stdin_bytes)


def test_exact_receiver_runs_privately_and_refuses_a_second_delivery(tmp_path: Path) -> None:
    payload = b"synthetic-private-key\n"
    manifest = _manifest(tmp_path, payload)
    receiver = LocalReceiver()
    _test_hooks.run = receiver
    deliver_private_bundle("hpc3", manifest, payload)
    directory = Path(manifest["destination"])
    assert (directory / "key").read_bytes() == payload
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "key").stat().st_mode) == 0o600
    assert receiver.arguments[0][:-1] == ("ssh", *SSH_OPTIONS, "hpc3")
    assert payload.decode().strip() not in receiver.arguments[0][-1]
    assert len(receiver.arguments[0][-1]) <= MAX_COMMAND_CHARS
    with pytest.raises(AppError) as failed:
        deliver_private_bundle("hpc3", manifest, payload)
    assert failed.value.message == (
        "PRIVATE_DELIVERY_REMOTE: receiver exit 1; delivery unverified, do not resend the grant"
    )
    assert (directory / "key").read_bytes() == payload


def test_receiver_checks_all_files_before_creating_destination(tmp_path: Path) -> None:
    first, second = b"first", b"second"
    manifest = _manifest(tmp_path, first)
    manifest["files"].append(
        StagedFile(name="metadata", size_bytes=len(second), sha256=sha256(second).hexdigest())
    )
    receive_private(manifest["destination"], _entries(manifest), io.BytesIO(first + second))
    assert (Path(manifest["destination"]) / "metadata").read_bytes() == second


@pytest.mark.parametrize(
    "payload", [b"", b"wrong-same-size-data!\n", b"synthetic-private-key\nextra"]
)
def test_remote_corruption_never_creates_a_directory(tmp_path: Path, payload: bytes) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="PRIVATE_DELIVERY_BYTES"):
        receive_private(manifest["destination"], _entries(manifest), io.BytesIO(payload))
    assert not Path(manifest["destination"]).exists()


@pytest.mark.parametrize("kind", ["public", "symlink"])
def test_receiver_refuses_an_unsafe_parent(tmp_path: Path, kind: str) -> None:
    manifest = _manifest(tmp_path)
    if kind == "public":
        tmp_path.chmod(0o755)
    else:
        link = tmp_path / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        manifest["destination"] = str(link / "delivery")
    with pytest.raises(ValueError, match="PRIVATE_DELIVERY_PARENT"):
        receive_private(manifest["destination"], _entries(manifest), io.BytesIO(b""))
    assert not Path(manifest["destination"]).exists()


@pytest.mark.parametrize("kind", ["public", "symlink", "drift"])
def test_readback_refuses_changed_file_evidence(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "key"
    path.write_bytes(b"secret")
    path.chmod(0o600)
    if kind == "public":
        path.chmod(0o644)
    elif kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError, match="PRIVATE_DELIVERY_READBACK"):
        verify_private_file(path, sha256(b"different").hexdigest())


@pytest.mark.parametrize("payload", [b"different", b"synthetic-private-key\nextra"])
def test_local_byte_refusal_never_starts_transport(
    tmp_path: Path, fake_run: FakeRun, payload: bytes
) -> None:
    with pytest.raises(ValueError, match="PRIVATE_DELIVERY_BYTES"):
        deliver_private_bundle("hpc3", _manifest(tmp_path), payload)
    assert fake_run.calls == []


@pytest.mark.parametrize("kind", ["size", "command"])
def test_delivery_bounds_precede_transport(tmp_path: Path, kind: str) -> None:
    manifest = _manifest(tmp_path)
    if kind == "size":
        manifest["files"][0]["size_bytes"] = 1024 * 1024 + 1
    else:
        manifest["destination"] = "/" + "x" * MAX_COMMAND_CHARS
    with pytest.raises(ValueError, match="PRIVATE_DELIVERY_LIMIT"):
        private_delivery_command(manifest)


@pytest.mark.parametrize("kind", ["exit", "stdout", "stderr"])
def test_unverified_replies_do_not_disclose_receiver_output(
    tmp_path: Path, fake_run: FakeRun, kind: str
) -> None:
    fake_run.add(
        "python3",
        returncode=1 if kind == "exit" else 0,
        stdout="secret-on-stdout" if kind == "stdout" else "delivered\n",
        stderr="secret-on-stderr" if kind == "stderr" else "",
    )
    with pytest.raises(AppError) as failed:
        deliver_private_bundle("hpc3", _manifest(tmp_path), b"synthetic-private-key\n")
    assert failed.value.message == (
        f"PRIVATE_DELIVERY_REMOTE: receiver exit {1 if kind == 'exit' else 0}; "
        "delivery unverified, do not resend the grant"
    )
