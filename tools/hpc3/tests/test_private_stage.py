"""The maintained private staging CLI uses bounded stdin and the real receiver."""

import subprocess
import sys
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str
from platform_core.process_input import process_input
from platform_core.stage_manifest import encode_stage_manifest

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.cli.private_stage import entrypoint, main
from hpc3.core import _test_hooks
from hpc3.core.private_delivery import PRIVATE_DELIVERY_MAX_BYTES, deliver_private_bundle
from tests.conftest import workspace_document
from tests.test_private_delivery import LocalReceiver, _manifest


def test_private_cli_uses_configured_transport_and_reports_no_secret(
    tmp_path: Path,
    emitted: list[str],
    argv: list[str],
) -> None:
    payload = b"synthetic-private-key\n"
    manifest = _manifest(tmp_path, payload)
    config = tmp_path / "workspace.json"
    config.write_text(dump_json_str(workspace_document()))
    document = tmp_path / "stage.json"
    document.write_text(dump_json_str(encode_stage_manifest(manifest)))
    receiver = LocalReceiver()
    _test_hooks.run = receiver
    limits: list[int] = []

    def read_input(limit: int) -> bytes:
        limits.append(limit)
        return payload

    cli_hooks.read_stdin = read_input
    args = ["--config", str(config), "--manifest", str(document)]
    assert main(args) == 0
    assert emitted == ["private delivery verified"]
    assert limits == [PRIVATE_DELIVERY_MAX_BYTES + 1]
    assert (Path(manifest["destination"]) / "key").read_bytes() == payload
    argv[:] = ["hpc3-private-stage", *args]
    with pytest.raises(SystemExit) as refused:
        entrypoint()
    assert refused.value.code == 2
    assert (Path(manifest["destination"]) / "key").read_bytes() == payload


def test_real_binary_input_is_bounded() -> None:
    with process_input(b"abcdef") as source:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from hpc3.cli import _test_hooks; import sys; "
                "sys.stdout.buffer.write(_test_hooks.read_stdin(3))",
            ],
            stdin=source,
            capture_output=True,
            timeout=10,
            check=False,
        )
    assert (completed.returncode, completed.stdout, completed.stderr) == (0, b"abc", b"")


def test_real_ssh_failure_never_reports_private_input(tmp_path: Path) -> None:
    payload = b"synthetic-private-key\n"
    with pytest.raises(AppError) as refused:
        deliver_private_bundle(
            "private-delivery-does-not-exist.invalid", _manifest(tmp_path), payload
        )
    assert refused.value.message == (
        "PRIVATE_DELIVERY_REMOTE: receiver exit 255; delivery unverified, do not resend the grant"
    )
    assert not (tmp_path / "delivery").exists()
