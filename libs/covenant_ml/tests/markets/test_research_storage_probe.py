"""Native storage measurements use real encrypted files and real subprocesses."""

import runpy
import socket
import subprocess
import sys
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_str
from scripts import research_storage_probe

from ._trial_fixtures import trial_document


@pytest.fixture()
def allocation(tmp_path: Path) -> Generator[dict[str, str], None, None]:
    """Supply actual temporary directories through the existing environment seam."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    output = tmp_path / "outputs"
    output.mkdir()
    environment = config_test_hooks.get_environment()
    environment.update(PROBE_SCRATCH=str(scratch), PROBE_OUTPUT=str(output))
    previous = config_test_hooks.get_environment
    config_test_hooks.get_environment = lambda: environment
    yield environment
    config_test_hooks.get_environment = previous


def probe_arguments(root: Path) -> list[str]:
    """Create one real synthetic bundle and its exact CLI reference."""
    document = trial_document(root)
    bundle = Path(require_str(document, "manifest"))
    return [
        "--bundle",
        str(bundle),
        "--sha256",
        sha256(bundle.read_bytes()).hexdigest(),
        "--access",
        str(root / "access.json"),
        "--scratch-env",
        "PROBE_SCRATCH",
        "--output-env",
        "PROBE_OUTPUT",
    ]


def test_measurement_authenticates_files_and_confirms_cleanup(
    tmp_path: Path, allocation: dict[str, str]
) -> None:
    args = probe_arguments(tmp_path)
    assert research_storage_probe.main(args) == 0
    output = Path(allocation["PROBE_OUTPUT"])
    receipt = narrow_json_to_dict(load_json_str((output / "receipt.json").read_text()))
    private = Path(require_str(receipt, "private"))
    assert receipt == {
        "node": socket.gethostname(),
        "scratch": allocation["PROBE_SCRATCH"],
        "private": str(private),
        "mode": 0o700,
        "manifest_sha256": sha256((tmp_path / "source/manifest.json").read_bytes()).hexdigest(),
        "files": 10,
        "plaintext_removed": True,
    }
    assert private.parent == Path(allocation["PROBE_SCRATCH"])
    assert list(private.parent.iterdir()) == []
    assert load_json_str((output / "metrics.json").read_text()) == [
        {"name": "authenticated_files", "value": 10.0},
        {"name": "private_mode", "value": 448.0},
        {"name": "plaintext_removed", "value": 1.0},
    ]
    before = (output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        research_storage_probe.main(args)
    assert (output / "receipt.json").read_bytes() == before
    assert list(private.parent.iterdir()) == []


def test_real_payload_succeeds_then_refuses_changed_bundle(
    tmp_path: Path, allocation: dict[str, str]
) -> None:
    args = probe_arguments(tmp_path)
    command = [sys.executable, "-m", "scripts.research_storage_probe", *args]
    result = subprocess.run(
        command, env=allocation, capture_output=True, text=True, timeout=30, check=False
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    output = Path(allocation["PROBE_OUTPUT"])
    receipt = (output / "receipt.json").read_bytes()
    Path(args[1]).write_text("{}")
    failed = subprocess.run(
        command, env=allocation, capture_output=True, text=True, timeout=30, check=False
    )
    assert (failed.returncode, failed.stdout) == (1, "")
    assert failed.stderr.splitlines()[-1] == (
        "ValueError: ENCRYPTED_MANIFEST: bundle differs from admission"
    )
    assert (output / "receipt.json").read_bytes() == receipt
    assert list(Path(allocation["PROBE_SCRATCH"]).iterdir()) == []


def test_entrypoint_requires_explicit_scheduler_paths() -> None:
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("scripts.research_storage_probe", run_name="__main__")
    assert stopped.value.code == 2
