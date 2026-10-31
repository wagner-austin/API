"""Run the maintained storage probe through real uncatchable termination."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_str

from .test_research_storage_probe import probe_arguments


def test_real_sigkill_leaves_plaintext_for_scheduler_reclamation(tmp_path: Path) -> None:
    """Kill after authentication; no Python cleanup or success receipt can run."""
    args = probe_arguments(tmp_path)
    args[-1] = "300"
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    output = tmp_path / "outputs"
    output.mkdir()
    environment = config_test_hooks.get_environment()
    environment.update(PROBE_SCRATCH=str(scratch), PROBE_OUTPUT=str(output))
    command = [sys.executable, "-m", "scripts.research_storage_probe", *args]
    opened = output / "opened.json"
    with subprocess.Popen(
        command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        try:
            deadline = time.monotonic() + 30
            while not (opened.exists() and opened.read_text().endswith("}")):
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.01)
        finally:
            process.kill()
            stdout, stderr = process.communicate(timeout=10)
        assert (process.returncode, stdout, stderr) == (-signal.SIGKILL, "", "")
    receipt = narrow_json_to_dict(load_json_str(opened.read_text()))
    assert receipt["mode"] == 0o700
    assert receipt["files"] == 10
    manifest = tmp_path / "source/manifest.json"
    assert receipt["manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
    private = Path(require_str(receipt, "private"))
    assert private.parent == scratch
    assert (private / "manifest.json").read_bytes() == manifest.read_bytes()
    assert sorted(path.name for path in output.iterdir()) == ["opened.json"]
