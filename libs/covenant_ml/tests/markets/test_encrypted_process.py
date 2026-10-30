"""Real process termination distinguishes Python cleanup from scheduler reclamation."""

from __future__ import annotations

import signal
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_str

from ._trial_fixtures import trial_document

PROBE_PROGRAM = """
import os
import signal
import socket
import stat
import sys
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str
from covenant_ml.markets.encrypted_storage import decrypted_bundle
from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
from covenant_ml.markets.validation import read_document

bundle, digest, access, scratch, output, termination = sys.argv[1:]
keys = EvaluatorKeys(decode_research_access(read_document(Path(access))))
with decrypted_bundle(
    Path(bundle), digest, partition="train-validation",
    identity=keys.acquire("train-validation"), scratch=Path(scratch),
    max_plaintext_bytes=keys.access["max_plaintext_bytes"],
) as manifest:
    receipt: JSONObject = {
        "node": socket.gethostname(), "scratch": scratch,
        "private": str(manifest.parent),
        "mode": stat.S_IMODE(manifest.parent.stat().st_mode),
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "files": len(list(manifest.parent.iterdir())),
    }
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(receipt))
        stream.flush()
        os.fsync(stream.fileno())
    if termination == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.mark.parametrize("termination", ["normal", "kill"])
def test_real_process_exit_and_sigkill_have_distinct_cleanup(
    tmp_path: Path, termination: str
) -> None:
    """SIGKILL leaves private files for the scheduler; normal exit removes them."""
    document = trial_document(tmp_path)
    bundle = Path(require_str(document, "manifest"))
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    output = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE_PROGRAM,
            str(bundle),
            sha256(bundle.read_bytes()).hexdigest(),
            str(tmp_path / "access.json"),
            str(scratch),
            str(output),
            termination,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == (-signal.SIGKILL if termination == "kill" else 0)
    assert (result.stdout, result.stderr) == ("", "")
    receipt = narrow_json_to_dict(load_json_str(output.read_text()))
    assert receipt["mode"] == 0o700
    assert receipt["files"] == 10
    manifest = tmp_path / "source/manifest.json"
    assert receipt["manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
    private = Path(require_str(receipt, "private"))
    assert private.parent == scratch
    assert private.is_dir() == (termination == "kill")
    if termination == "kill":
        assert (private / "manifest.json").read_bytes() == manifest.read_bytes()
    else:
        assert list(scratch.iterdir()) == []
