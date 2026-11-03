"""Run the real monthly research worker as a subprocess on sealed synthetic rows."""

import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, dump_json_str, require_dict
from scripts import research_trial

from ._crsp_fixtures import write_sources
from ._crsp_study_fixtures import study_protocol
from ._encrypted_fixtures import create_access, seal_extract
from ._trial_fixtures import frozen_request


class WorkerRun(NamedTuple):
    """The finished worker process and where it was told to publish."""

    process: subprocess.CompletedProcess[str]
    output: Path


def run_monthly_worker(root: Path, rows: dict[str, list[JSONObject]]) -> WorkerRun:
    """Seal ``rows`` for train-validation, delete the final key and run the worker.

    Args:
        root: Empty test directory holding sources, keys, scratch and result.
        rows: Synthetic relations written under the committed study design.

    Returns:
        The completed ``scripts.research_trial`` process and its output path.
    """
    source = root / "source"
    source.mkdir()
    manifest, _ = write_sources(source, rows, study_protocol())
    access = create_access(root)
    bundle = seal_extract(manifest, root / "encrypted", access, "train-validation")
    (root / "keys" / "final.agekey").unlink()
    evaluation: JSONObject = {
        "design": study_protocol(),
        "stopping_months": 6,
        "fold_months": 12,
        "confidence": 0.9,
    }
    document = frozen_request(evaluation, bundle, ["momentum", "liquidity"])
    require_dict(document, "policy")["code_digest"] = sha256(
        Path(research_trial.__file__).read_bytes()
    ).hexdigest()
    request = root / "request.json"
    request.write_text(dump_json_str(document))
    output = root / "result" / "evaluation.json"
    environment = dict(config_test_hooks.get_environment())
    environment["IMAGE_DIGEST"] = "d" * 64
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research_trial",
            "--request",
            str(request),
            "--out",
            str(output),
            "--access",
            str(root / "access.json"),
            "--scratch",
            str(root),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return WorkerRun(process, output)


__all__ = ["WorkerRun", "run_monthly_worker"]
