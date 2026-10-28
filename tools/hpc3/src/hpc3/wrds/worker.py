"""Validate immutable declarations on the compute node, then run and attest.

No success artifact exists until execution, output checks and input/runtime
rechecks all pass. The worker never prints dataset rows or payload logs.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import sys
from collections.abc import Sequence

from platform_core.comparability import cpu_run_fingerprint
from platform_core.determinism_record import determinism_record
from platform_core.environment_record import package_versions
from platform_core.json_utils import (
    dump_json_str,
    load_json_bytes,
    load_json_str,
    narrow_json_to_list,
)
from platform_core.run_record import decode_observation, encode_run_record, run_record

from hpc3.contracts.wrds_extract import decode_crsp_selection
from hpc3.contracts.wrds_run import (
    WrdsRun,
    decode_wrds_run,
    encode_wrds_runtime,
    run_identity,
)
from hpc3.wrds import _test_hooks
from hpc3.wrds.extract import extract


def checked_file(path: pathlib.Path, expected: str) -> str:
    """Require exact file contents.

    Args:
        path: Declared code or input file.
        expected: Expected SHA256.

    Returns:
        Observed SHA256.

    Raises:
        ValueError: On a digest mismatch.
    """
    actual = _test_hooks.digest_file(path)
    if actual != expected:
        raise ValueError(f"WRDS_INPUT_DRIFT: {path}: expected {expected}, observed {actual}")
    return actual


def check_inputs(run: WrdsRun, directory: pathlib.Path) -> None:
    """Check every declared source and data file on the actual compute node.

    Args:
        run: Portable execution document.
        directory: Attempt directory.

    Raises:
        ValueError: On a changed input or a symlink escaping the code directory.
    """
    code = (directory / "code").resolve()
    for item in run["code"]:
        path = code / item["name"]
        if not path.resolve().is_relative_to(code):
            raise ValueError("WRDS_INPUT_DRIFT: code symlink escapes the staged snapshot")
        checked_file(path, item["sha256"])
    for item in run["inputs"]:
        checked_file(pathlib.Path(item["path"]), item["sha256"])


def check_runtime(run: WrdsRun) -> None:
    """Refuse interpreter or installed-version drift.

    Args:
        run: Expected environment.

    Raises:
        ValueError: When the actual runtime differs, including extra packages.
    """
    actual = _test_hooks.runtime()
    if actual != run["runtime"]:
        raise ValueError(
            "WRDS_RUNTIME_DRIFT: interpreter bytes/version or installed packages changed"
        )


def execute_run(document: pathlib.Path, attempt: str) -> None:
    """Run a validated experiment and write the shared result record last.

    Args:
        document: Portable run document in the attempt directory.
        attempt: Attempt identifier for provenance.

    Raises:
        ValueError: On drift, invalid metrics, an output escape or nonzero exit.
        TimeoutExpired: When the child exceeds its declared runtime.
    """
    directory = document.parent.resolve()
    run = decode_wrds_run(load_json_bytes(_test_hooks.read(document)))
    environment = _test_hooks.environment()
    job_id = environment["JOB_ID"]
    if not job_id.isdecimal():
        raise ValueError("WRDS_JOB_ID: execution requires a numeric scheduler job id")
    _test_hooks.write(
        directory / "started.json", dump_json_str({"job_id": job_id, "attempt": attempt}).encode()
    )
    check_inputs(run, directory)
    check_runtime(run)
    settings = {
        "PYTHONHASHSEED": str(run["seed"]),
        "WRDS_SEED": str(run["seed"]),
        "OMP_NUM_THREADS": str(run["slots"]),
        "OPENBLAS_NUM_THREADS": str(run["slots"]),
        "MKL_NUM_THREADS": str(run["slots"]),
        "NUMEXPR_NUM_THREADS": str(run["slots"]),
    }
    environment.update(settings)
    environment["PYTHONPATH"] = str(directory / "code")
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("IMAGE_DIGEST", None)
    environment["WRDS_OUTPUT_DIR"] = str(directory / "outputs")
    environment["WRDS_INPUTS_JSON"] = dump_json_str(
        {item["name"]: item["path"] for item in run["inputs"]}
    )
    argv = [
        run["runtime"]["python"],
        "-s",
        str(directory / "code" / run["entrypoint"]),
        *run["arguments"],
    ]
    exit_code = _test_hooks.execute(argv, directory, environment, run["seconds"])
    if exit_code != 0:
        raise ValueError(f"WRDS_PAYLOAD_FAILED: exit {exit_code}; inspect payload.err on WRDS")
    check_inputs(run, directory)
    check_runtime(run)
    outputs = directory / "outputs"
    digests: dict[str, str] = {}
    for name in run["outputs"]:
        path = outputs / name
        if not path.resolve().is_relative_to(outputs.resolve()):
            raise ValueError("WRDS_OUTPUT_ESCAPE: output symlink escapes the run directory")
        digests[name] = _test_hooks.digest_file(path)
    metrics = narrow_json_to_list(load_json_bytes(_test_hooks.read(outputs / "metrics.json")))
    observations = tuple(decode_observation(value) for value in metrics)
    if not observations:
        raise ValueError("WRDS_METRICS_EMPTY: at least one named observation is required")
    if any(not math.isfinite(item["value"]) for item in observations):
        raise ValueError("WRDS_METRICS_NONFINITE: observations must be finite")
    fingerprint = cpu_run_fingerprint(
        determinism_record("wrds-launcher", settings),
        environment.get,
        _test_hooks.host(),
        package_versions(run["runtime"]["packages"]),
    )
    payload_digest = hashlib.sha256(dump_json_str(digests).encode()).hexdigest()
    record = run_record(run["project"], run["name"], fingerprint, observations, payload_digest)
    result = {
        "identity": run_identity(run),
        "attempt": attempt,
        "job_id": job_id,
        "worker_sha256": _test_hooks.digest_file(directory / "worker.pyz"),
        "runtime": encode_wrds_runtime(run["runtime"]),
        "outputs": digests,
        "record": encode_run_record(record),
    }
    _test_hooks.write(directory / "completion.json", (dump_json_str(result) + "\n").encode())


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect this runtime or execute one attempt.

    Args:
        argv: Worker arguments.

    Returns:
        Zero after completion.

    Raises:
        ValueError: For missing or unknown arguments.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens == ["inspect"]:
        sys.stdout.write(dump_json_str(encode_wrds_runtime(_test_hooks.runtime())) + "\n")
    elif len(tokens) == 3 and tokens[0] == "run":
        execute_run(pathlib.Path(tokens[1]), tokens[2])
    elif len(tokens) == 2 and tokens[0] == "extract":
        extract(decode_crsp_selection(load_json_str(tokens[1])))
    else:
        raise ValueError(
            "WRDS_WORKER_ARGS: expected inspect, run DOCUMENT ATTEMPT or extract SCOPE"
        )
    return 0


__all__ = ["check_inputs", "check_runtime", "checked_file", "execute_run", "main"]
