"""Measure authenticated synthetic storage inside a native scheduler allocation.

This entrypoint produces metadata and scalar observations, never source rows or
key material. It exercises normal cleanup; it does not certify SIGKILL reclamation
or isolation from processes sharing the evaluator's OS identity.
"""

from __future__ import annotations

import argparse
import socket
import stat
from hashlib import sha256
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.run_record import Observation, encode_observation

from covenant_ml.markets.encrypted_storage import decrypted_bundle
from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
from covenant_ml.markets.validation import read_document


def main(argv: list[str] | None = None) -> int:
    """Authenticate the validation bundle and measure private storage and cleanup.

    Args:
        argv: Explicit flags or process arguments. Environment variable names
            identify scheduler scratch and the launcher's output directory.

    Returns:
        Zero after normal cleanup and exclusive metadata publication.

    Raises:
        KeyError: A required scheduler environment variable is absent.
        ValueError: Bundle authentication or private plaintext validation fails.
        OSError: Input access, cleanup or exclusive output creation fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--access", required=True, type=Path)
    parser.add_argument("--scratch-env", required=True)
    parser.add_argument("--output-env", required=True)
    args = parser.parse_args(argv)
    bundle: Path = args.bundle
    digest: str = args.sha256
    access: Path = args.access
    scratch_variable: str = args.scratch_env
    output_variable: str = args.output_env
    environment = config_test_hooks.get_environment()
    scratch = Path(environment[scratch_variable])
    output = Path(environment[output_variable])
    keys = EvaluatorKeys(decode_research_access(read_document(access)))
    with decrypted_bundle(
        bundle,
        digest,
        partition="train-validation",
        identity=keys.acquire("train-validation"),
        scratch=scratch,
        max_plaintext_bytes=keys.access["max_plaintext_bytes"],
    ) as manifest:
        private = manifest.parent
        mode = stat.S_IMODE(private.stat().st_mode)
        files = len(list(private.iterdir()))
        receipt: JSONObject = {
            "node": socket.gethostname(),
            "scratch": str(scratch),
            "scratch_mode": stat.S_IMODE(scratch.stat().st_mode),
            "private": str(private),
            "mode": mode,
            "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
            "files": files,
        }
    removed = not private.exists()
    receipt["plaintext_removed"] = removed
    observations = (
        Observation(name="authenticated_files", value=float(files)),
        Observation(name="private_mode", value=float(mode)),
        Observation(name="plaintext_removed", value=float(removed)),
    )
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str(receipt))
    with (output / "metrics.json").open("x", encoding="utf-8") as stream:
        stream.write(dump_json_str([encode_observation(item) for item in observations]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
