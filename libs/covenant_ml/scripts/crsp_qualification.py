"""Compute entrypoint for one frozen CRSP outcome qualification, before any fitting."""

from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Measure runtime and source, validate the request, and qualify once.

    Args:
        argv: Explicit flags or process arguments.
        pin: Thread initializer, called before importing numerical libraries.

    Returns:
        Zero only after manifest and shared record publication.
    """
    determinism = pin()
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint
    from covenant_ml.markets.crsp_qualification_run import (
        decode_crsp_qualification_request,
        execute_crsp_qualification,
    )
    from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--access", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args(argv)
    request_path: Path = args.request
    output: Path = args.out
    access: Path = args.access
    scratch: Path = args.scratch
    request = decode_crsp_qualification_request(read_document(request_path))
    record = execute_crsp_qualification(
        request,
        benchmark_fingerprint(determinism, config_test_hooks.get_env),
        sha256(Path(__file__).read_bytes()).hexdigest(),
        output,
        keys=EvaluatorKeys(decode_research_access(read_document(access))),
        scratch=scratch,
    )
    sys.stdout.write(f"manifest -> {output}\nrun record -> {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
