"""Compute entrypoint for a frozen, admitted monthly validation trial."""

from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Measure runtime and source, validate the request, and execute once.

    Args:
        argv: Explicit flags or process arguments.
        pin: Thread initializer, called before importing numerical libraries.

    Returns:
        Zero only after manifest and shared record publication.
    """
    determinism = pin()
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint
    from covenant_ml.markets.research_trial import execute_trial
    from covenant_ml.markets.research_trial_records import decode_research_trial
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    request: Path = args.request
    output: Path = args.out
    record = execute_trial(
        decode_research_trial(read_document(request)),
        benchmark_fingerprint(determinism, config_test_hooks.get_env),
        sha256(Path(__file__).read_bytes()).hexdigest(),
        output,
    )
    sys.stdout.write(f"manifest -> {output}\nrun record -> {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
