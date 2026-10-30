"""Compute entrypoint for a frozen, admitted monthly validation trial.

Given ``--ledger`` and ``--final-manifest`` together it instead runs the
evaluator's single final-period score of the journal's selected attempt.
"""

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

    Raises:
        SystemExit: Exit 2 when only one of ``--ledger`` and ``--final-manifest``
            is given.
    """
    determinism = pin()
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint
    from covenant_ml.markets.research_final import execute_final
    from covenant_ml.markets.research_trial import execute_trial
    from covenant_ml.markets.research_trial_records import decode_research_trial
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--final-manifest", type=Path)
    args = parser.parse_args(argv)
    request: Path = args.request
    output: Path = args.out
    ledger: Path | None = args.ledger
    final_manifest: Path | None = args.final_manifest
    if (ledger is None) != (final_manifest is None):
        parser.error("--ledger and --final-manifest are required together")
    trial = decode_research_trial(read_document(request))
    fingerprint = benchmark_fingerprint(determinism, config_test_hooks.get_env)
    code_digest = sha256(Path(__file__).read_bytes()).hexdigest()
    if ledger is not None and final_manifest is not None:
        record = execute_final(ledger, trial, final_manifest, fingerprint, code_digest, output)
    else:
        record = execute_trial(trial, fingerprint, code_digest, output)
    sys.stdout.write(f"manifest -> {output}\nrun record -> {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
