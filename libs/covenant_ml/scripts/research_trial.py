"""Compute entrypoint for a frozen, admitted monthly validation trial.

Given ``--final-grant`` and ``--final-manifest`` together it consumes the
evaluator's released private grant; the authoritative journal stays off-node.
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
        SystemExit: Exit 2 when only one of ``--final-grant`` and ``--final-manifest``
            is given.
    """
    determinism = pin()
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint
    from covenant_ml.markets.research_final import execute_final
    from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
    from covenant_ml.markets.research_trial import execute_trial
    from covenant_ml.markets.research_trial_records import decode_research_trial
    from covenant_ml.markets.validation import read_document

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--access", type=Path)
    mode.add_argument("--final-grant", type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--final-manifest", type=Path)
    args = parser.parse_args(argv)
    request: Path = args.request
    output: Path = args.out
    grant: Path | None = args.final_grant
    final_manifest: Path | None = args.final_manifest
    scratch: Path = args.scratch
    if (grant is None) != (final_manifest is None):
        parser.error("--final-grant and --final-manifest are required together")
    trial = decode_research_trial(read_document(request))
    fingerprint = benchmark_fingerprint(determinism, config_test_hooks.get_env)
    code_digest = sha256(Path(__file__).read_bytes()).hexdigest()
    if grant is not None and final_manifest is not None:
        record = execute_final(
            trial,
            final_manifest,
            fingerprint,
            code_digest,
            output,
            grant_directory=grant,
            scratch=scratch,
        )
    else:
        access: Path = args.access
        keys = EvaluatorKeys(decode_research_access(read_document(access)))
        record = execute_trial(trial, fingerprint, code_digest, output, keys=keys, scratch=scratch)
    sys.stdout.write(f"manifest -> {output}\nrun record -> {record}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
