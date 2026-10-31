"""Extract and seal both CRSP partitions in a native allocation, or verify them afterwards.

``extract`` runs inside the scheduler allocation: plaintext stays in its
private scratch and only ciphertext, the receipt and metrics reach the output
directory named by an environment variable. ``verify`` runs on the evaluator
host against the collected directory and writes metadata to standard output;
it never prints a source row or key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.json_utils import dump_json_str

from covenant_ml.markets.crsp_extraction_records import decode_crsp_limits
from covenant_ml.markets.crsp_verify import verify_crsp_extraction
from covenant_ml.markets.crsp_writer import extract_crsp
from covenant_ml.markets.research_keys import EvaluatorKeys, decode_research_access
from covenant_ml.markets.validation import read_document


def _parser() -> argparse.ArgumentParser:
    """Declare both commands with every bound and path explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--protocol", required=True, type=Path)
    extract.add_argument("--access", required=True, type=Path)
    extract.add_argument("--schema", required=True)
    extract.add_argument("--setup", type=Path)
    extract.add_argument("--max-rows", required=True, type=int)
    extract.add_argument("--max-relation-bytes", required=True, type=int)
    extract.add_argument("--max-partition-bytes", required=True, type=int)
    extract.add_argument("--seconds", required=True, type=int)
    extract.add_argument("--scratch-env", required=True)
    extract.add_argument("--output-env", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--collected", required=True, type=Path)
    verify.add_argument("--access", required=True, type=Path)
    verify.add_argument("--scratch", required=True, type=Path)
    verify.add_argument("--open-final", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command.

    Args:
        argv: Explicit arguments, or the process arguments.

    Returns:
        Zero after a complete extraction or verification.

    Raises:
        KeyError: A named scheduler environment variable is absent.
        ValueError: Any writer or verifier refusal; nothing usable is published.
        OSError: Input, key or storage access fails.
    """
    args = _parser().parse_args(argv)
    command: str = args.command
    access: Path = args.access
    settings = decode_research_access(read_document(access))
    if command == "extract":
        environment = config_test_hooks.get_environment()
        scratch_variable: str = args.scratch_env
        output_variable: str = args.output_env
        protocol: Path = args.protocol
        schema: str = args.schema
        setup: Path | None = args.setup
        rows: int = args.max_rows
        relation_bytes: int = args.max_relation_bytes
        partition_bytes: int = args.max_partition_bytes
        seconds: int = args.seconds
        extract_crsp(
            protocol,
            settings,
            schema=schema,
            setup=setup,
            scratch=Path(environment[scratch_variable]),
            output=Path(environment[output_variable]),
            limits=decode_crsp_limits(
                {
                    "max_rows": rows,
                    "max_relation_bytes": relation_bytes,
                    "max_partition_bytes": partition_bytes,
                    "seconds": seconds,
                }
            ),
        )
        return 0
    collected: Path = args.collected
    scratch: Path = args.scratch
    open_final: bool = args.open_final
    result = verify_crsp_extraction(
        collected, EvaluatorKeys(settings), scratch=scratch, open_final=open_final
    )
    sys.stdout.write(dump_json_str(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
