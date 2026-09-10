"""Measure quantized-histogram quality and fit time for both engines.

A DECLARATION AND A CONFIG, nothing else. What this benchmark IS lives in
:data:`covenant_ml.benchmarking.declarations.QUANTIZED`; running it, writing its
manifest and emitting its run record are
:mod:`covenant_ml.benchmarking.harness`'s job, for every family. This file
cannot forget to write a record, because writing is not something it does.
"""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread

# NOTHING FROM covenant_ml IS IMPORTED AT MODULE SCOPE, and that is a
# correctness requirement rather than a preference. `covenant_ml/__init__`
# pulls numpy, the BLAS thread variables are read when numpy loads, and a pin
# after that point writes variables nobody reads. `apply_cpu_determinism`
# refuses in that case instead of reporting a posture the run does not have.
# The harness is subject to the same rule and is imported below, after the pin.


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Run the quantized-histogram benchmark and write its manifest and record.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
        pin: How to pin CPU determinism, defaulting to the real pin. A test
            supplies a stand-in for one reason only: the real pin refuses once
            a native numeric library is loaded, and a numpy test suite has
            numpy loaded before collection begins. Substituting it does NOT
            excuse this module from being pinnable -- that property is
            asserted directly, by importing this file and checking nothing
            numeric arrived with it.

    Returns:
        Process exit code.
    """
    determinism = pin()

    from covenant_ml.benchmarking.declarations import QUANTIZED
    from covenant_ml.benchmarking.harness import (
        shared_parser,
        summary_lines,
        write_manifest_and_record,
    )
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint
    from covenant_ml.benchmarking.quantized_quality import (
        QuantizedBenchConfig,
        encode_quantized_manifest,
        run_quantized_benchmark,
    )

    parsed = shared_parser(QUANTIZED).parse_args(argv)

    # Annotated at assignment: argparse hands back Any, and the annotation is
    # what makes these typed rather than the value's shape at run time.
    seeds: list[int] = parsed.seeds
    out_path: Path = parsed.out
    config = QuantizedBenchConfig(
        n_samples=parsed.samples,
        n_features=parsed.features,
        n_estimators=parsed.trees,
        max_depth=parsed.max_depth,
        learning_rate=parsed.learning_rate,
        max_bins=parsed.max_bins,
        min_samples_leaf=parsed.min_samples_leaf,
        quant_bins=parsed.quant_bins,
    )

    # Read through the config layer, not os.environ. Writing a variable a
    # native library requires is a different act from reading configuration,
    # and only the first is this script's business.
    fingerprint = benchmark_fingerprint(determinism, config_test_hooks.get_env)
    manifest = run_quantized_benchmark(config, seeds, fingerprint)
    encoded = encode_quantized_manifest(manifest)
    sys.stdout.write("\n".join(summary_lines(QUANTIZED, encoded)) + "\n")
    record_path = write_manifest_and_record(QUANTIZED, encoded, fingerprint, out_path, seeds)
    sys.stdout.write(f"manifest -> {out_path}\nrun record -> {record_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
