"""CLI: place manifest-described files on the cluster, verified on both sides.

Usage:
    hpc3-stage --config hpc3.json --manifest runs/stage.json \\
        --source-dir runs/corpora --expect-from runs/file_ids.txt

Three checks, and they answer different questions. The local digest proves the
emitter produced the file the manifest names. The cluster-side digest proves
that file arrived intact. ``--expect-from`` proves the manifest names the bytes
the published work actually used -- which the first two cannot, because a
manifest emitted alongside its files always agrees with them.

``--expect-from`` is required, not optional. The failure it catches leaves no
trace: a corpus regenerated from the wrong source state stages clean, verifies
clean, trains to completion, and is comparable to nothing. A check that only
runs when someone remembers to ask for it is not protection against a failure
nobody notices.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str
from platform_core.load_report import post_load_report, report_identity
from platform_core.stage_manifest import decode_stage_manifest, encode_stage_manifest
from platform_core.stage_provenance import format_provenance

from hpc3.cli import _config, _test_hooks
from hpc3.core import _test_hooks as core_hooks
from hpc3.core import digest, remote
from hpc3.core.expected import check_expected, read_expected_digests
from hpc3.core.stage import (
    certification_path,
    certification_text,
    inspect_staged_tree,
    stage_manifest,
)

_FLAGS = (
    _config.CONFIG_FLAG,
    "--manifest",
    "--source-dir",
    "--expect-from",
    "--report-task",
    "--inspect-root",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Stage every file a manifest describes.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        Exit code 0 when every file was placed and verified on the cluster.

    Raises:
        ValueError: Missing flags, invalid report task or independent digest
            record changed during staging (STAGE_EXPECTED_CHANGED).
        JSONTypeError: If the manifest is not a valid stage manifest, or its
            provenance record is missing or empty.
        AppError: With ``STAGED_DIGEST_UNEXPECTED`` if a digest is absent from
            the published record, ``MANIFEST_FILE_MISSING`` if a file is
            absent, ``DIGEST_MISMATCH`` if bytes differ on either side, or
            ``REMOTE_COMMAND_FAILED`` if a remote command fails. Nothing is
            caught: a partial stage is not a success and must not be reported
            as one. With --report-task, certificate readback and board
            publication must also succeed before the command returns zero.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    task = parsed.get("--report-task")
    if task is not None:
        report_identity(task, "hpc")
    inspect_root = parsed.get("--inspect-root")
    if inspect_root is not None and (task is None or "--source-dir" in parsed):
        raise ValueError("STAGE_INSPECTION: require report-task and omit source-dir for readback")
    # The connection only: staging bytes needs the host and takes nothing
    # from the project registry, so it must not fail on a registry entry
    # that is mid-onboarding.
    host = _config.load_workspace_connection(parsed)["host"]
    manifest_path = pathlib.Path(cli_args.require_flag(parsed, "--manifest"))
    expect_path = pathlib.Path(cli_args.require_flag(parsed, "--expect-from"))
    expected_bytes = core_hooks.read_bytes(expect_path)

    raw = core_hooks.read_bytes(manifest_path).decode("utf-8")
    manifest = decode_stage_manifest(load_json_str(raw))

    # Identity before transport. Verifying that the wrong bytes arrived intact
    # costs a network round trip per file and answers the wrong question.
    check_expected(manifest, read_expected_digests(expect_path), source=expect_path)
    _test_hooks.emit(f"digests vouched for by {expect_path}")
    _test_hooks.emit(f"provenance {format_provenance(manifest['provenance'])}")

    inventory: JSONObject | None = None
    if inspect_root is None:
        source_dir = pathlib.Path(cli_args.require_flag(parsed, "--source-dir"))
        placed = stage_manifest(host, source_dir, manifest, record_name=manifest_path.stem)
        for remote_path in placed:
            _test_hooks.emit(f"staged {remote_path}")
    else:
        inventory = inspect_staged_tree(
            host, inspect_root, manifest, record_name=manifest_path.stem
        )
    _test_hooks.emit(
        f"verified {len(manifest['files'])} file(s) on {host}:{manifest['destination']}"
    )
    # Reported separately from the verified count, and after it, because this
    # is what admits the staged bytes to a training run: a corpus whose digest
    # no such record names is refused by
    # `model_trainer.cluster.preflight.check_corpus_certified`.
    _test_hooks.emit(f"certified by {certification_path(manifest, manifest_path.stem)}")
    if task is not None:
        certificate = certification_path(manifest, manifest_path.stem)
        certificate_digest = digest.sha256_hex(certification_text(manifest).encode("utf-8"))
        arrived = digest.parse_remote_digest(remote.remote_digest(host, certificate), certificate)
        digest.check_remote_digest(certificate, certificate_digest, arrived)
        if core_hooks.read_bytes(expect_path) != expected_bytes:
            raise ValueError(
                "STAGE_EXPECTED_CHANGED: independent digest record changed during staging"
            )
        report: JSONObject = {
            "kind": "HPC3_STAGE_EVIDENCE",
            "host": host,
            "manifest": encode_stage_manifest(manifest),
            "manifest_sha256": digest.sha256_hex(raw.encode("utf-8")),
            "expected_sha256": digest.sha256_hex(expected_bytes),
            "verified_files": len(manifest["files"]),
            "certification_path": certificate,
            "certification_sha256": certificate_digest,
            "controller_python": sys.executable,
            "controller_python_sha256": digest.sha256_hex(
                core_hooks.read_bytes(pathlib.Path(sys.executable))
            ),
        }
        if inventory is not None:
            report["kind"] = "HPC3_STAGE_INSPECTION"
            report["inventory"] = inventory
        payload = dump_json_str(report)
        post_load_report(
            post=_test_hooks.http_post,
            read_text=_config.stack_file_text,
            task_id=task,
            body="HPC3 STAGE LOAD-REPORT\n" + payload,
            family="hpc",
        )
        _test_hooks.emit(payload)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(run_cli(main, emit=_test_hooks.emit, emit_error=_test_hooks.emit_error))


__all__ = ["entrypoint", "main"]


if __name__ == "__main__":
    entrypoint()
