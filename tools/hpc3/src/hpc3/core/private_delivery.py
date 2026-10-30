"""Exclusive evaluator-private delivery using shared SSH and staging contracts."""

from __future__ import annotations

import shlex
from hashlib import sha256
from pathlib import Path

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.stage_manifest import StageManifest, decode_stage_manifest, encode_stage_manifest

from hpc3.core import _private_receive, _test_hooks
from hpc3.core.remote import MAX_COMMAND_CHARS, ssh_argv

PRIVATE_DELIVERY_MAX_BYTES = 1024 * 1024


def private_delivery_command(manifest: StageManifest) -> str:
    """Render the actual typed receiver with validated non-secret metadata only.

    Args:
        manifest: Existing staging contract, restricted to one MiB of private files.

    Returns:
        Bounded POSIX command using the target's Python 3 standard library.

    Raises:
        ValueError: PRIVATE_DELIVERY_LIMIT for an oversized transfer or command.
    """
    manifest = decode_stage_manifest(encode_stage_manifest(manifest))
    entries = tuple((f["name"], f["size_bytes"], f["sha256"]) for f in manifest["files"])
    if sum(length for _, length, _ in entries) > PRIVATE_DELIVERY_MAX_BYTES:
        raise ValueError("PRIVATE_DELIVERY_LIMIT: private transfer exceeds one MiB")
    program = (
        _test_hooks.read_bytes(Path(_private_receive.__file__)).decode("utf-8")
        + "\nimport sys\n"
        + f"receive_private({manifest['destination']!r}, {entries!r}, sys.stdin.buffer)\n"
        + "sys.stdout.write('delivered\\n')\n"
    )
    command = shlex.join(["python3", "-c", program])
    if len(command) > MAX_COMMAND_CHARS:
        raise ValueError("PRIVATE_DELIVERY_LIMIT: receiver command exceeds SSH command budget")
    return command


def deliver_private_bundle(host: str, manifest: StageManifest, payload: bytes) -> None:
    """Deliver once, checking bytes locally and at the exclusive private destination.

    Args:
        host: Existing SSH alias, including its configured VPN hop.
        manifest: Public staging identity; never place secret bytes in provenance.
        payload: Concatenated private file bytes in manifest order, sent on stdin.

    Raises:
        ValueError: PRIVATE_DELIVERY_BYTES for invalid local bytes.
        AppError: PRIVATE_DELIVERY_REMOTE for a failed or ambiguous remote delivery.
            No receiver output is exposed, and automatic retry is forbidden.
    """
    command = private_delivery_command(manifest)
    offset = 0
    for item in manifest["files"]:
        length = item["size_bytes"]
        if sha256(payload[offset : offset + length]).hexdigest() != item["sha256"]:
            raise ValueError("PRIVATE_DELIVERY_BYTES: local digest differs")
        offset += length
    if len(payload) != offset:
        raise ValueError("PRIVATE_DELIVERY_BYTES: local payload length differs")
    result = _test_hooks.run(ssh_argv(host, command), stdin_bytes=payload)
    if result["returncode"] != 0 or result["stdout"] != "delivered\n" or result["stderr"]:
        raise AppError(
            Hpc3ErrorCode.REMOTE_COMMAND_FAILED,
            f"PRIVATE_DELIVERY_REMOTE: receiver exit {result['returncode']}; "
            "delivery unverified, do not resend the grant",
        )


__all__ = ["PRIVATE_DELIVERY_MAX_BYTES", "deliver_private_bundle", "private_delivery_command"]
