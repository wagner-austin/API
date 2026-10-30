"""Private frozen inputs and bounded invocations of maintained research CLIs.

The queue carries no local paths. Its digest resolves below a tenant-specific,
operator-provisioned directory. Referenced input bytes are verified on every
poll; mutable journals and native receipts remain owned by their controllers.
"""

from __future__ import annotations

import hashlib
import pathlib

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_str,
)

from fleet.contracts.research import ResearchRequest, require_digest
from fleet.core import _test_hooks


def read_pinned(root: pathlib.Path, digest: str, name: str) -> JSONObject:
    """Read one content-addressed private document and require its exact bytes.

    Args:
        root: Operator-provisioned specification directory for this tenant.
        digest: Validated SHA-256 naming the document's directory and bytes.
        name: Document file name inside that directory.

    Returns:
        The decoded document.

    Raises:
        ValueError: RESEARCH_SPEC_DRIFT when the bytes no longer match the digest.
        OSError: Missing or unreadable specification; no native operation occurs.
    """
    payload = _test_hooks.read_bytes(root / digest / name)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("RESEARCH_SPEC_DRIFT: private specification bytes changed")
    return narrow_json_to_dict(load_json_str(payload.decode()))


def read_specification(root: pathlib.Path, request: ResearchRequest, tenant: str) -> JSONObject:
    """Resolve exact private bytes and enforce tenant, backend and attempt binding.

    Args:
        root: Operator-provisioned specification directory for this tenant.
        request: Validated queue reference.
        tenant: Authenticated queue tenant, never read from a submitted payload.

    Returns:
        The pinned document for the backend's own decoder.

    Raises:
        ValueError: RESEARCH_SPEC_DRIFT or RESEARCH_SPEC_OWNER on a mismatch.
        OSError: Missing or unreadable specification; no native operation occurs.
    """
    document = read_pinned(root, request["spec_sha256"], "request.json")
    if (
        require_str(document, "tenant") != tenant
        or require_str(document, "backend") != request["backend"].value
        or require_str(document, "attempt") != request["attempt"]
    ):
        raise ValueError("RESEARCH_SPEC_OWNER: tenant, backend or attempt differs from the queue")
    return document


def require_path(document: JSONObject, key: str) -> pathlib.Path:
    """Require an explicit absolute controller path in a private specification.

    Args:
        document: Verified private specification.
        key: Path field.

    Returns:
        Absolute path with no implicit working-directory dependency.

    Raises:
        ValueError: RESEARCH_SPEC_PATH for a relative path.
    """
    path = pathlib.Path(require_str(document, key))
    if not path.is_absolute():
        raise ValueError(f"RESEARCH_SPEC_PATH: {key} must be absolute")
    return path


def verified_input(document: JSONObject, key: str) -> pathlib.Path:
    """Check one named input before the maintained controller reads it.

    Args:
        document: Verified private specification.
        key: Object containing path and sha256.

    Returns:
        Verified absolute input path.

    Raises:
        ValueError: RESEARCH_INPUT_FIELDS or RESEARCH_INPUT_DRIFT.
        OSError: Missing or unreadable input.
    """
    item = require_dict(document, key)
    if set(item) != {"path", "sha256"}:
        raise ValueError(f"RESEARCH_INPUT_FIELDS: {key} requires path and sha256")
    path = require_path(item, "path")
    expected = require_digest(require_str(item, "sha256"))
    if hashlib.sha256(_test_hooks.read_bytes(path)).hexdigest() != expected:
        raise ValueError(f"RESEARCH_INPUT_DRIFT: {key} differs from its frozen bytes")
    return path


def invoke(
    argv: tuple[str, ...],
    log: pathlib.Path,
    timeout_seconds: int,
    environment: tuple[tuple[str, str], ...] = (),
    accepted: tuple[int, ...] = (0,),
) -> JSONObject:
    """Run an existing controller with durable private output and a hard deadline.

    Args:
        argv: Fixed controller command and validated file arguments.
        log: Private append-only transcript; never placed on the shared board.
        timeout_seconds: Per-command bound, shorter than the scheduled tick.
        environment: Explicit private credentials, excluded from the transcript.
        accepted: Exit codes on which the controller reported structured
            metadata, e.g. admission's 2 for a recorded policy refusal.

    Returns:
        Decoded controller metadata on an accepted process completion.

    Raises:
        ValueError: RESEARCH_CONTROLLER_PROCESS on timeout or any other exit;
            the queue must retain its native receipt for the next reconciliation.
        OSError: A process or log could not be started or recorded.
    """
    result = _test_hooks.run(argv, timeout_seconds=timeout_seconds, set_env=environment)
    _test_hooks.append_text(
        log,
        dump_json_str(
            {
                "at": _test_hooks.now(),
                "argv": list(argv),
                "exit": result["returncode"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
            }
        ),
    )
    if result["returncode"] not in accepted:
        raise ValueError(
            f"RESEARCH_CONTROLLER_PROCESS: exit {result['returncode']}; private transcript {log}"
        )
    return narrow_json_to_dict(load_json_str(result["stdout"]))


__all__ = ["invoke", "read_pinned", "read_specification", "require_path", "verified_input"]
