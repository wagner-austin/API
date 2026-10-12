"""The arithmetic and exact implementation behind a market research run."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from platform_core import json_utils
from platform_core.comparability import RunFingerprint, cpu_run_fingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.environment_record import capture_host_record, capture_package_versions

from . import _test_hooks


def market_fingerprint(
    determinism: DeterminismRecord, get_env: Callable[[str], str | None]
) -> RunFingerprint:
    """Capture the versions that determine data parsing and statistical arithmetic.

    Args:
        determinism: Posture pinned before loading numeric libraries.
        get_env: Environment reader used by the shared fingerprint contract.

    Returns:
        Actual host, configuration, and installed runtime versions.

    Raises:
        PackageNotFoundError: A required distribution is absent.
    """
    return cpu_run_fingerprint(
        determinism,
        get_env,
        capture_host_record(_test_hooks.host_probe),
        capture_package_versions(
            ("covenant-ml", "platform-core", "numpy", "scipy", "httpx", "scikit-learn"),
            _test_hooks.installed_version,
        ),
    )


def implementation_digest() -> str:
    """Hash this package's implementation, shared core, CLI, and locked dependencies.

    Returns:
        SHA-256 over sorted relative filenames and exact file bytes.

    Raises:
        OSError: An implementation file cannot be read.
    """
    package = Path(__file__).parents[3]
    core = Path(json_utils.__file__).parent
    files = {
        "covenant_ml/" + p.relative_to(package / "src").as_posix(): p
        for p in (package / "src").rglob("*.py")
    }
    files.update({"platform_core/" + p.relative_to(core).as_posix(): p for p in core.rglob("*.py")})
    files.update(
        {
            "scripts/" + p.relative_to(package / "scripts").as_posix(): p
            for p in (package / "scripts").rglob("*.py")
        }
    )
    for name in ("pyproject.toml", "poetry.lock"):
        files[name] = package / name
    digest = sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(sha256(files[name].read_bytes()).digest())
    return digest.hexdigest()


__all__ = ["implementation_digest", "market_fingerprint"]
