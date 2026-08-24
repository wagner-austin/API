"""Running one training job on a Slurm compute node.

This is a composition root, not a second trainer. ``worker_entry`` wires the
service deployment -- Redis, RQ, the data-bank API; this wires the cluster --
an in-process store, a staged corpus, a local artifact directory. Both hand
the same payload to the same :func:`process_train_job`, so the training code
has no idea which one started it and cannot drift between them.

What a compute node changes, and what it does not:

* **Progress state** has no reader. The API that would have consumed it is not
  here, so it lives in memory and the run's real signal is its stdout, which
  Slurm captures and ``hpc3-triage`` already watches for silence.
* **Cancellation** is Slurm's. ``scancel`` terminates the process; there is no
  API to write a cancel key, so the key is never set and the poll that reads
  it always answers False. That is correct rather than a gap -- a second
  cancellation mechanism that the scheduler did not know about would be the
  bug.
* **The corpus** must already be there. ``hpc3-stage`` places it and verifies
  its SHA-256 on both sides; a compute node has no service to fetch from.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path

from platform_core.json_utils import JSONObject, load_json_str
from platform_core.logging import get_logger
from platform_workers.local_kv import LocalKV
from platform_workers.redis import RedisStrProto

from model_trainer.cluster import _test_hooks as cluster_hooks
from model_trainer.cluster.stores import LocalArtifacts, StagedCorpus
from model_trainer.core import _test_hooks
from model_trainer.core._hook_protocols_ml import CorpusFetcherProto

_log = get_logger(__name__)

CORPUS_FLAG = "--corpus-dir"
ARTIFACTS_FLAG = "--artifacts-dir"
PAYLOAD_FLAG = "--payload"
_FLAGS = (PAYLOAD_FLAG, CORPUS_FLAG, ARTIFACTS_FLAG)


def _publish_to_log(channel: str, message: str) -> None:
    """Put a published event where a cluster operator will actually find it.

    Args:
        channel: Channel the event was published to.
        message: The event body.
    """
    _log.info("event", extra={"channel": channel, "message": message})


def install_cluster_hooks(*, corpus_dir: Path, artifacts_dir: Path) -> None:
    """Point the trainer's three service dependencies at the local machine.

    Production wiring through the hook surface, which is what that surface is
    for: the service deployment installs HTTP-backed implementations at
    startup and this installs filesystem-backed ones. Neither is a fake, and
    the training code between them is identical.

    Args:
        corpus_dir: Directory holding staged corpora, keyed by digest.
        artifacts_dir: Directory to write run artifacts into.
    """

    def _kv(url: str) -> RedisStrProto:
        """Build the in-process store, ignoring the Redis URL.

        Args:
            url: Configured Redis URL, unused. Accepted because the factory
                signature is shared with the deployment that needs it.

        Returns:
            A store backed by this process.
        """
        _log.info("kv store is in-process", extra={"configured_url": url})
        return LocalKV(publish=_publish_to_log, clock=time.monotonic)

    def _corpus(api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        """Build the staged-corpus resolver, ignoring the API settings.

        Args:
            api_url: Data-bank URL, unused on a compute node.
            api_key: Data-bank key, unused on a compute node.
            cache_dir: Download cache, unused -- nothing is downloaded.

        Returns:
            A resolver reading the staged directory.
        """
        return StagedCorpus(corpus_dir)

    _test_hooks.kv_store_factory = _kv
    _test_hooks.corpus_fetcher_factory = _corpus
    _test_hooks.artifact_store_factory = lambda base_url, api_key, **kwargs: LocalArtifacts(
        artifacts_dir
    )


def _require_flag(parsed: dict[str, str], flag: str) -> str:
    """Read a flag the caller must have supplied.

    Args:
        parsed: Parsed arguments.
        flag: Flag name.

    Returns:
        Its value.

    Raises:
        ValueError: If the flag is absent. Every one of these names a
            directory or file with no sane default: guessing where a corpus
            lives would find the wrong one as easily as none.
    """
    value = parsed.get(flag)
    if value is None:
        raise ValueError(f"{flag} is required; got {sorted(parsed)}")
    return value


def _parse(tokens: Sequence[str]) -> dict[str, str]:
    """Parse ``--flag value`` pairs.

    Args:
        tokens: Arguments excluding the program name.

    Returns:
        Flag to value.

    Raises:
        ValueError: If a token is not a known flag, or a flag has no value.
    """
    parsed: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        if flag not in _FLAGS:
            raise ValueError(f"Unknown argument {flag!r}; expected one of {list(_FLAGS)}")
        if index + 1 >= len(tokens):
            raise ValueError(f"{flag} needs a value")
        parsed[flag] = tokens[index + 1]
        index += 2
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run one training job from a payload file.

    Args:
        argv: Arguments excluding the program name. Defaults to the process
            arguments.

    Returns:
        0 when the job completed.

    Raises:
        ValueError: If a required flag is missing or an argument is unknown.
        JSONTypeError: If the payload is not a JSON object.
        AppError: If the corpus is not staged, or training fails.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = _parse(tokens)

    payload_path = Path(_require_flag(parsed, PAYLOAD_FLAG))
    corpus_dir = Path(_require_flag(parsed, CORPUS_FLAG))
    artifacts_dir = Path(_require_flag(parsed, ARTIFACTS_FLAG))

    install_cluster_hooks(corpus_dir=corpus_dir, artifacts_dir=artifacts_dir)

    raw = load_json_str(payload_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{payload_path} must hold a JSON object")
    payload: JSONObject = raw

    _log.info(
        "cluster training start",
        extra={"payload": str(payload_path), "corpus_dir": str(corpus_dir)},
    )
    cluster_hooks.run_job(payload)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = [
    "ARTIFACTS_FLAG",
    "CORPUS_FLAG",
    "PAYLOAD_FLAG",
    "entrypoint",
    "install_cluster_hooks",
    "main",
]
