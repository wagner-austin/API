"""The configuration a measured number was produced under.

A cloze accuracy is not a property of a model and an item set. It is a
property of a model and an item set scored on a particular image, on a
particular card, under particular determinism settings. Recording the number
without those makes it uncheckable: a later measurement that disagrees cannot
be told apart from a working image that merely moved to a different GPU.

That is not hypothetical here. The unexposed-gpt2 floor of 52.3030% is
load-bearing -- ``tools/extraction-eval/README.md`` instructs readers to
report lift above it and never absolute accuracy -- and the record behind it
carried no device, no driver, no torch version and no timestamp. Meanwhile
the training manifests beside it DID record ``versions.torch``, which is how
a 2026-08-25 audit found that the published arms span two torch major
versions with four contrasts crossing the boundary. The scoring path simply
never got the treatment the training path had.

This module builds the fingerprint that closes that gap, from the same hooks
the training manifest already reads, so there is one way to answer "what did
this run on" rather than two that can drift. Encoding and decoding are
:func:`platform_ml.encode_run_fingerprint` and
:func:`platform_ml.decode_run_fingerprint` directly -- callers use those, and
this module does not restate them.
"""

from __future__ import annotations

from platform_ml import DeterminismReport, RunFingerprint

from model_trainer.core import _test_hooks

CUDA_DEVICE = "cuda"

# A run that used no GPU has no card and no driver, and the empty string is
# how RunFingerprint spells that: it compares as a difference against any
# known value rather than matching everything, so a cpu-scored number never
# silently compares equal to a cuda-scored one.
NO_GPU = ""


def capture_run_fingerprint(device: str, determinism: DeterminismReport) -> RunFingerprint:
    """Record what this process is about to compute a number on.

    The card and driver are read only for a cuda run. Querying them for a cpu
    run would initialise a CUDA context to describe hardware the run does not
    touch, and would put a card in the record of a measurement that never used
    one.

    ``determinism`` is passed in rather than applied here: pinning is a
    process-global side effect that belongs to the job, at the point that
    precedes any CUDA work, and this function only describes. Taking the
    report as an argument also means the fingerprint can only claim a posture
    that some caller actually applied.

    Args:
        device: The device the measurement runs on, ``"cuda"`` or ``"cpu"``.
        determinism: What determinism was put in force, as returned by
            :func:`platform_ml.determinism.apply_determinism`.

    Returns:
        The fingerprint, with the image digest taken from the build-stamped
        commit. An unstamped build records the empty string, which reads as
        "unknown" and differs from every known digest rather than matching
        any of them.
    """
    stamped = _test_hooks.env_git_commit()
    on_cuda = device == CUDA_DEVICE
    return RunFingerprint(
        image_digest=stamped if stamped is not None else NO_GPU,
        gpu_model=_test_hooks.cuda_device_name() if on_cuda else NO_GPU,
        driver_version=_test_hooks.cuda_driver_version() if on_cuda else NO_GPU,
        determinism=determinism,
    )


def describe_run_fingerprint(fingerprint: RunFingerprint) -> str:
    """Render a fingerprint as one line for a job log.

    Args:
        fingerprint: The fingerprint to render.

    Returns:
        A line naming the image, card and driver, so a reader who has only
        the logs can tell two measurements' configurations apart without
        fetching the stored record. An empty field renders as a word rather
        than as nothing, because a blank in a log reads as a formatting fault
        instead of as the absence it records.
    """
    return (
        f"image={fingerprint['image_digest'] or 'unknown'} "
        f"gpu={fingerprint['gpu_model'] or 'none'} "
        f"driver={fingerprint['driver_version'] or 'none'}"
    )


__all__ = [
    "CUDA_DEVICE",
    "NO_GPU",
    "capture_run_fingerprint",
    "describe_run_fingerprint",
]
