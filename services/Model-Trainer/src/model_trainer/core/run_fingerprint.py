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
the training manifests beside it DID record ``versions.torch``. The scoring
path simply never got the treatment the training path had.

A CORRECTION TO WHAT THAT FIELD WAS TAKEN TO SHOW. This docstring previously
said a 2026-08-25 audit found the published arms spanning two torch major
versions with four contrasts crossing the boundary. **That reading does not
survive checking the field itself.** Fifteen of the thirty-nine archived
manifests record ``versions.torch == "2.13.0"``; PyTorch has never released a
2.13.0, this repository's lock has pinned ``2.6.0+cu124`` continuously since
2025-12-19, no commit on any branch ever pinned 2.13, and no Dockerfile or
requirement names it. The value also changes at a clean time boundary
(2026-08-16 20:38 to 2026-08-17 00:57) while ``transformers``, ``tokenizers``
and ``datasets`` stay identical across it.

So the field recorded something other than torch's version for those runs, and
what it recorded is not recoverable from the archive. The accurate statement is
weaker and less alarming than the one it replaces: **fifteen of thirty-nine
runs carry an unusable version record**, which is a provenance gap rather than
a demonstrated split. It is still a reason this module exists -- a field nobody
can interpret is exactly as useless as a field nobody wrote.

This module builds the fingerprint that closes that gap, from the same hooks
the training manifest already reads, so there is one way to answer "what did
this run on" rather than two that can drift. Encoding and decoding are
:func:`platform_ml.encode_run_fingerprint` and
:func:`platform_ml.decode_run_fingerprint` directly -- callers use those, and
this module does not restate them.
"""

from __future__ import annotations

from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.environment_record import (
    capture_host_record,
    capture_package_versions,
)

from model_trainer.core import _test_hooks

CUDA_DEVICE = "cuda"

#: The libraries whose arithmetic decides this service's numbers.
#:
#: Not every installed distribution. A fingerprint over all of them would
#: differ between two runs over a dev-dependency bump that cannot reach a
#: matmul, and every spurious difference makes a real one harder to see.
#: These three can: ``torch`` selects the kernels, ``transformers`` decides
#: which attention path the model takes -- the 2026-08-27 measurement traced
#: a cross-card divergence to `GPT2SdpaAttention` choosing
#: `EFFICIENT_ATTENTION` -- and ``numpy`` backs the scoring arithmetic.
FINGERPRINT_DISTRIBUTIONS: tuple[str, ...] = ("numpy", "torch", "transformers")

# A run that used no GPU has no card and no driver, and the empty string is
# how RunFingerprint spells that: it compares as a difference against any
# known value rather than matching everything, so a cpu-scored number never
# silently compares equal to a cuda-scored one.
NO_GPU = ""


def capture_run_fingerprint(
    device: str,
    determinism: DeterminismRecord,
    distributions: tuple[str, ...] = FINGERPRINT_DISTRIBUTIONS,
) -> RunFingerprint:
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
        distributions: The libraries whose versions decide this run's
            numbers. Defaults to :const:`FINGERPRINT_DISTRIBUTIONS`, which is
            what every benchmark records. Training passes a wider set when
            the run used adapters or quantization, because `peft` and
            `bitsandbytes` decide that run's arithmetic and are absent from
            a benchmark's. Naming them is the caller's job by
            :func:`~platform_core.environment_record.capture_package_versions`'s
            own contract: recording every installed package would make two
            runs differ over a dev-dependency bump that cannot reach a
            number.

    Returns:
        The fingerprint. ``image_digest`` is the digest of the image that
        ran, read from the variable the launcher exports; a run with no
        image records the empty string, which reads as "unknown" and differs
        from every known digest rather than matching any of them.

        It is deliberately NOT the code commit. This field held the commit
        while no image existed, and the two are different questions: a
        commit says which code was built, a digest says which environment
        ran it. Two runs can share a commit and differ in torch -- which is
        precisely the difference that put four published contrasts across a
        torch major-version boundary, and precisely what a fingerprint is
        for. The commit remains in the training manifest, where it answers
        its own question.
    """
    digest = _test_hooks.env_image_digest()
    on_cuda = device == CUDA_DEVICE
    return RunFingerprint(
        image_digest=digest if digest is not None else NO_GPU,
        gpu_model=_test_hooks.cuda_device_name() if on_cuda else NO_GPU,
        driver_version=_test_hooks.cuda_driver_version() if on_cuda else NO_GPU,
        determinism=determinism,
        host=capture_host_record(_test_hooks.host_probe()),
        packages=capture_package_versions(distributions, _test_hooks.installed_version),
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
    "FINGERPRINT_DISTRIBUTIONS",
    "NO_GPU",
    "capture_run_fingerprint",
    "describe_run_fingerprint",
]
