"""Stable digests over canonically encoded observations.

BLAKE2b at a pinned 32-byte output. Pinned because the digest width is part of
the on-disk record format: widening it later would silently invalidate every
stored comparison rather than failing loudly.

The domain separator is what stops a step digest from ever equalling a run
digest built from the same bytes. Without it, a run of exactly one step could
produce a run digest identical to its own step digest, and a comparison that
mixed the two would report agreement it never established.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from navprobe.canonical import encode_row, encode_text

#: Digest width in bytes. Part of the record format, so it is pinned rather
#: than defaulted.
DIGEST_SIZE = 32

#: Prefix mixed into a per-step digest.
_STEP_DOMAIN = b"navprobe.step.v1"

#: Prefix mixed into a whole-run digest.
_RUN_DOMAIN = b"navprobe.run.v1"


def digest_step(step_index: int, values: Sequence[float]) -> str:
    """Digest one step's observation.

    The step index is mixed in, so an identical observation appearing at two
    different steps yields two different digests. A rollout that stalled and
    repeated a frame is therefore visible in the step sequence rather than
    collapsing into a run of equal digests.

    Args:
        step_index: Zero-based position of this step within the rollout.
        values: The step's observation, already flattened to floats.

    Returns:
        The digest as lowercase hexadecimal.

    Raises:
        CanonicalEncodingError: When ``values`` cannot be canonically encoded.
    """
    hasher = hashlib.blake2b(digest_size=DIGEST_SIZE)
    hasher.update(_STEP_DOMAIN)
    hasher.update(encode_row([float(step_index)]))
    hasher.update(encode_row(values))
    return hasher.hexdigest()


def digest_run(step_digests: Sequence[str]) -> str:
    """Fold a rollout's step digests into one run digest.

    Each digest is length-prefixed through :func:`navprobe.canonical.encode_text`
    rather than concatenated. Concatenation is not injective over a list of
    strings: ``["aab", "b"]`` and ``["aa", "bb"]`` flatten to the same bytes and
    carry the same step count, so a folded digest would report two different
    rollouts as identical. The count prefix alone does not close that gap
    because it only separates lists of different lengths.

    Args:
        step_digests: The per-step digests, in step order.

    Returns:
        The run digest as lowercase hexadecimal. A rollout of zero steps has a
        well-defined digest, which is what lets an empty run compare equal to
        another empty run instead of being a special case at every call site.
    """
    hasher = hashlib.blake2b(digest_size=DIGEST_SIZE)
    hasher.update(_RUN_DOMAIN)
    hasher.update(encode_row([float(len(step_digests))]))
    for step_digest in step_digests:
        hasher.update(encode_text(step_digest))
    return hasher.hexdigest()


__all__ = ["DIGEST_SIZE", "digest_run", "digest_step"]
