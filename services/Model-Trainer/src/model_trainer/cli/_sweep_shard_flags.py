"""The three flags that say whether a sweep job is straight, a shard, or a merge.

Parsed at the edge, once, for every sweep that shards: the trait grid and the
trait repair grid read the same flags and must mean the same thing by them,
because one sweep document submits the shards and one chain stage the merge.
What a shard and a merge DO is
:mod:`~model_trainer.core.services.model.cartridge_sweep_shards`'s business;
this module only decides which one a command line asked for, and refuses a
combination that is neither.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

from typing_extensions import TypedDict

from model_trainer.core.services.model.cartridge_sweep_shards import SweepMerge, SweepShard

SHARDS_FLAG = "--shards"
SHARD_COUNT_FLAG = "--shard-count"
SHARD_FLAG = "--shard"

SHARD_FLAGS = (SHARDS_FLAG, SHARD_COUNT_FLAG, SHARD_FLAG)


class ShardRole(TypedDict):
    """What one job is in a sharded sweep. Both None is a straight run.

    Attributes:
        shard: Set when the job measures one share and writes no record.
        merge: Set when the job adopts every share and writes the record.
    """

    shard: SweepShard | None
    merge: SweepMerge | None


def _count(text: str, flag: str) -> int:
    """Read a non-negative integer flag value.

    Args:
        text: The value given.
        flag: The flag it was given for, to name in a refusal.

    Returns:
        The integer.

    Raises:
        ValueError: When the value is not a run of decimal digits.
    """
    if not text.isdigit():
        raise ValueError(f"{flag} takes a non-negative integer, got {text!r}")
    return int(text)


def read_shard_role(parsed: Mapping[str, str]) -> ShardRole:
    """Decide what the shard flags given ask this job to be.

    ``--shards DIR --shard-count N --shard K`` is shard K of N;
    ``--shards DIR --shard-count N`` is the merge over all N; none of them is a
    straight run. Anything else is refused, because each half-given form has
    a plausible wrong reading -- ``--shard`` alone would otherwise run the
    whole sweep and call it a shard.

    Args:
        parsed: The command line's flags.

    Returns:
        The role.

    Raises:
        ValueError: When the flags given are not one of the three forms, or
            the count is zero, or the index is not below the count.
    """
    given = tuple(flag for flag in SHARD_FLAGS if flag in parsed)
    if not given:
        return ShardRole(shard=None, merge=None)
    if SHARDS_FLAG not in parsed or SHARD_COUNT_FLAG not in parsed:
        raise ValueError(
            f"{' '.join(given)} given without both {SHARDS_FLAG} and {SHARD_COUNT_FLAG}; a "
            f"shard is {SHARDS_FLAG} DIR {SHARD_COUNT_FLAG} N {SHARD_FLAG} K and a merge "
            f"is {SHARDS_FLAG} DIR {SHARD_COUNT_FLAG} N"
        )
    root = pathlib.Path(parsed[SHARDS_FLAG])
    count = _count(parsed[SHARD_COUNT_FLAG], SHARD_COUNT_FLAG)
    if count == 0:
        raise ValueError(f"{SHARD_COUNT_FLAG} must be at least 1")
    if SHARD_FLAG not in parsed:
        return ShardRole(shard=None, merge=SweepMerge(root=root, count=count))
    index = _count(parsed[SHARD_FLAG], SHARD_FLAG)
    if index >= count:
        raise ValueError(f"{SHARD_FLAG} {index} is not below {SHARD_COUNT_FLAG} {count}")
    return ShardRole(shard=SweepShard(root=root, count=count, index=index), merge=None)


__all__ = [
    "SHARDS_FLAG",
    "SHARD_COUNT_FLAG",
    "SHARD_FLAG",
    "SHARD_FLAGS",
    "ShardRole",
    "read_shard_role",
]
