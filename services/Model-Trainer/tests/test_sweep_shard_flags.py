"""The three shard flags, read into exactly one of three roles or refused."""

from __future__ import annotations

import pathlib

import pytest

from model_trainer.cli._sweep_shard_flags import (
    SHARD_COUNT_FLAG,
    SHARD_FLAG,
    SHARDS_FLAG,
    read_shard_role,
)
from model_trainer.core.services.model.cartridge_sweep_shards import SweepMerge, SweepShard


class TestTheThreeForms:
    """Straight, shard, merge."""

    def test_no_shard_flag_is_a_straight_run(self) -> None:
        """The ordinary invocation is untouched."""
        assert read_shard_role({"--plan": "p"}) == {"shard": None, "merge": None}

    def test_all_three_are_one_shard(self) -> None:
        """Shard K of N, under the root."""
        role = read_shard_role({SHARDS_FLAG: "root", SHARD_COUNT_FLAG: "46", SHARD_FLAG: "45"})
        assert role["shard"] == SweepShard(root=pathlib.Path("root"), count=46, index=45)
        assert role["merge"] is None

    def test_root_and_count_are_the_merge(self) -> None:
        """Every share, adopted."""
        role = read_shard_role({SHARDS_FLAG: "root", SHARD_COUNT_FLAG: "2"})
        assert role["merge"] == SweepMerge(root=pathlib.Path("root"), count=2)
        assert role["shard"] is None


class TestRefusals:
    """Each half-given form has a plausible wrong reading."""

    def test_a_shard_index_alone_is_refused(self) -> None:
        """It would otherwise run the whole sweep and call it a shard."""
        with pytest.raises(ValueError, match="given without both"):
            read_shard_role({SHARD_FLAG: "0"})

    def test_a_non_integer_count_is_refused(self) -> None:
        """Named, with the value given."""
        with pytest.raises(ValueError, match="'two'"):
            read_shard_role({SHARDS_FLAG: "r", SHARD_COUNT_FLAG: "two"})

    def test_zero_shards_is_refused(self) -> None:
        """A sweep cut into nothing."""
        with pytest.raises(ValueError, match="at least 1"):
            read_shard_role({SHARDS_FLAG: "r", SHARD_COUNT_FLAG: "0"})

    def test_an_index_past_the_count_is_refused(self) -> None:
        """Shards are numbered from zero."""
        with pytest.raises(ValueError, match="is not below"):
            read_shard_role({SHARDS_FLAG: "r", SHARD_COUNT_FLAG: "2", SHARD_FLAG: "2"})
