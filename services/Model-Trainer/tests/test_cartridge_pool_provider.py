"""The seed formula three sweeps share, tested where it now lives.

MOVED HERE WITH THE CLASS. These tests were the diverse sweep's, because the
diverse sweep was one of the three places the provider was written out. What
they actually pin is the seed formula and the per-seed caching, and both are
now owned by one module.

THE NESTING TEST IS THE LOAD-BEARING ONE. It asserts that this provider's
member zero is byte-identical to the varied sweep's, which is what lets the
recorded grids be compared as supersets along one chain. That claim is the
reason three copies of the formula were dangerous: nothing would have failed
if one drifted, and the records would still have been compared.
"""

from __future__ import annotations

import torch

from model_trainer.cli import cartridge_varied_companion_sweep as varied_sweep
from model_trainer.cli.cartridge_pool_provider import SeededPoolProvider
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.model.cartridge_pool_plans import VariedCompanionSweepPlan
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import CacheCapableLMProto

TINY_DIVERSE_PLAN: VariedCompanionSweepPlan = {
    "model_id": "gpt2",
    "window": 8,
    "held_out_stride": 3,
    "compartment_counts": (2, 3),
    "slots": 2,
    "probability": 0.5,
    "max_companions": 2,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
    "precision_selector": "policy",
}

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]


def _train_windows(seed: int, rows: int) -> list[torch.Tensor]:
    """Draw deterministic training windows for a provider under test.

    Args:
        seed: Seed for the draw.
        rows: How many windows.

    Returns:
        One (1, 8) id tensor per window.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return [
        torch.randint(0, _VOCAB, (1, 8), generator=generator, dtype=torch.long) for _ in range(rows)
    ]


def _cache_capable_base() -> CacheCapableLMProto:
    """Build the tiny base the provider tests share.

    Returns:
        The cache-capable tiny GPT-2.
    """
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_cache_capable(model)


def _provider(*trains: list[torch.Tensor]) -> SeededPoolProvider:
    """Build a provider over the given companion corpora.

    Args:
        *trains: One training-window sequence per pool member, in pool order.

    Returns:
        The provider, carrying the tiny plan's knobs.
    """
    return SeededPoolProvider(
        _cache_capable_base(),
        list(trains),
        slots=TINY_DIVERSE_PLAN["slots"],
        seeds=TINY_DIVERSE_PLAN["seeds"],
        epochs=TINY_DIVERSE_PLAN["epochs"],
        learning_rate=TINY_DIVERSE_PLAN["learning_rate"],
    )


class TestSeededPoolProvider:
    def test_the_pool_is_one_object_per_seed_and_one_member_per_corpus(self) -> None:
        provider = _provider(_train_windows(31, 4), _train_windows(37, 4))

        assert provider.pool(7) is provider.pool(7)
        assert provider.pool(7) is not provider.pool(8)
        assert len(provider.pool(7)) == TINY_DIVERSE_PLAN["max_companions"]

    def test_members_trained_on_different_corpora_differ(self) -> None:
        """The pool is diverse in fact, not only in flag order."""
        provider = _provider(_train_windows(31, 4), _train_windows(37, 4))

        first, second = provider.pool(7)
        differing = [
            name
            for name, tensor in first.state_dict().items()
            if not torch.equal(tensor, second.state_dict()[name])
        ]
        assert sorted(differing) == sorted(first.state_dict())

    def test_the_first_member_nests_the_varied_pools_first_member(self) -> None:
        """With the first corpus shared, the two providers' member zero agree.

        Both derive member zero's seed from the same formula, so the diverse
        record's pool contains the varied record's first member byte for
        byte -- the three grids compare as supersets along one chain.
        """
        base = _cache_capable_base()
        shared = _train_windows(31, 4)
        diverse = SeededPoolProvider(
            base,
            [shared, _train_windows(37, 4)],
            slots=TINY_DIVERSE_PLAN["slots"],
            seeds=TINY_DIVERSE_PLAN["seeds"],
            epochs=TINY_DIVERSE_PLAN["epochs"],
            learning_rate=TINY_DIVERSE_PLAN["learning_rate"],
        )
        varied = varied_sweep._PoolProvider(base, shared, TINY_DIVERSE_PLAN)

        mine = diverse.pool(7)[0].state_dict()
        theirs = varied.pool(7)[0].state_dict()
        assert sorted(mine) == sorted(theirs)
        for name, tensor in mine.items():
            assert torch.equal(tensor, theirs[name]), name
