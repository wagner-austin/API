"""What a cartridge measurement COST, as opposed to what it found.

Split from `test_cartridge_benchmark.py` by role rather than by size: that
module exercises the measurement -- real GPT-2, real cartridges, real scored
items -- while everything here is pure arithmetic over a plan and two window
counts, and needs no model at all.

EVERY ASSERTION DISTINGUISHES THE REAL COUNT FROM A PLAUSIBLE MIS-DERIVATION
OF IT. A cost number that is merely self-consistent is worse than no cost
number, because it would be believed. So the tests pin the two places the
arithmetic could quietly go wrong -- assuming the second corpus matches the
primary, and assuming the composition arm trains one cartridge per seed
instead of two.
"""

from __future__ import annotations

from model_trainer.cli import cartridge_benchmark as bench
from model_trainer.core.services.model.cartridge_plans import CartridgePlan
from tests.test_cartridge_benchmark import TINY_PLAN


class TestWindowsTrained:
    def test_the_second_corpus_is_counted_at_its_own_size(self) -> None:
        """The tempting bug is assuming the two corpora are the same size.

        `measure_plan` truncates the second to at most the primary's length,
        so they are equal only when the second corpus is at least as large.
        A smaller second corpus must cost strictly less.
        """
        equal = bench.windows_trained(TINY_PLAN, train_windows=10, second_train_windows=10)
        smaller = bench.windows_trained(TINY_PLAN, train_windows=10, second_train_windows=4)

        # 2 slot counts x 3 seeds x 1 epoch x 10 = 60 sweep, plus
        # 3 seeds x 1 epoch x (10 + S) composition.
        assert equal == 60 + 3 * (10 + 10)
        assert smaller == 60 + 3 * (10 + 4)
        assert smaller < equal

    def test_the_composition_arm_trains_two_cartridges_per_seed(self) -> None:
        """An empty sweep isolates composition, which is where the 2x lives.

        Halve this and you get the count a reader would derive from assuming
        one cartridge per seed, which is what `measure_composition` does NOT
        do -- it trains a second under a different seed so the pair are not
        one draw trained twice.
        """
        composition_only: CartridgePlan = {**TINY_PLAN, "slot_counts": ()}

        trained = bench.windows_trained(composition_only, train_windows=10, second_train_windows=10)

        assert trained == 3 * (10 + 10)
        assert trained != 3 * 10

    def test_epochs_multiply_every_arm(self) -> None:
        two_epochs: CartridgePlan = {**TINY_PLAN, "epochs": 2}

        assert bench.windows_trained(
            two_epochs, train_windows=10, second_train_windows=10
        ) == 2 * bench.windows_trained(TINY_PLAN, train_windows=10, second_train_windows=10)

    def test_an_untrained_control_costs_nothing(self) -> None:
        """A plan with no sweep and no windows trains nothing at all.

        `measure_untrained` draws a RANDOM prefix and never optimises it, so
        it contributes no pass. A count that included it would be reporting
        work the control exists to avoid doing.
        """
        empty: CartridgePlan = {**TINY_PLAN, "slot_counts": ()}

        assert bench.windows_trained(empty, train_windows=0, second_train_windows=0) == 0


class TestCostObservations:
    def test_it_names_the_counts_the_record_used_to_drop(self) -> None:
        named = bench.cost_observations(
            TINY_PLAN,
            corpus_tokens=400,
            second_corpus_tokens=250,
            train_windows=10,
            held_out_windows=5,
            second_train_windows=10,
        )

        assert {observation["name"]: observation["value"] for observation in named} == {
            "corpus_tokens": 400.0,
            "second_corpus_tokens": 250.0,
            "window_tokens": 8.0,
            "train_windows": 10.0,
            "held_out_windows": 5.0,
            "second_train_windows": 10.0,
            "windows_trained": 120.0,
            "training_tokens": 960.0,
        }

    def test_training_tokens_is_the_passes_times_the_window(self) -> None:
        """Recorded separately because they answer different questions, and a
        reader must be able to check one against the other.
        """
        named = {
            observation["name"]: observation["value"]
            for observation in bench.cost_observations(
                TINY_PLAN,
                corpus_tokens=400,
                second_corpus_tokens=400,
                train_windows=7,
                held_out_windows=3,
                second_train_windows=2,
            )
        }

        assert named["training_tokens"] == named["windows_trained"] * named["window_tokens"]

    def test_a_corpus_that_trains_nothing_still_reports_its_size(self) -> None:
        """Zero passes is a FINDING, not an absence.

        A record missing these names reads as "not measured"; a zero reads as
        "measured, and it trained on nothing", which is the thing a reader
        chasing an empty result needs to see.
        """
        empty: CartridgePlan = {**TINY_PLAN, "slot_counts": ()}

        named = {
            observation["name"]: observation["value"]
            for observation in bench.cost_observations(
                empty,
                corpus_tokens=400,
                second_corpus_tokens=0,
                train_windows=0,
                held_out_windows=0,
                second_train_windows=0,
            )
        }

        assert named["corpus_tokens"] == 400.0
        assert named["windows_trained"] == 0.0
        assert named["training_tokens"] == 0.0
