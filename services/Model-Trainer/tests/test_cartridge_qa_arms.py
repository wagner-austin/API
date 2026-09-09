"""The question set's split, and what each retrieval arm costs to serve.

Split from `test_cartridge_qa_benchmark.py` by role when that module passed
the 600-line ceiling: this one is about WHICH text each arm sees and how
long each arm takes, while the other covers the record the run emits.

THE TWO THINGS WORTH KNOWING BEFORE EDITING EITHER SET.

The split tests exist because a defect here is invisible from every arm's
score. Three of twelve real pages were once trained on and never examined,
and no arm could have revealed it -- a short question set is
indistinguishable from a short corpus.

The latency tests script the clock rather than reading a real one, because
a real clock supports only "the name is present". That would pass just as
happily if two arms were timed the wrong way round, if a bracket read the
same instant twice, or if a cost belonging to one arm were charged to
another -- which is exactly the class of error the arms' asymmetric
accounting invites.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator

import pytest

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import cartridge_qa_benchmark as bench
from model_trainer.core.services.model.cartridge_qa_plans import QaPlan
from tests._qa_benchmark_support import (
    DOCUMENTS,
    TINY_PLAN,
    Tokenizer,
    install_fakes,
    restore_fakes,
)


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the shared fakes, and put the real hooks back afterwards."""
    install_fakes()
    yield None
    restore_fakes()


class TestHeldOutSplit:
    """The split has to examine every page it trains on.

    Measured on the real twelve-page corpus, the first version of this strode
    over the GLOBAL window index, and three of twelve pages held out nothing:
    trained on, never tested. A short question set looks exactly like a short
    corpus, so nothing surfaced it.
    """

    def test_every_document_is_examined_not_only_trained_on(self) -> None:
        """The assertion the global stride failed on the real corpus.

        Asserted through the ITEM IDS, which carry their document index as a
        ``d{index:03d}`` prefix, because that is the observable consequence:
        a page that holds out nothing produces no item, and the arm is then
        fitted to more pages than it is scored on.

        `max_items` is raised for this test only. The tiny plan stops at six
        items, which could exhaust the budget inside the first documents and
        fail this for a reason that has nothing to do with the split.
        """
        plan: QaPlan = {**TINY_PLAN, "max_items": 120}
        tok = Tokenizer()
        encoder = bench.HFTokenizerEncoder(tok)
        encoded = [encoder.encode(document).ids for document in DOCUMENTS]

        items, _training = bench.build_question_set(DOCUMENTS, encoded, encoder, plan)

        examined = {item["item_id"].split("-")[0] for item in items}
        assert examined == {f"d{index:03d}" for index in range(len(DOCUMENTS))}

    def test_a_single_window_document_is_trained_on_rather_than_tested(self) -> None:
        """It cannot be both, and training is the useful half.

        Holding out its only window would leave that page's terms absent from
        the training text, and `build_items` would then refuse every item
        drawn from it as unanswerable -- so the page would be excluded either
        way, but silently and for a confusing reason.
        """
        one_window = "  ".join(DOCUMENTS[0].split()[: TINY_PLAN["window"] - 2])
        documents = (one_window, DOCUMENTS[1], DOCUMENTS[2], DOCUMENTS[3])
        tok = Tokenizer()
        encoder = bench.HFTokenizerEncoder(tok)
        encoded = [encoder.encode(document).ids for document in documents]

        _items, training = bench.build_question_set(documents, encoded, encoder, TINY_PLAN)

        first_terms = set(one_window.split())
        assert first_terms & set(training.split()), "the single-window page was not trained on"


class TestLatencyObservations:
    def test_it_names_every_arm_including_the_costs_it_excludes(self) -> None:
        """Two costs are recorded and left OUT of their arm's total, for
        opposite reasons, and the record has to show both.

        The oracle's build is excluded because it cheats -- no real pipeline
        pays it. The BM25 index build is excluded because it is OFFLINE -- a
        deployment pays it once per corpus change, not per query. The BM25
        SELECT is the one selection cost that is charged, because searching
        from the question is what every real retriever does per request.
        """
        named = {
            observation["name"]: observation["value"]
            for observation in bench.latency_observations(
                base_seconds=2.0,
                retrieval_seconds=10.0,
                cartridge_seconds=3.0,
                retrieval_build_seconds=0.5,
                real_seconds=7.0,
                real_select_seconds=1.0,
                real_index_seconds=0.25,
                dense_seconds=6.0,
                dense_select_seconds=0.5,
                dense_index_seconds=40.0,
                fused_seconds=9.0,
                fused_select_seconds=2.0,
            )
        }

        assert named == {
            "base_serve_seconds": 2.0,
            "retrieval_serve_seconds": 10.0,
            "cartridge_serve_seconds": 3.0,
            "retrieval_oracle_build_seconds": 0.5,
            "bm25_serve_seconds": 7.0,
            "bm25_select_seconds": 1.0,
            "bm25_total_serve_seconds": 8.0,
            "bm25_index_seconds": 0.25,
            "dense_serve_seconds": 6.0,
            "dense_select_seconds": 0.5,
            "dense_total_serve_seconds": 6.5,
            "dense_index_seconds": 40.0,
            "fused_serve_seconds": 9.0,
            "fused_select_seconds": 2.0,
            # 9.0 scoring + 2.0 fusing + 0.5 for the dense ranking it fused.
            "fused_total_serve_seconds": 11.5,
        }

    def test_a_huge_dense_index_never_reaches_a_per_request_total(self) -> None:
        """THE DEFECT THIS ARM SHIPPED WITH, pinned as arithmetic.

        Embedding the corpus inside every query recorded 17452 ms/item
        against BM25's 72. The index cost is offline and belongs out of both
        totals, symmetric with `bm25_index_seconds` -- so a forty-second
        corpus embed must leave a half-second query untouched.
        """
        named = {
            observation["name"]: observation["value"]
            for observation in bench.latency_observations(
                base_seconds=1.0,
                retrieval_seconds=1.0,
                cartridge_seconds=1.0,
                retrieval_build_seconds=1.0,
                real_seconds=1.0,
                real_select_seconds=1.0,
                real_index_seconds=1.0,
                dense_seconds=6.0,
                dense_select_seconds=0.5,
                dense_index_seconds=40.0,
                fused_seconds=9.0,
                fused_select_seconds=2.0,
            )
        }

        assert named["dense_total_serve_seconds"] == 6.5
        assert named["fused_total_serve_seconds"] == 11.5
        assert named["dense_index_seconds"] == 40.0

    def test_the_bm25_total_charges_selection_and_not_indexing(self) -> None:
        """The asymmetry is the whole design, so it is asserted directly."""
        named = {
            observation["name"]: observation["value"]
            for observation in bench.latency_observations(
                base_seconds=1.0,
                retrieval_seconds=1.0,
                cartridge_seconds=1.0,
                retrieval_build_seconds=1.0,
                real_seconds=7.0,
                real_select_seconds=1.0,
                real_index_seconds=100.0,
                dense_seconds=1.0,
                dense_select_seconds=1.0,
                dense_index_seconds=1.0,
                fused_seconds=1.0,
                fused_select_seconds=1.0,
            )
        }

        assert named["bm25_total_serve_seconds"] == 8.0
        assert named["bm25_index_seconds"] == 100.0


class TestServeLatency:
    def test_each_arm_is_timed_against_a_scripted_clock(self, tmp_path: pathlib.Path) -> None:
        """Twenty-six reads: five arms, then two per seed.

        The load-bearing assertions are the two totals. The cartridge's three
        seeds are scripted at 1.0, 2.0 and 3.0, so the MEAN is 2.0 and a sum
        would be 6.0 -- recording the sum would make the arm look worse the
        more seeds a plan declared. And the fused total must include the
        dense ranking it consumed, or the hybrid would appear to cost less
        than the arm it is built on.
        """
        ticks = iter(
            [
                100.0,
                102.0,  # base: 2.0
                102.0,
                102.5,  # oracle build: 0.5
                200.0,
                210.0,  # oracle retrieval: 10.0
                210.0,
                210.25,  # bm25 index: 0.25
                211.0,
                212.0,  # bm25 select: 1.0
                220.0,
                227.0,  # bm25 scoring: 7.0
                228.0,
                268.0,  # dense INDEX: 40.0, offline and out of every total
                230.0,
                232.0,  # dense select: 2.0
                240.0,
                248.0,  # dense scoring: 8.0
                250.0,
                253.0,  # fused select: 3.0
                260.0,
                269.0,  # fused scoring: 9.0
                300.0,
                301.0,  # seed 7: 1.0
                400.0,
                402.0,  # seed 8: 2.0
                500.0,
                503.0,  # seed 9: 3.0
            ]
        )
        cli_hooks.monotonic_clock = lambda: next(ticks)
        try:
            observations, _digest = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")
        finally:
            cli_hooks.monotonic_clock = cli_hooks._default_monotonic_clock

        named = {observation["name"]: observation["value"] for observation in observations}
        assert named["base_serve_seconds"] == 2.0
        assert named["retrieval_oracle_build_seconds"] == 0.5
        assert named["retrieval_serve_seconds"] == 10.0
        assert named["bm25_index_seconds"] == 0.25
        assert named["bm25_select_seconds"] == 1.0
        assert named["bm25_serve_seconds"] == 7.0
        assert named["bm25_total_serve_seconds"] == 8.0
        assert named["dense_index_seconds"] == 40.0
        assert named["dense_select_seconds"] == 2.0
        assert named["dense_serve_seconds"] == 8.0
        # The forty-second corpus embed is offline and stays out of this.
        assert named["dense_total_serve_seconds"] == 10.0
        assert named["fused_select_seconds"] == 3.0
        assert named["fused_serve_seconds"] == 9.0
        # Fusion CONSUMES the dense ranking, so a deployment pays it too:
        # 9.0 scoring + 3.0 fusing + 2.0 for the dense ranking it fused.
        assert named["fused_total_serve_seconds"] == 14.0
        assert named["cartridge_serve_seconds"] == 2.0

    def test_the_oracle_build_is_not_folded_into_the_retrieval_arm(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The two are adjacent in time and must stay separate in the record.

        A single bracket around selection-plus-scoring would charge the
        cartridge's competitor for an answer-aware search no real retriever
        can run, which is the one way this comparison could be made dishonest
        in retrieval's favour.
        """
        ticks = iter(
            [
                0.0,
                1.0,  # base
                1.0,
                8.0,  # oracle build, deliberately large
                10.0,
                11.0,  # oracle scoring, deliberately small
                11.0,
                12.0,  # bm25 index
                12.0,
                13.0,  # bm25 select
                13.0,
                14.0,  # bm25 scoring
                14.0,
                14.5,  # dense index
                14.0,
                15.0,  # dense select
                15.0,
                16.0,  # dense scoring
                16.0,
                17.0,  # fused select
                17.0,
                18.0,  # fused scoring
                20.0,
                21.0,
                30.0,
                31.0,
                40.0,
                41.0,
            ]
        )
        cli_hooks.monotonic_clock = lambda: next(ticks)
        try:
            observations, _digest = bench.measure_qa_plan(TINY_PLAN, corpus=tmp_path, device="cpu")
        finally:
            cli_hooks.monotonic_clock = cli_hooks._default_monotonic_clock

        named = {observation["name"]: observation["value"] for observation in observations}
        assert named["retrieval_serve_seconds"] == 1.0
        assert named["retrieval_oracle_build_seconds"] == 7.0
        # A single bracket from build-start to scoring-end would read 10.0.
        assert named["retrieval_serve_seconds"] != 10.0
