"""Ask whether a cartridge gives a model corpus knowledge it can USE.

THE MEASUREMENT THE LOSS BENCHMARK CANNOT MAKE.
:mod:`model_trainer.cli.cartridge_benchmark` reports held-out loss, and
:mod:`model_trainer.core.contracts.cloze` says why that is not enough: "a
model can memorise text word-by-word and still fail every question about it."
This runs a question set instead, built from text the cartridge never trained
on, and scores three arms on it: the model alone, the model with the prefix,
and the model with the evidence in its context window.

WHAT IT FOUND ON gpt2, and why both instruments are recorded. Over 24 items
from twelve wiki pages, the cartridge nearly halved the surprise on the
correct term -- 18.46 to 10.68 summed negative log-likelihood, better on 19 of
24 items, p = 0.0066 -- while multiple-choice ACCURACY did not move at all
(0.5417 to 0.5833, p = 1.0). Oracle retrieval answered every item. So the
prefix does carry usable corpus knowledge, and at this scale it raises the
likelihood of corpus vocabulary generally rather than sharpening the choice
between corpus terms.

AND THE ACCURACY ARM IS SENSITIVE TO SOMETHING THAT IS NOT THE MODEL. The
first item set built here repeated one distractor triple across nearly every
item; on that set the base model sat exactly at chance and the cartridge
looked significant at p = 0.006. Rotating distractors per item moved the base
to 0.5417 and the effect vanished. Same corpus, same items, same models. That
is why :func:`answer_nll_pairs` exists and why its numbers lead: scoring the
answer's own tokens has no distractor policy to be sensitive to.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.cloze import BLANK_MARKER
from model_trainer.core.contracts.replicated_measurement import (
    gain_observations,
    per_seed_observations,
    replicate,
)
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm.encoding import HFTokenizerEncoder
from model_trainer.core.services.model.cartridge_corpus import (
    build_windows,
    split_by_stride,
)
from model_trainer.core.services.model.cartridge_dense import dense_ranking, embed_chunks
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_qa import (
    answer_nll_pairs,
    bm25_retrieval_items,
    compare_arms,
    ranked_retrieval_items,
    retrieval_items,
)
from model_trainer.core.services.model.cartridge_qa_plans import (
    QA_EXPERIMENT,
    QaPlan,
    qa_plan_label,
)
from model_trainer.core.services.model.cartridge_question_set import build_question_set
from model_trainer.core.services.model.cartridge_retrieval import (
    RETRIEVED_CHUNKS,
    build_index,
    fuse_by_reciprocal_rank,
    rank_chunks,
)
from model_trainer.core.services.model.cloze.score import score_cloze_items
from model_trainer.core.services.model.control_arms import CONTROLS_FLAG, require_control_arm
from model_trainer.core.services.model.gemm_timing import synchroniser

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG, CONTROLS_FLAG)


def latency_observations(
    *,
    base_seconds: float,
    retrieval_seconds: float,
    cartridge_seconds: float,
    retrieval_build_seconds: float,
    real_seconds: float,
    real_select_seconds: float,
    real_index_seconds: float,
    dense_seconds: float,
    dense_select_seconds: float,
    dense_index_seconds: float,
    fused_seconds: float,
    fused_select_seconds: float,
) -> tuple[Observation, ...]:
    """Name what each arm cost to SERVE, per pass over the question set.

    WHAT IS BEING COMPARED, precisely, because the arms are not symmetric.
    All three run the same scorer over the same items; they differ only in
    what precedes the question -- nothing, retrieved evidence, or a trained
    prefix. So the difference between them is prefill, which is the thing a
    serving comparison is actually about: the retrieval arm re-encodes its
    evidence on every query, and the cartridge arm does not.

    THE ORACLE'S SELECTION IS MEASURED AND THEN EXCLUDED FROM THE COMPARISON,
    rather than quietly left out. ``retrieval_build_seconds`` is the time to
    pick each item's evidence by searching for its own ANSWER -- something no
    real retriever can do, so charging it to retrieval would invent a cost,
    and dropping it silently would hide that a step happened at all. It is
    recorded so a reader can see both the number and the argument.

    WHICH DIRECTION THIS BOUND CUTS. The oracle arm pays no embedding, no
    index search and no ranking, so it is the CHEAPEST any retrieval could
    be. A cartridge that beats it beats a real pipeline by more; a cartridge
    that loses to it has proven nothing about real pipelines. Only the first
    direction is conclusive, and the write-up has to say so.

    Args:
        base_seconds: Scoring the question set with no context added.
        retrieval_seconds: Scoring it with evidence in the prompt.
        cartridge_seconds: Scoring it behind a trained prefix, MEAN over the
            plan's seeds so it is one pass like the other two rather than a
            sum over however many seeds the plan happens to declare.
        retrieval_build_seconds: Assembling the oracle's evidence. Reported,
            not charged.
        real_seconds: Scoring behind BM25-retrieved evidence.
        real_select_seconds: Querying the index. CHARGED, unlike the oracle's
            selection, because searching from the question is work every
            deployment does per request. ``bm25_total_serve_seconds`` is the
            sum, and it is the number to compare against the cartridge.
        real_index_seconds: Building the index. Reported separately and NOT
            in the total: a deployment pays it once when its corpus changes,
            so charging it per query would overstate retrieval exactly as
            charging the oracle's cheating would.
        dense_seconds: Scoring behind embedding-retrieved evidence.
        dense_select_seconds: Embedding the QUERY and ranking pre-embedded
            chunks against it. Charged, for the reason BM25's select is.
        dense_index_seconds: Embedding the corpus. Offline and excluded from
            the total, symmetric with ``real_index_seconds`` -- and the
            asymmetry the first version of this got wrong, by embedding the
            corpus inside every query and recording 17452 ms/item.
        fused_seconds: Scoring behind reciprocal-rank-fused evidence.
        fused_select_seconds: Fusing, plus the lexical ranking fusion needs.
            The dense ranking is an INPUT to it, so the fused total adds
            ``dense_select_seconds`` as well -- a hybrid cannot cost less
            than an arm it is built on.

    Returns:
        The named durations. Every arm's per-request total is present as its
        own name, so a reader compares totals without re-deriving which
        components belong to which arm.
    """
    return tuple(
        Observation(name=name, value=value)
        for name, value in (
            ("base_serve_seconds", base_seconds),
            ("retrieval_serve_seconds", retrieval_seconds),
            ("cartridge_serve_seconds", cartridge_seconds),
            ("retrieval_oracle_build_seconds", retrieval_build_seconds),
            ("bm25_serve_seconds", real_seconds),
            ("bm25_select_seconds", real_select_seconds),
            ("bm25_total_serve_seconds", real_seconds + real_select_seconds),
            ("bm25_index_seconds", real_index_seconds),
            ("dense_serve_seconds", dense_seconds),
            ("dense_select_seconds", dense_select_seconds),
            ("dense_total_serve_seconds", dense_seconds + dense_select_seconds),
            ("dense_index_seconds", dense_index_seconds),
            ("fused_serve_seconds", fused_seconds),
            ("fused_select_seconds", fused_select_seconds),
            (
                "fused_total_serve_seconds",
                fused_seconds + fused_select_seconds + dense_select_seconds,
            ),
        )
    )


def measure_qa_plan(
    plan: QaPlan, *, corpus: pathlib.Path, device: str
) -> tuple[tuple[Observation, ...], str]:
    """Run every arm of one question-set plan and name what they produced.

    THE BASE AND RETRIEVAL ARMS ARE SCORED ONCE, not once per seed. Neither
    carries a cartridge, so neither depends on the initialisation seed;
    running them three times would spend three times the compute to produce
    the same number and would report a spread of zero as if it were measured.

    Args:
        plan: The measurement to run.
        corpus: Directory of markdown documents.
        device: Device to measure on.

    Returns:
        ``(observations, digest)``.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when the corpus yields no
            question set, ``CLOZE_ITEM_UNSCOREABLE`` when an item cannot carry
            evidence, or ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` when the plan
            names too few seeds.
    """
    documents = _test_hooks.read_corpus_documents(corpus)
    digest = corpus_digest(documents)
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(plan["model_id"])
    encoder = HFTokenizerEncoder(tokenizer)
    encoded = [tokenizer.encode(document) for document in documents]

    items, training_text = build_question_set(documents, encoded, encoder, plan)
    chance = 1.0 / float(plan["distractor_count"] + 1)
    _log.info("built %d items over %d documents, chance %.4f", len(items), len(documents), chance)

    windows = build_windows(encoded, window=plan["window"], device=device)
    train, _held = split_by_stride(windows, held_out_stride=plan["held_out_stride"])

    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(plan["model_id"], None))
    base.to(device)
    max_seq = plan["max_seq_len"]

    # Every timed boundary waits first, for the reason `gemm_timing`'s
    # docstring gives: a CUDA launch is asynchronous, so an unwaited clock
    # read measures how long it took to QUEUE the work rather than to do it.
    wait = synchroniser(device)
    clock = _test_hooks.monotonic_clock

    # ONE DISCARDED PASS BEFORE ANY CLOCK STARTS, and it is not politeness.
    # Without it every one-time cost -- cuDNN autotuning, kernel selection,
    # the caching allocator's first big reservation -- lands on whichever arm
    # runs first, which is `base`. The first run measured that way reported
    # the CARTRIDGE arm as faster than the base it wraps (1.733s against
    # 2.325s), which cannot happen: a cartridge runs the same model over the
    # same items with 128 extra prefix positions, so it is strictly more
    # work. The impossible number was the warmup landing on base and nothing
    # else. `gemm_timing` has carried WARMUP for the same reason all along;
    # this module took its synchroniser and left its warmup behind.
    score_cloze_items(items=items, model=base, encoder=encoder, device=device, max_seq_len=max_seq)

    wait()
    started = clock()
    scored_base = score_cloze_items(
        items=items, model=base, encoder=encoder, device=device, max_seq_len=max_seq
    )
    wait()
    base_seconds = clock() - started

    # The oracle's SELECTION is timed apart from the scoring it feeds. It
    # searches each item's own answer, which no real retriever can do, so its
    # cost belongs in the record but not in the comparison.
    started = clock()
    retrieval_set = retrieval_items(items, [training_text], encoder, max_seq_len=max_seq)
    retrieval_build_seconds = clock() - started

    wait()
    started = clock()
    scored_retrieval = score_cloze_items(
        items=retrieval_set,
        model=base,
        encoder=encoder,
        device=device,
        max_seq_len=max_seq,
    )
    wait()
    retrieval_seconds = clock() - started

    # THE REAL ARM. Indexing is timed apart from querying because a
    # deployment pays them at different times -- the index is built once when
    # the corpus changes, the query runs per request. Unlike the oracle's
    # selection, the query time here IS chargeable: searching an index from
    # the question is work every real retriever does.
    started = clock()
    index = build_index([training_text])
    real_index_seconds = clock() - started

    started = clock()
    real_set = bm25_retrieval_items(items, index, encoder, max_seq_len=max_seq)
    real_select_seconds = clock() - started

    wait()
    started = clock()
    scored_real = score_cloze_items(
        items=real_set,
        model=base,
        encoder=encoder,
        device=device,
        max_seq_len=max_seq,
    )
    wait()
    real_seconds = clock() - started
    _log.info("bm25 retrieval %.4f over %d chunks", scored_real["accuracy"], len(index["chunks"]))

    # THE DENSE ARM, and the FUSION of it with BM25. All three rank the same
    # chunks, so they differ only in HOW they choose -- which is the
    # comparison worth making. Queries strip the blank marker for the reason
    # `bm25_retrieval_items` documents.
    queries = [item["template"].replace(BLANK_MARKER, " ") for item in items]

    # OFFLINE, exactly as the BM25 index build is. The first version of this
    # embedded the whole corpus inside every query and recorded 17452 ms/item
    # against BM25's 72 -- real arithmetic over a design nobody deploys.
    started = clock()
    dense_vectors = embed_chunks(index, _test_hooks.embed_texts)
    dense_index_seconds = clock() - started

    started = clock()
    dense_ranks = [
        dense_ranking(dense_vectors, query, _test_hooks.embed_texts) for query in queries
    ]
    dense_select_seconds = clock() - started

    dense_set = ranked_retrieval_items(
        items,
        index,
        encoder,
        [ranking[:RETRIEVED_CHUNKS] for ranking in dense_ranks],
        max_seq_len=max_seq,
    )
    wait()
    started = clock()
    scored_dense = score_cloze_items(
        items=dense_set, model=base, encoder=encoder, device=device, max_seq_len=max_seq
    )
    wait()
    dense_seconds = clock() - started

    # Fusion re-uses the dense ranking rather than recomputing it, so this
    # times the FUSION plus the lexical ranking it still needs. A deployment
    # pays the dense arm on top; the record carries the numbers separately
    # so a reader can add whichever total they mean.
    started = clock()
    fused_ranks = [
        fuse_by_reciprocal_rank(ranking, rank_chunks(index, query), limit=RETRIEVED_CHUNKS)
        for ranking, query in zip(dense_ranks, queries, strict=True)
    ]
    fused_select_seconds = clock() - started

    fused_set = ranked_retrieval_items(items, index, encoder, fused_ranks, max_seq_len=max_seq)
    wait()
    started = clock()
    scored_fused = score_cloze_items(
        items=fused_set, model=base, encoder=encoder, device=device, max_seq_len=max_seq
    )
    wait()
    fused_seconds = clock() - started
    _log.info("dense %.4f, fused %.4f", scored_dense["accuracy"], scored_fused["accuracy"])
    _log.info("base %.4f, retrieval %.4f", scored_base["accuracy"], scored_retrieval["accuracy"])

    accuracy_gains: list[tuple[int, float]] = []
    nll_gains: list[tuple[int, float]] = []
    cartridge_seconds_total = 0.0
    for seed in plan["seeds"]:
        slots = train_cartridge(
            base,
            train,
            num_slots=plan["num_slots"],
            seed=seed,
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
        )
        cartridge = CartridgeModel(base=base, slots=slots)
        # Scoring only. Training the prefix is a ONE-TIME cost that the
        # capacity benchmark already records; charging it to serving would
        # compare a cartridge's whole life against retrieval's per-query.
        wait()
        started = clock()
        scored = score_cloze_items(
            items=items, model=cartridge, encoder=encoder, device=device, max_seq_len=max_seq
        )
        wait()
        cartridge_seconds_total += clock() - started
        nll = answer_nll_pairs(items, base, cartridge, encoder, device=device, max_seq_len=max_seq)
        accuracy_gains.append((seed, scored["accuracy"] - scored_base["accuracy"]))
        nll_gains.append((seed, nll["mean_baseline"] - nll["mean_treatment"]))
        _log.info(
            "seed %d: accuracy %.4f, answer-nll %.4f -> %.4f (p=%.6f)",
            seed,
            scored["accuracy"],
            nll["mean_baseline"],
            nll["mean_treatment"],
            nll["p_value"],
        )

    retrieval_pair = compare_arms(scored_base, scored_retrieval)
    observations: list[Observation] = [
        Observation(name="items", value=float(len(items))),
        Observation(name="chance_accuracy", value=chance),
        Observation(name="base_accuracy", value=scored_base["accuracy"]),
        Observation(name="retrieval_accuracy", value=scored_retrieval["accuracy"]),
        Observation(
            name="retrieval_accuracy_gain",
            value=scored_retrieval["accuracy"] - scored_base["accuracy"],
        ),
        Observation(name="base_to_retrieval_p_value", value=retrieval_pair["p_value"]),
        Observation(name="dense_accuracy", value=scored_dense["accuracy"]),
        Observation(name="fused_accuracy", value=scored_fused["accuracy"]),
        Observation(name="bm25_accuracy", value=scored_real["accuracy"]),
        Observation(
            name="bm25_accuracy_gain",
            value=scored_real["accuracy"] - scored_base["accuracy"],
        ),
        Observation(
            name="base_to_bm25_p_value",
            value=compare_arms(scored_base, scored_real)["p_value"],
        ),
        # The gap the oracle arm exists to bound: how much of retrieval's
        # advantage is the retriever finding the right sentences, and how
        # much is knowing the answer outright.
        Observation(
            name="oracle_over_bm25_accuracy",
            value=scored_retrieval["accuracy"] - scored_real["accuracy"],
        ),
    ]
    # Replicated once and reused, so the summary and the per-seed rows
    # describe the same arm rather than two independent reductions of it.
    accuracy_arm = replicate("cartridge-accuracy-gain", accuracy_gains)
    nll_arm = replicate("cartridge-answer-nll-gain", nll_gains)
    for arm in (accuracy_arm, nll_arm):
        observations.extend(gain_observations(arm))
        observations.extend(per_seed_observations(arm))
    observations.extend(
        latency_observations(
            base_seconds=base_seconds,
            retrieval_seconds=retrieval_seconds,
            cartridge_seconds=cartridge_seconds_total / float(len(plan["seeds"])),
            retrieval_build_seconds=retrieval_build_seconds,
            real_seconds=real_seconds,
            real_select_seconds=real_select_seconds,
            real_index_seconds=real_index_seconds,
            dense_seconds=dense_seconds,
            dense_select_seconds=dense_select_seconds,
            dense_index_seconds=dense_index_seconds,
            fused_seconds=fused_seconds,
            fused_select_seconds=fused_select_seconds,
        )
    )
    return tuple(observations), digest


def qa_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    device: str,
    remove_split_k: bool,
    math_attention: bool,
) -> RunRecord:
    """Pin determinism, run every arm, and record it.

    The posture is an argument for the reason set out in
    :func:`~model_trainer.cli.cartridge_benchmark.cartridge_run_record`: this
    command could otherwise only ever observe the untreated arm, and the
    question the controls exist to answer is a comparison between arms.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents.
        device: Device to measure on.
        remove_split_k: Whether to take split-K out of cuBLASLt's options.
        math_attention: Whether to restrict attention to the math kernel.

    Returns:
        The record, its fingerprint carrying whichever controls were applied.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        AppError: Propagated from the arms.
    """
    plan = require_cartridge_plan(_measurement_hooks.qa_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=remove_split_k, math_attention=math_attention),
    )
    observations, digest = measure_qa_plan(plan, corpus=corpus, device=device)
    return run_record(
        experiment=QA_EXPERIMENT,
        label=qa_plan_label(plan_name, plan, digest=digest),
        fingerprint=fingerprint,
        observations=observations,
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one question-set plan and write the record.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent.
        KeyError: If the plan name is unknown.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    remove_split_k, math_attention = require_control_arm(
        cli_args.require_flag(parsed, CONTROLS_FLAG)
    )

    record = qa_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        remove_split_k=remove_split_k,
        math_attention=math_attention,
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "cartridge question set %s %s -> %s",
        record["label"],
        describe_run_fingerprint(record["fingerprint"]),
        out,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="cartridge-qa-benchmark",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "HFTokenizerEncoder",
    "build_question_set",
    "entrypoint",
    "latency_observations",
    "main",
    "measure_qa_plan",
    "qa_run_record",
]


# Without this, `python -m model_trainer.cli.cartridge_qa_benchmark` imports
# the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
