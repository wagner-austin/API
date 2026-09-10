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
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.qa_checkpoint import SeedRecord, with_seed
from model_trainer.core.contracts.qa_plan import QA_EXPERIMENT, QaPlan
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
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_qa import answer_nll_pairs, compare_arms
from model_trainer.core.services.model.cartridge_qa_checkpoint import (
    delete_qa_checkpoint,
    resume_or_start,
    save_qa_checkpoint,
)
from model_trainer.core.services.model.cartridge_qa_power import require_resolvable_question_set
from model_trainer.core.services.model.cartridge_qa_report import (
    ArmScores,
    QaMeasurement,
    accuracy_observations,
    latency_observations,
)
from model_trainer.core.services.model.cartridge_qa_retrieval_arms import score_retrieval_arms
from model_trainer.core.services.model.cartridge_question_set import build_question_set
from model_trainer.core.services.model.cloze.identity import qa_plan_label, question_set_digest
from model_trainer.core.services.model.cloze.score import score_cloze_items, scored_and_timed
from model_trainer.core.services.model.control_arms import CONTROLS_FLAG, require_control_arm
from model_trainer.core.services.model.gemm_timing import synchroniser

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG, CONTROLS_FLAG)


def measure_qa_plan(
    plan_name: str,
    plan: QaPlan,
    *,
    corpus: pathlib.Path,
    device: str,
    checkpoints: pathlib.Path,
) -> QaMeasurement:
    """Run every arm of one question-set plan and name what they produced.

    THE BASE AND RETRIEVAL ARMS ARE SCORED ONCE, not once per seed. Neither
    carries a cartridge, so neither depends on the initialisation seed;
    running them three times would spend three times the compute to produce
    the same number and would report a spread of zero as if it were measured.

    RESUMABLE AT THE ARM AND THE SEED. ``free-gpu`` cancels an evicted job
    outright and Slurm resubmits nothing, so a run this long that kept no
    checkpoint would lose everything it had done. Each retrieval arm and each
    cartridge seed is recorded the moment it finishes; a restart skips what is
    already there and re-derives only the shared structure the later arms need.
    The checkpoint is deleted on success, because a leftover file is
    indistinguishable from an interrupted run and the NEXT submission of this
    plan would skip arms it should have re-measured.

    Args:
        plan_name: Which plan this is, as named in ``QA_PLANS``. Part of the
            checkpoint's identity, so a resume cannot adopt another plan's
            arms, and it is not derivable from the plan itself.
        plan: The measurement to run.
        corpus: Directory of markdown documents.
        device: Device to measure on.
        checkpoints: Directory holding this plan's checkpoint.

    Returns:
        The numbers and both identities of the run that produced them.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when the corpus yields no
            question set, ``CLOZE_ITEM_UNSCOREABLE`` when an item cannot carry
            evidence, ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` when the plan
            names too few seeds, or ``CARTRIDGE_CHECKPOINT_FOREIGN`` when a
            checkpoint on disk describes a different measurement.
    """
    documents = _test_hooks.read_corpus_documents(corpus)
    digest = corpus_digest(documents)
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(plan["model_id"])
    encoder = HFTokenizerEncoder(tokenizer)
    encoded = [tokenizer.encode(document) for document in documents]

    items, training_text = build_question_set(documents, encoded, encoder, plan)
    chance = 1.0 / float(plan["distractor_count"] + 1)
    _log.info("built %d items over %d documents, chance %.4f", len(items), len(documents), chance)

    # BEFORE THE MODEL LOADS, WHICH IS THE POINT. Everything below this line
    # costs GPU hours, and a question set too small to resolve what the plan
    # declares will still produce a full arms table at the end of them --
    # which is how a difference of 1.3 items over 32 reached a wiki hub and
    # had to be retracted. The refusal is here, against the REALISED item
    # count, because `max_items` is a cap the corpus is free to fall short of.
    floor = require_resolvable_question_set(plan, len(items))
    _log.info(
        "question set resolves nothing smaller than %.4f; plan declares %.4f",
        floor,
        plan["smallest_effect_of_interest"],
    )

    # AFTER THE POWER GATE, BEFORE ANY SCORING. A checkpoint is only valid for
    # the measurement that wrote it, and two of the four things that identify
    # one -- the realised item count and the corpus digest -- are not known
    # until the question set is built. Resuming earlier would mean checking a
    # fingerprint against values nobody had yet.
    checkpoint = resume_or_start(
        checkpoints,
        plan_name=plan_name,
        corpus_digest=digest,
        item_count=len(items),
        model_id=plan["model_id"],
    )

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

    scored_base, base_seconds = scored_and_timed(
        items, base, encoder, device=device, max_seq_len=max_seq, wait=wait, clock=clock
    )

    # THE ARMS THEMSELVES LIVE IN `cartridge_qa_retrieval_arms`. What is an
    # arm, what is timed apart from what, and which cost a deployment pays
    # are measurement questions and belong beside the arms rather than beside
    # this module's argument parsing. What stays here is the wiring: corpus,
    # power gate, base model, cartridge seeds, and the record.
    checkpoint, arms = score_retrieval_arms(
        items,
        plan=plan,
        training_text=training_text,
        base=base,
        encoder=encoder,
        device=device,
        wait=wait,
        clock=clock,
        make_embedder=_test_hooks.make_embedder,
        checkpoint=checkpoint,
        checkpoints=checkpoints,
    )
    scored_retrieval = arms["oracle"]
    scored_real = arms["bm25"]
    scored_dense = arms["dense"]
    scored_fused = arms["fused"]
    scored_expanded = arms["expanded"]
    scored_reranked = arms["reranked"]
    scored_long_context = arms["long_context"]
    long_context_fraction = arms["long_context_corpus_fraction"]

    _log.info("base %.4f, retrieval %.4f", scored_base["accuracy"], scored_retrieval["accuracy"])

    accuracy_gains: list[tuple[int, float]] = []
    nll_gains: list[tuple[int, float]] = []
    cartridge_seconds_total = 0.0
    completed = {record["seed"]: record for record in checkpoint["seeds"]}
    for seed in plan["seeds"]:
        # THE MOST EXPENSIVE UNIT IN THE RUN, which is why it is checkpointed
        # at all: every seed TRAINS a cartridge over the whole corpus before
        # it scores anything, so a seed lost to an eviction costs more than
        # any retrieval arm. Training is skipped entirely on resume -- unlike
        # the arms, nothing later needs a cartridge that has already reported.
        found = completed.get(seed)
        if found is None:
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
            seed_seconds = clock() - started
            nll = answer_nll_pairs(
                items, base, cartridge, encoder, device=device, max_seq_len=max_seq
            )
            found = SeedRecord(seed=seed, result=scored, answer_nll=nll, score_seconds=seed_seconds)
            checkpoint = with_seed(checkpoint, found)
            save_qa_checkpoint(checkpoints, checkpoint)
        else:
            _log.info("resuming seed %d from checkpoint, no cartridge retrained", seed)
        cartridge_seconds_total += found["score_seconds"]
        accuracy_gains.append((seed, found["result"]["accuracy"] - scored_base["accuracy"]))
        nll_gains.append(
            (seed, found["answer_nll"]["mean_baseline"] - found["answer_nll"]["mean_treatment"])
        )
        _log.info(
            "seed %d: accuracy %.4f, answer-nll %.4f -> %.4f (p=%.6f)",
            seed,
            found["result"]["accuracy"],
            found["answer_nll"]["mean_baseline"],
            found["answer_nll"]["mean_treatment"],
            found["answer_nll"]["p_value"],
        )

    retrieval_pair = compare_arms(scored_base, scored_retrieval)
    observations = accuracy_observations(
        ArmScores(
            base=scored_base,
            oracle=scored_retrieval,
            bm25=scored_real,
            dense=scored_dense,
            fused=scored_fused,
            expanded=scored_expanded,
            reranked=scored_reranked,
            long_context=scored_long_context,
        ),
        items=len(items),
        chance=chance,
        long_context_corpus_fraction=long_context_fraction,
        reranked_seconds=arms["reranked_seconds"],
        expanded_seconds=arms["expanded_seconds"],
        long_context_seconds=arms["long_context_seconds"],
    )
    observations.append(
        Observation(name="base_to_retrieval_p_value", value=retrieval_pair["p_value"])
    )
    observations.append(
        Observation(
            name="base_to_bm25_p_value",
            value=compare_arms(scored_base, scored_real)["p_value"],
        )
    )
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
            retrieval_seconds=arms["retrieval_seconds"],
            cartridge_seconds=cartridge_seconds_total / float(len(plan["seeds"])),
            retrieval_build_seconds=arms["retrieval_build_seconds"],
            real_seconds=arms["real_seconds"],
            real_select_seconds=arms["real_select_seconds"],
            real_index_seconds=arms["real_index_seconds"],
            dense_seconds=arms["dense_seconds"],
            dense_select_seconds=arms["dense_select_seconds"],
            dense_index_seconds=arms["dense_index_seconds"],
            fused_seconds=arms["fused_seconds"],
            fused_select_seconds=arms["fused_select_seconds"],
        )
    )
    # A COMPLETED RUN DELETES ITS OWN CHECKPOINT. A leftover file is
    # indistinguishable from an interrupted run, so the next submission of
    # this plan would skip arms it should have re-measured and report numbers
    # from a previous execution as if this one had produced them. Deleted
    # here rather than by the caller, because the caller cannot tell whether
    # the measurement finished.
    delete_qa_checkpoint(checkpoints, plan_name)

    return QaMeasurement(
        observations=tuple(observations),
        corpus_digest=digest,
        question_set_digest=question_set_digest(items),
    )


def qa_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    device: str,
    checkpoints: pathlib.Path,
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
        checkpoints: Directory holding this plan's checkpoint, so an evicted
            run resumes at its last arm or seed rather than at zero.
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
    measured = measure_qa_plan(
        plan_name, plan, corpus=corpus, device=device, checkpoints=checkpoints
    )
    return run_record(
        experiment=QA_EXPERIMENT,
        label=qa_plan_label(plan_name, plan, digest=measured["corpus_digest"]),
        fingerprint=fingerprint,
        observations=measured["observations"],
        # THE QUESTION SET, not the corpus, and the two are not the same
        # identity. `cloze.identity` carries the two records that proved it.
        payload_digest=measured["question_set_digest"],
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

    # RESOLVED BEFORE THE RUN, not after it. The checkpoint directory derives
    # from where the artifact goes, so `--out` has to be known before any
    # scoring starts. A DERIVED path rather than a flag of its own: a second
    # flag could disagree with this one, and then a resumed run would look for
    # its own work in a directory nothing wrote to.
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = qa_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        checkpoints=out.parent / "checkpoints",
        remove_split_k=remove_split_k,
        math_attention=math_attention,
    )

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
    "QaMeasurement",
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
