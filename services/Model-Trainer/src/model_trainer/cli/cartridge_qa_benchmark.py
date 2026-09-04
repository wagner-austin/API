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
from model_trainer.core.contracts.cloze import ClozeItem
from model_trainer.core.contracts.replicated_measurement import (
    gain_observations,
    replicate,
)
from model_trainer.core.encoding import Encoder
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
    window_documents,
)
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_qa import (
    answer_nll_pairs,
    compare_arms,
    retrieval_items,
)
from model_trainer.core.services.model.cartridge_qa_plans import (
    QA_EXPERIMENT,
    QaPlan,
    qa_plan_label,
)
from model_trainer.core.services.model.cloze.score import score_cloze_items
from model_trainer.core.services.model.corpus_cloze import build_items

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG)


def build_question_set(
    documents: Sequence[str],
    encoded: Sequence[Sequence[int]],
    encoder: Encoder,
    plan: QaPlan,
) -> tuple[list[ClozeItem], str]:
    """Split a corpus and build items from the half the cartridge will not read.

    The split is by window, not by document, and that is what makes the set
    answerable. Pages here are about different projects, so a document-level
    split would leave held-out terms that appear nowhere in the training text
    and every item would be unanswerable from the corpus. Splitting within
    each page keeps a term learnable from the windows the cartridge trains on
    while testing it in a sentence those windows do not contain.

    The window text is recovered by DECODING the ids rather than by slicing
    the document string, because the split is defined on tokens and a
    character offset cannot name a token boundary.

    Args:
        documents: Document bodies, in the order they were encoded.
        encoded: Token ids for each document.
        encoder: Tokenizer, used to read each window's text back.
        plan: The measurement being run.

    Returns:
        ``(items, training_text)``. A plain pair rather than a named record:
        the two have different types, so nothing can transpose them, and a
        class holding them would carry no behaviour of its own.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when the corpus cannot
            supply windows, a split, or items.
    """
    owners = window_documents(encoded, window=plan["window"])
    stride = plan["held_out_stride"]
    held_by_document: dict[int, list[str]] = {}
    training: list[str] = []
    seen_per_document: dict[int, int] = {}
    for index, owner in enumerate(owners):
        start = seen_per_document.get(owner, 0)
        seen_per_document[owner] = start + 1
        window = plan["window"]
        text = encoder.decode(list(encoded[owner])[start * window : (start + 1) * window])
        if index % stride == 0:
            held_by_document.setdefault(owner, []).append(text)
        else:
            training.append(text)
    held_documents = [
        " ".join(held_by_document.get(document, [])) for document in range(len(documents))
    ]
    training_text = " ".join(training)
    return (
        build_items(
            held_documents,
            training_text,
            distractor_count=plan["distractor_count"],
            max_items=plan["max_items"],
        ),
        training_text,
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

    scored_base = score_cloze_items(
        items=items, model=base, encoder=encoder, device=device, max_seq_len=max_seq
    )
    scored_retrieval = score_cloze_items(
        items=retrieval_items(items, [training_text], encoder, max_seq_len=max_seq),
        model=base,
        encoder=encoder,
        device=device,
        max_seq_len=max_seq,
    )
    _log.info("base %.4f, retrieval %.4f", scored_base["accuracy"], scored_retrieval["accuracy"])

    accuracy_gains: list[tuple[int, float]] = []
    nll_gains: list[tuple[int, float]] = []
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
        scored = score_cloze_items(
            items=items, model=cartridge, encoder=encoder, device=device, max_seq_len=max_seq
        )
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
    ]
    observations.extend(gain_observations(replicate("cartridge-accuracy-gain", accuracy_gains)))
    observations.extend(gain_observations(replicate("cartridge-answer-nll-gain", nll_gains)))
    return tuple(observations), digest


def qa_run_record(plan_name: str, *, corpus: pathlib.Path, device: str) -> RunRecord:
    """Pin determinism, run every arm, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents.
        device: Device to measure on.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        AppError: Propagated from the arms.
    """
    plan = require_cartridge_plan(_measurement_hooks.qa_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
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

    record = qa_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
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
    "main",
    "measure_qa_plan",
    "qa_run_record",
]


# Without this, `python -m model_trainer.cli.cartridge_qa_benchmark` imports
# the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
