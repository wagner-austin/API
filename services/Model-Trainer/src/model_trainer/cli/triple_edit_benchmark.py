"""Run the corpus-representation arm and write its record.

Board task 3fc98ed6, criteria 3 and 4. The arm's reasoning lives in
:mod:`~model_trainer.core.services.model.editing.triple_edit_arm`; this module
is the entry: resolve a plan, build the same question set the cartridge arm
was scored on, hand both to the arm, and record what came back under a
fingerprint that says which kernel posture it ran under.

THE CORPUS DIGEST IS CHECKED, NOT ASSUMED. The curated triples are grounded in
one exact text, so a run against a different corpus would report a reject rate
about a corpus nobody curated for. The digest is compared before any model is
loaded and the run is refused rather than annotated.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import RunRecord, encode_run_record, run_record

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm.encoding import HFTokenizerEncoder
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_question_set import build_question_set
from model_trainer.core.services.model.cloze.identity import question_set_digest
from model_trainer.core.services.model.control_arms import CONTROLS_FLAG, require_control_arm
from model_trainer.core.services.model.editing.sites import require_traceable_model
from model_trainer.core.services.model.editing.triple_edit_arm import (
    run_triple_edit_arm,
    triple_edit_observations,
)
from model_trainer.core.services.model.editing.triple_edit_plans import (
    TRIPLE_EDIT_EXPERIMENT,
    TripleEditPlan,
    triple_edit_plan_label,
)

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG, CONTROLS_FLAG)


def triple_edit_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    device: str,
    remove_split_k: bool,
    math_attention: bool,
) -> RunRecord:
    """Pin determinism, run the arm, and record it.

    Args:
        plan_name: Which triple-edit plan to run.
        corpus: Directory of markdown documents.
        device: Device to measure on.
        remove_split_k: Whether to take split-K out of cuBLASLt's options.
        math_attention: Whether to restrict attention to the math kernel.

    Returns:
        The record, its payload digest naming the exact question set asked.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` when the corpus is not
            the one the triples were curated against, or propagated from the
            arm.
    """
    plan: TripleEditPlan = require_cartridge_plan(_measurement_hooks.triple_edit_plans(), plan_name)
    qa_plan = require_cartridge_plan(_measurement_hooks.qa_plans(), plan["qa_plan"])

    documents = _test_hooks.read_corpus_documents(corpus)
    digest = corpus_digest(documents)
    if digest != plan["corpus_digest"]:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE,
            (
                f"plan {plan_name!r} was curated against corpus "
                f"{plan['corpus_digest'][:12]} and this one is {digest[:12]}; a reject "
                f"rate measured here would be about a corpus nobody curated for"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE),
        )

    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=remove_split_k, math_attention=math_attention),
    )
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(qa_plan["model_id"])
    encoder = HFTokenizerEncoder(tokenizer)
    encoded = [tokenizer.encode(document) for document in documents]
    items, training_text = build_question_set(documents, encoded, encoder, qa_plan)
    _log.info("built %d items over %d documents", len(items), len(documents))

    model = require_traceable_model(hf_hooks.Hooks.load_hf_model(qa_plan["model_id"], None))
    model.to(device)
    outcome = run_triple_edit_arm(
        model=model,
        encoder=encoder,
        items=items,
        candidates=plan["candidates"],
        training_text=training_text,
        site=plan["site"],
        value_steps=plan["value_steps"],
        value_learning_rate=plan["value_learning_rate"],
        device=device,
        max_seq_len=qa_plan["max_seq_len"],
    )
    return run_record(
        experiment=TRIPLE_EDIT_EXPERIMENT,
        label=triple_edit_plan_label(plan_name, plan, digest=digest),
        fingerprint=fingerprint,
        observations=triple_edit_observations(outcome),
        payload_digest=question_set_digest(items),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one triple-edit plan and write the record.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent.
        KeyError: If the plan name is unknown.
        AppError: Propagated from the arm.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    remove_split_k, math_attention = require_control_arm(
        cli_args.require_flag(parsed, CONTROLS_FLAG)
    )
    record = triple_edit_run_record(
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
        "corpus representation %s %s -> %s",
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
        service_name="triple-edit-benchmark",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "entrypoint",
    "main",
    "triple_edit_run_record",
]


# Without this, `python -m model_trainer.cli.triple_edit_benchmark` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
