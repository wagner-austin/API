"""Generate one arm of a whole-file continuation sweep, and record what produced it.

WHY THIS IS NOT PART OF THE SCORER. ``code-style-eval`` scores files that
already exist, so the same instrument scores a base model, a trained adapter,
or a person writing by hand, and a sweep can be re-scored after a guard rule
changes without regenerating anything. It is CPU work that runs against the
repository's own checkers. This is GPU work against pinned weights. They meet
at a directory of files and a manifest, and the shape of that meeting is
:mod:`platform_core.continuation_task`, which both import and neither owns.

WHY IT LIVES HERE rather than beside the scorer. Reloading a saved adapter
onto the base it was trained against, under the quantization it was trained
under, is :func:`load_prepared_hf_lm_from_handle` and has been for months. A
sweep that re-implemented it would be a second loader, and the two would
agree until the day one of them changed.

WHAT IT LEAVES BEHIND is a directory of generated files, a manifest saying
which completions ended on their own, and a
:class:`~platform_core.run_record.RunRecord`. The record is the half the
first version of this sweep did not have: the generation configuration was
covered only indirectly, through a digest of the outcome files, so nothing
could say which decoding parameters or which weights produced them.

Usage:
    modeltrainer-continuations --spec SPEC.json --out-dir DIR --record REC.json
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.continuation_task import (
    EvalPrompt,
    GenerationEntry,
    TokenCounter,
    batches,
    build_prompts,
    decode_generation_entry,
    encode_generation_entry,
    finishable,
    generated_path,
    manifest_path,
)
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.logging import get_logger
from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record
from typing_extensions import TypedDict

from model_trainer.cli import _test_hooks
from model_trainer.core.contracts.continuation_sweep import (
    Completion,
    ContinuationSweepSpec,
    decode_continuation_sweep_spec,
)
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.run_fingerprint import capture_run_fingerprint
from model_trainer.core.services.model.backends.hf_lm.io import read_hf_lm_metadata
from model_trainer.core.services.training.run_records import model_distributions

_log = get_logger(__name__)

SPEC_FLAG = "--spec"
OUT_DIR_FLAG = "--out-dir"
RECORD_FLAG = "--record"

_FLAGS = (SPEC_FLAG, OUT_DIR_FLAG, RECORD_FLAG)

PARTIAL_SUFFIX = ".partial"
"""What a file being written is called until it is complete.

The resume check reads a file's EXISTENCE as proof it finished, so a process
killed mid-write would otherwise leave a truncated completion that every
later run trusts and every later score reports as the model's work.
"""

CONTINUATION_EXPERIMENT = "model-trainer-continuations"
"""Name under which two continuation sweeps are comparable with each other."""


def load_spec(path: pathlib.Path) -> ContinuationSweepSpec:
    """Read one sweep document off disk.

    Args:
        path: The document.

    Returns:
        The validated spec.

    Raises:
        JSONTypeError: If the file does not hold a JSON object, or holds one
            this cannot read.
    """
    return decode_continuation_sweep_spec(
        narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))
    )


def load_arm(spec: ContinuationSweepSpec) -> PreparedLMModel:
    """Load the weights this arm generates from.

    Args:
        spec: The sweep document.

    Returns:
        The prepared model.
    """
    return _test_hooks.load_continuation_arm(spec["artifact_path"], spec["arm"])


class SweepScope(TypedDict):
    """How much of a holdout one arm is answerable for.

    Attributes:
        built: How many held-out files are continuation tasks at all.
        prompts: The subset whose reference fits the token budget, in corpus
            order. Both arms compute the same subset, because both measure
            the same references with the tokenizer of the same base model.
    """

    built: int
    prompts: list[EvalPrompt]


def token_counter(model: PreparedLMModel) -> TokenCounter:
    """Measure text with one arm's own tokenizer.

    Defined once and used for both the budget filter and the batch sort. Two
    spellings of "how long is this" would be two ways for the arms to
    disagree about which items are in scope and about who shares a batch with
    whom -- and either one silently unpairs the comparison.

    Args:
        model: The loaded arm.

    Returns:
        A counter over that arm's tokenizer.
    """

    def measure(text: str) -> int:
        """Count a string in tokens.

        Args:
            text: The string.

        Returns:
            Its token count.
        """
        return len(model.tok_for_dataset.encode(text).ids)

    return measure


def sweep_prompts(spec: ContinuationSweepSpec, model: PreparedLMModel) -> SweepScope:
    """Build every prompt this arm is answerable for.

    Args:
        spec: The sweep document.
        model: The loaded arm, whose tokenizer measures the references.

    Returns:
        How many tasks the holdout holds, and which of them are in scope.

    Raises:
        MalformedRecordError: If the holdout carries a line that is not a
            usable document record.
    """
    records = pathlib.Path(spec["holdout_path"]).read_text(encoding="utf-8").splitlines()
    built = build_prompts(records, spec["prompt_lines"])
    return SweepScope(
        built=len(built),
        prompts=finishable(built, token_counter(model), spec["max_new_tokens"]),
    )


def write_completion(out_dir: pathlib.Path, completion: Completion) -> None:
    """Put one generated file where the scorer will look for it.

    Written aside and moved into place, so a file that exists is a file that
    finished.

    Args:
        out_dir: Directory of generated files.
        completion: What the model produced.
    """
    target = generated_path(out_dir, completion["item_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + PARTIAL_SUFFIX)
    staged.write_text(completion["text"], encoding="utf-8")
    staged.replace(target)


def append_manifest(out_dir: pathlib.Path, completions: Sequence[Completion]) -> None:
    """Record what a batch produced, as soon as it produces it.

    Appended per batch rather than written once at the end. A resumed run
    regenerates only the batches that were incomplete, so a manifest written
    at the end would describe that invocation rather than the arm. One line
    per item; on a redone batch the last line for an item wins.

    Args:
        out_dir: Directory of generated files.
        completions: The batch's completions.
    """
    path = manifest_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        dump_json_str(
            encode_generation_entry(GenerationEntry(item_id=c["item_id"], finished=c["finished"]))
        )
        + "\n"
        for c in completions
    )
    with path.open("a", encoding="utf-8") as handle:
        _ = handle.write(body)


def read_manifest(out_dir: pathlib.Path) -> dict[str, bool]:
    """Read back everything this arm has recorded, across every invocation.

    Read back rather than remembered, because a resumed run holds only the
    batches it redid. The record has to describe the ARM -- what is on disk
    now -- and not the last process that touched it.

    Args:
        out_dir: Directory of generated files.

    Returns:
        Whether each item's completion ended on its own. A later line for an
        item replaces an earlier one, which is what makes a redone batch
        correct the row it wrote the first time.

    Raises:
        JSONTypeError: If a line is not a readable manifest row.
    """
    path = manifest_path(out_dir)
    if not path.is_file():
        return {}
    finished: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = decode_generation_entry(narrow_json_to_dict(load_json_str(line)))
        finished[entry["item_id"]] = entry["finished"]
    return finished


def manifest_digest(prompts: Sequence[EvalPrompt], finished: dict[str, bool]) -> str:
    """Digest WHICH items this arm finished, in sweep order.

    Two arms agreeing on a finish rate can still disagree on which items ran
    out of budget, and the rate cannot tell you. Order is preserved rather
    than sorted, because both arms walk the same in-scope list and a
    reordering is itself a difference worth catching.

    Args:
        prompts: The in-scope prompts, in sweep order.
        finished: What the manifest recorded.

    Returns:
        ``sha256:`` followed by the hex digest of the canonical JSON of
        ``[item_id, finished]`` pairs.

    Raises:
        JSONTypeError: If an in-scope item has no manifest row. That means a
            file was generated without being recorded, or recorded without
            being generated, and either way the digest would describe a
            different set of items than the directory holds.
    """
    rows: list[list[str | bool]] = []
    for prompt in prompts:
        item_id = prompt["item_id"]
        if item_id not in finished:
            raise JSONTypeError(
                f"item {item_id!r} is in scope but has no manifest row; the "
                "generated directory and its manifest disagree"
            )
        rows.append([item_id, finished[item_id]])
    canonical = dump_json_str(rows)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def batch_is_complete(out_dir: pathlib.Path, batch: Sequence[EvalPrompt]) -> bool:
    """Say whether every item in a batch is already on disk.

    Resume is at BATCH granularity, never at item granularity. Dropping
    finished items out of a batch would repack the survivors with different
    neighbours, and since padding is what a neighbour changes, a resumed arm
    would not be numerically the arm the other arm is compared against. A
    batch is redone whole unless all of it is present, which costs at most
    one batch.

    Args:
        out_dir: Directory of generated files.
        batch: The batch.

    Returns:
        Whether it can be skipped.
    """
    return all(generated_path(out_dir, prompt["item_id"]).is_file() for prompt in batch)


def sweep_fingerprint(spec: ContinuationSweepSpec) -> RunFingerprint:
    """Pin determinism and describe what this arm is about to compute on.

    Determinism is pinned BEFORE the weights are loaded, because
    ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is created and
    loading weights is enough to create it. A pin after that is accepted in
    silence and does nothing.

    Both controls are on, matching the scoring path. Math attention costs
    real time on sequences this long, and it is paid rather than skipped: the
    sweep runs on a free partition where the currency is wall clock, and an
    arm whose attention kernel was chosen by a heuristic is an arm that can
    change between two runs of the same document.

    The package axis is read from the ARTIFACT's metadata rather than from
    the model that was loaded, so both arms name the same libraries. The base
    arm attaches no adapter, so a set read off the loaded model would omit
    ``peft`` on one side and name it on the other -- and the two records
    would then differ on an axis every single sweep, which is a difference
    that carries no information. What the axis states is which libraries this
    CONTRAST depends on, and peft decides what the candidate is.

    Args:
        spec: The sweep document.

    Returns:
        The fingerprint.
    """
    determinism = _test_hooks.apply_determinism_hook(remove_split_k=True, math_attention=True)
    metadata = read_hf_lm_metadata(spec["artifact_path"])
    return capture_run_fingerprint(
        spec["device"],
        determinism,
        model_distributions(
            uses_transformers=True,
            uses_peft=metadata["is_peft"],
            uses_quantization=metadata["quantization"] is not None,
        ),
    )


def generate_arm(
    spec: ContinuationSweepSpec, out_dir: pathlib.Path, fingerprint: RunFingerprint
) -> RunRecord:
    """Generate every in-scope item for one arm and record the result.

    Args:
        spec: The sweep document.
        out_dir: Directory of generated files.
        fingerprint: What this arm is computing on, captured before the
            weights were loaded.

    Returns:
        The run record.

    Raises:
        ValueError: If the holdout yields no in-scope item. An arm that
            generated nothing and reported success is the failure this
            refuses: the scorer would find no files and report having scored
            nothing, which is what a crashed generation also looks like.
    """
    model = load_arm(spec)
    scope = sweep_prompts(spec, model)
    prompts = scope["prompts"]
    if not prompts:
        raise ValueError(
            f"no held-out item's continuation fits {spec['max_new_tokens']} tokens; "
            "nothing would be generated and the sweep would report success"
        )

    grouped = batches(prompts, token_counter(model), spec["batch_size"])
    _log.info(
        "arm %s: %d of %d prompt(s) fit the %d-token budget, in %d batch(es) of up to %d",
        spec["arm"],
        len(prompts),
        scope["built"],
        spec["max_new_tokens"],
        len(grouped),
        spec["batch_size"],
    )

    generated = 0
    reused = 0
    for number, batch in enumerate(grouped, start=1):
        if batch_is_complete(out_dir, batch):
            reused += len(batch)
            _log.info("batch %d/%d already complete, skipping", number, len(grouped))
            continue
        completions = _test_hooks.generate_continuation_batch(
            model=model,
            prompts=batch,
            max_new_tokens=spec["max_new_tokens"],
            max_prompt_tokens=spec["max_prompt_tokens"],
            repetition_penalty=spec["repetition_penalty"],
            device=spec["device"],
            seed=spec["seed"],
        )
        for completion in completions:
            write_completion(out_dir, completion)
        append_manifest(out_dir, completions)
        generated += len(completions)
        _log.info("batch %d/%d wrote %d item(s)", number, len(grouped), len(completions))

    finished = read_manifest(out_dir)
    digest = manifest_digest(prompts, finished)
    ended = sum(1 for prompt in prompts if finished[prompt["item_id"]])
    _log.info(
        "arm %s: %d in scope, %d generated now, %d reused, %d ended on their own",
        spec["arm"],
        len(prompts),
        generated,
        reused,
        ended,
    )
    return run_record(
        experiment=CONTINUATION_EXPERIMENT,
        label=spec["label"],
        fingerprint=fingerprint,
        observations=(
            Observation(name="items_in_scope", value=float(len(prompts))),
            Observation(name="items_generated", value=float(generated)),
            Observation(name="items_reused", value=float(reused)),
            Observation(name="completions_finished", value=float(ended)),
            Observation(name="finished_rate", value=ended / len(prompts)),
        ),
        payload_digest=digest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one arm and write its record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the files, the manifest and the record are written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent. Nothing is generated on a command line
            that was not understood.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    spec = load_spec(pathlib.Path(cli_args.require_flag(parsed, SPEC_FLAG)))
    out_dir = pathlib.Path(cli_args.require_flag(parsed, OUT_DIR_FLAG))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Before the weights, for the reason `sweep_fingerprint` gives.
    fingerprint = sweep_fingerprint(spec)
    record = generate_arm(spec, out_dir, fingerprint)

    record_path = pathlib.Path(cli_args.require_flag(parsed, RECORD_FLAG))
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "sweep %s arm %s written: files under %s, record %s",
        spec["run_id"],
        spec["arm"],
        out_dir,
        record_path,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = [
    "CONTINUATION_EXPERIMENT",
    "OUT_DIR_FLAG",
    "PARTIAL_SUFFIX",
    "RECORD_FLAG",
    "SPEC_FLAG",
    "SweepScope",
    "append_manifest",
    "batch_is_complete",
    "entrypoint",
    "generate_arm",
    "load_arm",
    "load_spec",
    "main",
    "manifest_digest",
    "read_manifest",
    "sweep_fingerprint",
    "sweep_prompts",
    "token_counter",
    "write_completion",
]


# Without this, `python -m model_trainer.cli.continuations` imports the
# module, runs nothing and exits 0 -- having generated nothing while
# reporting success. A batch script is exactly where nobody is watching a
# terminal. The cluster entry point and the baseline scorer carry the same
# guard, for the same reason and after the same incident.
if __name__ == "__main__":
    entrypoint()
