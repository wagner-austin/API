"""What produced a trained model, recorded beside the model.

Training already recorded a full ``RunFingerprint``. This module does not fix
an absence, and it is worth being exact about that, because the first version
of this docstring said training recorded nothing and that was wrong: the
manifest written beside every saved model carries the card, the driver, the
host, the determinism posture and the resolved package versions, from the
same ``capture_run_fingerprint`` the benchmarks use.

What the manifest is NOT is a ``RunRecord``. So ``compare_run_records`` and
``agree_across_runs`` cannot read a training run beside a benchmark or beside
another experiment, and the shape a reader must parse to find a loss is local
to this service. That is the same defect as ``covenant_ml``'s
``BenchmarkManifest``, which had a correct fingerprint inside a private
envelope, and which is why nothing could read its numbers beside another
experiment's.

Two things the manifest genuinely lacks, and this module adds:

* **A digest over the saved weights.** Two runs can now be checked for
  bit-identity without this layer understanding safetensors.
* **``peft`` and ``bitsandbytes`` in the package axis.** The manifest records
  ``numpy``, ``torch`` and ``transformers`` for every run. A QLoRA fine-tune's
  arithmetic is decided by the other two as well, and neither appeared in the
  fingerprint of the adapter this project trained on 2026-09-01.

WHICH LIBRARIES GO IN THE FINGERPRINT DEPENDS ON THE RUN, and that is
deliberate rather than lazy. ``capture_package_versions`` is explicit that
naming the distributions is the caller's job, because a fingerprint recording
every installed package differs between two runs over a dev-dependency bump
that cannot reach the arithmetic. A QLoRA fine-tune's numbers depend on
``peft`` and ``bitsandbytes``; a character-LSTM run's do not, and that
environment may not have them installed at all, where naming them would raise
rather than record.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.json_utils import dump_json_str
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
    run_record_sidecar,
)

from model_trainer.core.contracts.model import QuantizationConfig, TrainOutcome
from model_trainer.core.run_fingerprint import capture_run_fingerprint


class PreparedModelFacts(Protocol):
    """The two facts about a prepared model that decide the library set.

    A Protocol rather than :class:`PreparedLMModel` because that class holds a
    torch module, and nothing here reads it. Depending on the minimal
    interface is what lets this be tested without loading weights to answer
    two boolean questions.

    Attributes:
        is_peft: Whether adapters were attached.
        quantization: The quantization the model was loaded under, or None.
    """

    is_peft: bool
    quantization: QuantizationConfig | None


#: Name under which trained models are comparable with each other.
TRAINING_EXPERIMENT = "model-trainer-training"

#: Libraries every training run's numbers depend on, whatever the backend.
BASE_TRAINING_DISTRIBUTIONS: tuple[str, ...] = ("numpy", "torch")

#: Added when the run went through the HuggingFace stack.
TRANSFORMERS_DISTRIBUTION = "transformers"

#: Added when adapters were attached. PEFT decides which tensors exist and
#: how they are merged, so two runs differing in it are not comparable.
PEFT_DISTRIBUTION = "peft"

#: Added when weights were quantized. bitsandbytes chooses the NF4 kernels
#: that do the arithmetic, so its version moves the loss on its own.
QUANTIZATION_DISTRIBUTION = "bitsandbytes"


def model_distributions(
    *, uses_transformers: bool, uses_peft: bool, uses_quantization: bool
) -> tuple[str, ...]:
    """Name the libraries a model's arithmetic depends on, from what it used.

    The table itself lives here and nowhere else. Training reaches it through
    :func:`training_distributions`, which knows how to read those three
    answers off a prepared model; an evaluation sweep reaches it with the
    answers read out of a saved run's metadata instead. Two copies of the
    table would be two package axes that drift, and a fingerprint axis that
    drifts is worse than one that is absent -- an absent one is obviously
    unanswerable, while a drifted one answers wrongly.

    Args:
        uses_transformers: Whether the HuggingFace stack decided the model's
            attention path.
        uses_peft: Whether adapters were attached. PEFT decides which tensors
            exist and how they are merged.
        uses_quantization: Whether weights were quantized. bitsandbytes
            chooses the NF4 kernels that do the arithmetic.

    Returns:
        The distribution names, without duplicates, in a stable order.
    """
    names = list(BASE_TRAINING_DISTRIBUTIONS)
    if uses_transformers:
        names.append(TRANSFORMERS_DISTRIBUTION)
    if uses_peft:
        names.append(PEFT_DISTRIBUTION)
    if uses_quantization:
        names.append(QUANTIZATION_DISTRIBUTION)
    return tuple(names)


def training_distributions(prepared: PreparedModelFacts, model_family: str) -> tuple[str, ...]:
    """Name the libraries whose versions decide this training run's numbers.

    Args:
        prepared: The model as it was prepared, which knows whether adapters
            were attached and whether it was quantized.
        model_family: Which backend ran, e.g. ``"hf_lm"`` or ``"char_lstm"``.

    Returns:
        The distribution names, without duplicates, in a stable order.
    """
    return model_distributions(
        uses_transformers=model_family != "char_lstm",
        uses_peft=prepared.is_peft,
        uses_quantization=prepared.quantization is not None,
    )


def training_observations(outcome: TrainOutcome) -> tuple[Observation, ...]:
    """Name the numbers a later contrast would read.

    ``steps`` travels with the losses on purpose. A loss reached in 1,633
    steps and the same loss reached in 200 are not the same result, and a
    record carrying only the loss cannot tell them apart.

    The optional held-out figures are omitted when absent rather than
    recorded as zero or as a sentinel: an observation named
    ``best_val_loss`` with a value of 0.0 would be read as a measurement.

    Args:
        outcome: What the training returned.

    Returns:
        The observations, in any order; the record sorts them.
    """
    observations = [
        Observation(name="train_loss", value=outcome["loss"]),
        Observation(name="train_perplexity", value=outcome["perplexity"]),
        Observation(name="steps", value=float(outcome["steps"])),
        Observation(name="early_stopped", value=float(outcome["early_stopped"])),
        Observation(name="cancelled", value=float(outcome["cancelled"])),
    ]
    test_loss = outcome["test_loss"]
    if test_loss is not None:
        observations.append(Observation(name="test_loss", value=test_loss))
    test_perplexity = outcome["test_perplexity"]
    if test_perplexity is not None:
        observations.append(Observation(name="test_perplexity", value=test_perplexity))
    best_val_loss = outcome["best_val_loss"]
    if best_val_loss is not None:
        observations.append(Observation(name="best_val_loss", value=best_val_loss))
    return tuple(observations)


def saved_model_digest(out_dir: Path) -> str:
    """Digest the weights a run saved.

    Every file in the directory contributes its name and its bytes, walked in
    sorted order, so two runs can be checked for bit-identity without this
    layer understanding safetensors. The name is hashed with the bytes so a
    rename is a difference.

    Args:
        out_dir: Directory the run saved into.

    Returns:
        A hex digest, or :const:`NO_PAYLOAD` when the directory holds no
        files. An empty directory is recorded as no payload rather than as
        the digest of nothing, which every empty run would share.
    """
    files = sorted((p for p in out_dir.rglob("*") if p.is_file()), key=_relative_name)
    if not files:
        return NO_PAYLOAD
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(out_dir).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _relative_name(path: Path) -> str:
    """Sort key for a saved file.

    Args:
        path: The file.

    Returns:
        Its name. A named function rather than a lambda, which the strict
        typing here rejects.
    """
    return path.name


def training_run_record(
    outcome: TrainOutcome,
    *,
    label: str,
    device: str,
    determinism: DeterminismRecord,
    prepared: PreparedModelFacts,
    model_family: str,
) -> RunRecord:
    """Build the record for a finished training run.

    Args:
        outcome: What the training returned.
        label: Which run this was, normally the run id.
        device: The device it trained on.
        determinism: The posture actually in force, from whatever pinner the
            run's stack used. An unpinned run records that honestly.
        prepared: The prepared model, for the library set.
        model_family: Which backend ran.

    Returns:
        The record.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        determinism,
        distributions=training_distributions(prepared, model_family),
    )
    return run_record(
        experiment=TRAINING_EXPERIMENT,
        label=label,
        fingerprint=fingerprint,
        observations=training_observations(outcome),
        payload_digest=saved_model_digest(Path(outcome["out_dir"])),
    )


def write_training_run_record(
    outcome: TrainOutcome,
    *,
    label: str,
    device: str,
    determinism: DeterminismRecord,
    prepared: PreparedModelFacts,
    model_family: str,
) -> Path:
    """Write the record beside the model a run just saved.

    Args:
        outcome: What the training returned.
        label: Which run this was, normally the run id.
        device: The device it trained on.
        determinism: The posture actually in force.
        prepared: The prepared model, for the library set.
        model_family: Which backend ran.

    Returns:
        The path written.
    """
    record = training_run_record(
        outcome,
        label=label,
        device=device,
        determinism=determinism,
        prepared=prepared,
        model_family=model_family,
    )
    path = training_record_path(Path(outcome["out_dir"]))
    path.write_text(
        dump_json_str(encode_run_record(record), compact=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def training_record_path(out_dir: Path) -> Path:
    """Name the record that belongs beside a saved model.

    Args:
        out_dir: Directory the run saved into.

    Returns:
        The sidecar path. Named from the directory rather than from a file
        inside it, because which file holds the weights differs between an
        adapter and a full model, and the record describes the run rather
        than any one of its outputs.
    """
    return run_record_sidecar(out_dir)


__all__ = [
    "BASE_TRAINING_DISTRIBUTIONS",
    "PEFT_DISTRIBUTION",
    "QUANTIZATION_DISTRIBUTION",
    "TRAINING_EXPERIMENT",
    "TRANSFORMERS_DISTRIBUTION",
    "PreparedModelFacts",
    "model_distributions",
    "saved_model_digest",
    "training_distributions",
    "training_observations",
    "training_record_path",
    "training_run_record",
    "write_training_run_record",
]
