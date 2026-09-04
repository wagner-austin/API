"""Scaffolding for the continuation-sweep command's tests.

Its own module because both halves of that suite need all of it -- the sweep
behaviour and the provenance record are separate concerns and separate files,
but they drive the same command through the same three seams. Sharing this
rather than copying it is what stops the two files drifting into testing two
slightly different commands.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Sequence

import pytest
import torch
from platform_core.continuation_task import EvalPrompt
from platform_core.determinism_record import TRUE, DeterminismRecord, determinism_record
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.core import _test_hooks as core_hooks
from model_trainer.core.contracts.continuation_sweep import Completion, ContinuationArm
from model_trainer.core.contracts.model import PreparedLMModel, QuantizationConfig
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import (
    HFTokenizerProto,
)
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import (
    FakeEncoder,
    FakeHFModel,
    FakeHFTokenizer,
)

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})

PREPARED: PreparedLMModel = PreparedLMModel(
    model=FakeHFModel(),
    tokenizer_id=None,
    eos_id=0,
    pad_id=0,
    max_seq_len=512,
    tok_for_dataset=FakeEncoder(),
)

LONG_SOURCE = "".join(f"line{index}\n" for index in range(30))
"""A file long enough to be a continuation task at twenty prompt lines."""


def fake_model_loader(
    model_id_or_path: str, quantization: QuantizationConfig | None
) -> LMModelProto:
    """Stand in for the HuggingFace model loader.

    Args:
        model_id_or_path: What was asked for.
        quantization: The quantization requested, or None.

    Returns:
        A fake carrying the requested name.
    """
    return FakeHFModel(model_id_or_path)


def fake_full_model_loader(model_path: str) -> LMModelProto:
    """Stand in for the full-finetune strategy's reload.

    Args:
        model_path: The saved directory.

    Returns:
        A fake carrying the path it was given.
    """
    return FakeHFModel(f"loaded-{model_path}")


def fake_tokenizer_loader(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the HuggingFace tokenizer loader.

    Args:
        model_id_or_path: What was asked for.

    Returns:
        A fake tokenizer.
    """
    return FakeHFTokenizer()


class _GeneratingModel(FakeHFModel):
    """A model that answers ``generate`` by appending one canned token.

    Only the production default decoder needs this, and it needs only enough
    of a model to be called: what that default is checked for is that it
    forwards its arguments to the real generator, not what the generator
    computes.
    """

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        repetition_penalty: float,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Append one token to every row.

        Args:
            input_ids: Left-padded prompt ids.
            attention_mask: 1 on real positions, 0 on padding.
            max_new_tokens: Token budget per row.
            do_sample: Whether to sample.
            repetition_penalty: Penalty on tokens already emitted.
            pad_token_id: What finished rows are padded with.

        Returns:
            The prompts with one token appended.
        """
        appended = torch.full((int(input_ids.size(0)), 1), 9, dtype=torch.long)
        return torch.cat((input_ids, appended), dim=1)


def generating_model() -> PreparedLMModel:
    """Build a prepared model whose inner model can generate.

    Returns:
        The prepared model.
    """
    return PreparedLMModel(
        model=_GeneratingModel(),
        tokenizer_id=None,
        eos_id=0,
        pad_id=1,
        max_seq_len=512,
        tok_for_dataset=FakeEncoder(),
    )


class Recorder:
    """Records the order of the seams and what each batch was asked for."""

    def __init__(self, *, finished_ids: Sequence[str] = ()) -> None:
        """Bind the recorder to which items it will claim ended on their own.

        Args:
            finished_ids: Item ids whose completions emitted end-of-sequence.
        """
        self.order: list[str] = []
        self.batches: list[list[str]] = []
        self.postures: list[tuple[bool, bool]] = []
        self._finished = set(finished_ids)

    def apply_determinism(self, *, remove_split_k: bool, math_attention: bool) -> DeterminismRecord:
        """Stand in for the determinism pin.

        Args:
            remove_split_k: Whether split-K removal was requested.
            math_attention: Whether the math attention path was requested.

        Returns:
            A fixed posture.
        """
        self.order.append("pin")
        self.postures.append((remove_split_k, math_attention))
        return PINNED

    def load_arm(self, artifact_path: str, arm: ContinuationArm, /) -> PreparedLMModel:
        """Stand in for the arm loader.

        Args:
            artifact_path: The saved run.
            arm: Which side to load.

        Returns:
            A prepared model holding shared fakes.
        """
        self.order.append(f"load:{arm}")
        return PREPARED

    def generate(
        self,
        *,
        model: PreparedLMModel,
        prompts: Sequence[EvalPrompt],
        max_new_tokens: int,
        max_prompt_tokens: int,
        repetition_penalty: float,
        device: str,
        seed: int,
    ) -> list[Completion]:
        """Stand in for the batched decoder.

        Args:
            model: The loaded arm.
            prompts: The batch.
            max_new_tokens: Token budget.
            max_prompt_tokens: Prompt budget.
            repetition_penalty: Penalty on repeats.
            device: Where tensors would go.
            seed: Seed for this batch.

        Returns:
            One completion per prompt.
        """
        self.order.append("generate")
        self.batches.append([prompt["item_id"] for prompt in prompts])
        return [
            Completion(
                item_id=prompt["item_id"],
                text=prompt["prompt"] + f"# written for {prompt['item_id']}\n",
                finished=prompt["item_id"] in self._finished,
            )
            for prompt in prompts
        ]


def install(recorder: Recorder) -> None:
    """Point the two GPU seams and the pin at a recorder.

    Args:
        recorder: The stand-in.
    """
    cli_hooks.apply_determinism_hook = recorder.apply_determinism
    cli_hooks.load_continuation_arm = recorder.load_arm
    cli_hooks.generate_continuation_batch = recorder.generate


def _restore_hooks() -> Generator[None, None, None]:
    """Put the module-global hooks back after each test.

    Left swapped they would answer for every later test in the same worker.

    Yields:
        None, for the duration of one test.
    """
    saved = (
        cli_hooks.apply_determinism_hook,
        cli_hooks.load_continuation_arm,
        cli_hooks.generate_continuation_batch,
    )
    git = core_hooks.env_git_commit
    name = core_hooks.cuda_device_name
    driver = core_hooks.cuda_driver_version
    yield
    (
        cli_hooks.apply_determinism_hook,
        cli_hooks.load_continuation_arm,
        cli_hooks.generate_continuation_batch,
    ) = saved
    core_hooks.env_git_commit = git
    core_hooks.cuda_device_name = name
    core_hooks.cuda_driver_version = driver


restore_hooks = pytest.fixture(_restore_hooks)


def _holdout(tmp_path: pathlib.Path, paths: Sequence[str]) -> pathlib.Path:
    """Write a holdout carrying one long document per path.

    Args:
        tmp_path: The test's temporary directory.
        paths: Repository-relative item paths.

    Returns:
        The holdout file.
    """
    path = tmp_path / "holdout.jsonl"
    path.write_text(
        "".join(
            dump_json_str({"repo": "api", "path": p, "text": LONG_SOURCE}) + "\n" for p in paths
        ),
        encoding="utf-8",
    )
    return path


def _artifact(tmp_path: pathlib.Path, *, quantized: bool) -> pathlib.Path:
    """Write a saved run's metadata, which both arms are defined by.

    Args:
        tmp_path: The test's temporary directory.
        quantized: Whether the run recorded a quantization.

    Returns:
        The artifact directory.
    """
    directory = tmp_path / "artifact"
    directory.mkdir(parents=True, exist_ok=True)
    quantization: JSONObject | None = (
        {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        if quantized
        else None
    )
    metadata: JSONObject = {
        "strategy_name": "qlora",
        "hub_model_id": "Qwen/Qwen2.5-Coder-1.5B",
        "tokenizer_id": None,
        "is_peft": True,
        "quantization": quantization,
    }
    (directory / "hf_lm_metadata.json").write_text(dump_json_str(metadata), encoding="utf-8")
    return directory


def spec_file(
    tmp_path: pathlib.Path,
    *,
    paths: Sequence[str] = ("src/a.py", "src/b.py", "src/c.py"),
    arm: str = "candidate",
    batch_size: int = 2,
    max_new_tokens: int = 4096,
    quantized: bool = True,
) -> pathlib.Path:
    """Write a valid sweep document.

    Args:
        tmp_path: The test's temporary directory.
        paths: Item paths in the holdout.
        arm: Which side to generate.
        batch_size: Prompts per batch.
        max_new_tokens: Token budget, which also decides scope.
        quantized: Whether the artifact records a quantization.

    Returns:
        The spec file.
    """
    document: JSONObject = {
        "run_id": "qlora-qwen-code-v1",
        "arm": arm,
        "artifact_path": str(_artifact(tmp_path, quantized=quantized)),
        "holdout_path": str(_holdout(tmp_path, paths)),
        "prompt_lines": 20,
        "max_new_tokens": max_new_tokens,
        "max_prompt_tokens": 1024,
        "batch_size": batch_size,
        "repetition_penalty": 1.1,
        "seed": 0,
        "device": "cpu",
        "experiment": "code-style-guard-pass",
        "label": f"qwen-qlora-v1-{arm}",
    }
    path = tmp_path / f"{arm}.spec.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    return path


def command_line(tmp_path: pathlib.Path, spec: pathlib.Path) -> list[str]:
    """Build a full command line for one arm.

    Args:
        tmp_path: The test's temporary directory.
        spec: The sweep document.

    Returns:
        The arguments, excluding the program name.
    """
    return [
        "--spec",
        str(spec),
        "--out-dir",
        str(tmp_path / "generated" / "candidate"),
        "--record",
        str(tmp_path / "out" / "record.json"),
    ]


def read_record(tmp_path: pathlib.Path) -> dict[str, float]:
    """Read a written record's observations by name.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        Observation name to value.
    """
    path = tmp_path / "out" / "record.json"
    record = decode_run_record(load_json_str(path.read_text(encoding="utf-8")))
    return {o["name"]: o["value"] for o in record["observations"]}


__all__ = [
    "PINNED",
    "PREPARED",
    "Recorder",
    "fake_full_model_loader",
    "fake_model_loader",
    "fake_tokenizer_loader",
    "generating_model",
    "install",
    "read_record",
    "restore_hooks",
    "spec_file",
]
