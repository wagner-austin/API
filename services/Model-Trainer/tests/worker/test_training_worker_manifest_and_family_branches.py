from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str

from model_trainer.worker import manifest


def test_as_model_family_variants_and_invalid() -> None:
    assert manifest.as_model_family("gpt2") == "gpt2"
    assert manifest.as_model_family("llama") == "llama"
    assert manifest.as_model_family("qwen") == "qwen"
    assert manifest.as_model_family("char_lstm") == "char_lstm"
    with pytest.raises(JSONTypeError):
        _ = manifest.as_model_family("bert")


def _base_manifest() -> _ManifestDict:
    return {
        "run_id": "r",
        "model_family": "gpt2",
        "model_size": "small",
        "epochs": 1,
        "batch_size": 1,
        "max_seq_len": 8,
        "steps": 0,
        "loss": 0.0,
        "learning_rate": 0.001,
        "tokenizer_id": "tok",
        "corpus_path": "/tmp/x",
        "optimizer": "adamw",
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "device": "cpu",
        "precision": "fp32",
        "early_stopping_patience": 5,
        "test_split_ratio": 0.15,
        "finetune_lr_cap": 5e-5,
        "early_stopped": False,
        "versions": {
            "torch": "0",
            "transformers": "0",
            "tokenizers": "0",
            "datasets": "0",
        },
        "system": {
            "cpu_count": 1,
            "platform": "x",
            "platform_release": "y",
            "machine": "z",
        },
        "seed": 0,
        "holdout_fraction": 0.1,
        "pretrained_run_id": None,
        "git_commit": "g",
    }


_ManifestDict = dict[str, str | int | float | bool | None | dict[str, str | int | float]]


def _manifest_unknown() -> _ManifestDict:
    return _base_manifest().copy()


def test_load_manifest_from_text_invalid_json() -> None:
    with pytest.raises(JSONTypeError):
        _ = manifest.load_manifest_from_text("[]")


def test_load_manifest_from_text_invalid_versions_system() -> None:
    bad = _base_manifest()
    bad["versions"] = "oops"
    txt = dump_json_str(bad)
    with pytest.raises(JSONTypeError, match="versions"):
        _ = manifest.load_manifest_from_text(txt)


def test_load_manifest_from_text_expect_str_error() -> None:
    bad = _base_manifest()
    bad["model_family"] = 123  # not a string
    txt = dump_json_str(bad)
    with pytest.raises(JSONTypeError):
        _ = manifest.load_manifest_from_text(txt)


def test_load_manifest_from_text_expect_int_and_num_errors() -> None:
    bad1 = _base_manifest()
    bad1["epochs"] = "one"
    with pytest.raises(JSONTypeError):
        _ = manifest.load_manifest_from_text(dump_json_str(bad1))

    bad2 = _base_manifest()
    bad2["learning_rate"] = "fast"
    with pytest.raises(JSONTypeError):
        _ = manifest.load_manifest_from_text(dump_json_str(bad2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torch", 123),
        ("transformers", 456),
        ("tokenizers", 789),
        ("datasets", 999),
    ],
)
def test_load_manifest_versions_require_strings(field: str, value: int) -> None:
    base = _manifest_unknown()
    versions_raw = base["versions"]
    assert isinstance(versions_raw, dict) and len(versions_raw) == 4
    versions = dict(versions_raw)
    versions[field] = value
    bad_manifest: dict[str, str | int | float | bool | None | dict[str, str | int | float]] = {
        **base,
        "versions": versions,
    }
    txt = dump_json_str(bad_manifest)
    with pytest.raises(JSONTypeError, match=field):
        _ = manifest.load_manifest_from_text(txt)


def test_load_manifest_system_not_dict() -> None:
    base = _manifest_unknown()
    bad_manifest: dict[str, str | int | float | bool | None | dict[str, str | int | float]] = {
        **base,
        "system": "oops",
    }
    with pytest.raises(JSONTypeError, match="system"):
        _ = manifest.load_manifest_from_text(dump_json_str(bad_manifest))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cpu_count", "one", "cpu_count"),
        ("platform", 123, "platform must be str"),
        ("platform_release", 5, "platform_release"),
        ("machine", 9, "machine must be str"),
    ],
)
def test_load_manifest_system_field_types(field: str, value: str | int, message: str) -> None:
    base = _manifest_unknown()
    system_raw = base["system"]
    assert isinstance(system_raw, dict) and len(system_raw) == 4
    system = dict(system_raw)
    system[field] = value
    bad_manifest: dict[str, str | int | float | bool | None | dict[str, str | int | float]] = {
        **base,
        "system": system,
    }
    with pytest.raises(JSONTypeError, match=field):
        _ = manifest.load_manifest_from_text(dump_json_str(bad_manifest))


def test_load_manifest_loss_must_be_number() -> None:
    bad_manifest: dict[str, str | int | float | bool | None | dict[str, str | int | float]] = {
        **_manifest_unknown(),
        "loss": "high",
    }
    with pytest.raises(JSONTypeError, match="loss"):
        _ = manifest.load_manifest_from_text(dump_json_str(bad_manifest))


def test_load_manifest_pretrained_run_id_must_be_str_or_null() -> None:
    """Cover manifest.py lines 100-101 - _decode_manifest_str_or_none error case."""
    bad_manifest: dict[str, str | int | float | bool | None | dict[str, str | int | float]] = {
        **_base_manifest(),
        "pretrained_run_id": 123,  # should be str or null, not int
    }
    with pytest.raises(JSONTypeError, match="pretrained_run_id"):
        _ = manifest.load_manifest_from_text(dump_json_str(bad_manifest))


def test_load_manifest_pretrained_run_id_valid_string() -> None:
    """Cover manifest.py line 102 - _decode_manifest_str_or_none returns valid string."""
    valid_manifest: dict[str, str | int | float | bool | None | dict[str, str | int | float]] = {
        **_base_manifest(),
        "pretrained_run_id": "run-base-123",  # valid string value
    }
    result = manifest.load_manifest_from_text(dump_json_str(valid_manifest))
    assert result["pretrained_run_id"] == "run-base-123"


def test_as_optimizer_variants_and_invalid() -> None:
    """Cover manifest.py as_optimizer branches."""
    assert manifest.as_optimizer("adamw") == "adamw"
    assert manifest.as_optimizer("adam") == "adam"
    assert manifest.as_optimizer("sgd") == "sgd"
    with pytest.raises(JSONTypeError, match="optimizer"):
        _ = manifest.as_optimizer("rmsprop")


def test_as_device_variants_and_invalid() -> None:
    """Cover manifest.py as_device branches (lines 42-46)."""
    assert manifest.as_device("cpu") == "cpu"
    assert manifest.as_device("cuda") == "cuda"
    with pytest.raises(JSONTypeError, match="device"):
        _ = manifest.as_device("tpu")


def test_as_precision_variants_and_invalid() -> None:
    """Cover manifest.py as_precision branches."""
    assert manifest.as_precision("fp32") == "fp32"
    assert manifest.as_precision("fp16") == "fp16"
    assert manifest.as_precision("bf16") == "bf16"
    with pytest.raises(JSONTypeError, match="precision"):
        _ = manifest.as_precision("int8")


def test_load_manifest_freeze_embed_must_be_bool() -> None:
    """Cover manifest.py _decode_manifest_bool error case."""
    bad_manifest: _ManifestDict = {
        **_base_manifest(),
        "freeze_embed": "yes",  # should be bool, not string
    }
    with pytest.raises(JSONTypeError, match="freeze_embed"):
        _ = manifest.load_manifest_from_text(dump_json_str(bad_manifest))


def test_load_manifest_float_or_none_error_case() -> None:
    """Cover manifest.py _decode_manifest_float_or_none error (line 137)."""
    bad_manifest: _ManifestDict = {
        **_base_manifest(),
        "test_loss": "not-a-number",  # should be float or null
    }
    with pytest.raises(JSONTypeError, match="test_loss"):
        _ = manifest.load_manifest_from_text(dump_json_str(bad_manifest))
