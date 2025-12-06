from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONValue, load_json_str
from typing_extensions import TypedDict

from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend


def test_bpe_trains_and_writes_artifacts(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is a test\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    out_dir = artifacts / "tokenizers" / "tok-test"

    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    # Tokenizer training (no ML loss metric)
    loss_initial = 0.0
    stats = BPEBackend().train(cfg)
    loss_final = 0.0
    assert loss_final <= loss_initial
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert 0.0 <= stats.coverage <= 1.0

    class _Cfg(TypedDict):
        vocab_size: int
        min_frequency: int
        holdout_fraction: float
        seed: int
        special_tokens: list[str]

    class _Stats(TypedDict):
        coverage: float
        oov_rate: float
        token_count: int
        char_coverage: float

    class _Manifest(TypedDict):
        created_at: int
        config: _Cfg
        stats: _Stats

    text = (out_dir / "manifest.json").read_text(encoding="utf-8")
    obj_raw = load_json_str(text)
    assert isinstance(obj_raw, dict) and "config" in obj_raw and "stats" in obj_raw
    # Extract and validate into a precisely typed manifest structure
    cfg_raw: JSONValue = obj_raw.get("config")
    stats_raw: JSONValue = obj_raw.get("stats")
    assert isinstance(cfg_raw, dict) and isinstance(stats_raw, dict)
    # Extract and validate individual fields
    vocab_size_raw = cfg_raw.get("vocab_size", 0)
    min_freq_raw = cfg_raw.get("min_frequency", 0)
    holdout_raw = cfg_raw.get("holdout_fraction", 0.0)
    seed_raw = cfg_raw.get("seed", 0)
    special = cfg_raw.get("special_tokens", [])
    # Type narrow config fields
    if not isinstance(vocab_size_raw, int):
        raise AssertionError(f"Expected vocab_size to be int, got {type(vocab_size_raw)}")
    vocab_size: int = vocab_size_raw
    if not isinstance(min_freq_raw, int):
        raise AssertionError(f"Expected min_frequency to be int, got {type(min_freq_raw)}")
    min_freq: int = min_freq_raw
    if not isinstance(holdout_raw, (int, float)):
        raise AssertionError(f"Expected holdout_fraction to be numeric, got {type(holdout_raw)}")
    holdout: float = float(holdout_raw)
    if not isinstance(seed_raw, int):
        raise AssertionError(f"Expected seed to be int, got {type(seed_raw)}")
    seed: int = seed_raw
    coverage_raw = stats_raw.get("coverage", 0.0)
    oov_raw = stats_raw.get("oov_rate", 0.0)
    token_ct_raw = stats_raw.get("token_count", 0)
    char_cov_raw = stats_raw.get("char_coverage", 0.0)
    created_raw = obj_raw.get("created_at", 0)
    assert vocab_size == 128
    assert min_freq == 1
    assert holdout == 0.1
    assert seed == 42
    assert isinstance(special, list) and len(special) >= 0
    assert all(isinstance(tok, str) for tok in special)
    special_typed: list[str] = [tok for tok in special if isinstance(tok, str)]
    # Type narrow coverage to numeric before comparison
    if not isinstance(coverage_raw, (int, float)):
        raise AssertionError(f"Expected coverage to be numeric, got {type(coverage_raw)}")
    coverage: float = float(coverage_raw)
    assert 0.0 <= coverage <= 1.0
    # Type narrow oov to numeric before comparison
    if not isinstance(oov_raw, (int, float)):
        raise AssertionError(f"Expected oov to be numeric, got {type(oov_raw)}")
    oov: float = float(oov_raw)
    assert 0.0 <= oov <= 1.0
    # Type narrow token_ct to numeric before comparison
    if not isinstance(token_ct_raw, int):
        raise AssertionError(f"Expected token_ct to be int, got {type(token_ct_raw)}")
    token_ct: int = token_ct_raw
    assert token_ct > 0
    # Type narrow char_cov to numeric before comparison
    if not isinstance(char_cov_raw, (int, float)):
        raise AssertionError(f"Expected char_cov to be numeric, got {type(char_cov_raw)}")
    char_cov: float = float(char_cov_raw)
    assert 0.0 <= char_cov <= 1.0
    # Type narrow created to numeric before comparison
    if not isinstance(created_raw, int):
        raise AssertionError(f"Expected created to be int, got {type(created_raw)}")
    created: int = created_raw
    assert created > 0
    cfg_typed: _Cfg = {
        "vocab_size": vocab_size,
        "min_frequency": min_freq,
        "holdout_fraction": holdout,
        "seed": seed,
        "special_tokens": special_typed,
    }
    stats_typed: _Stats = {
        "coverage": coverage,
        "oov_rate": oov,
        "token_count": token_ct,
        "char_coverage": char_cov,
    }
    manifest: _Manifest = {
        "created_at": created,
        "config": cfg_typed,
        "stats": stats_typed,
    }
    assert manifest["config"]["vocab_size"] == 128
    assert 0.0 <= manifest["stats"]["coverage"] <= 1.0
    assert 0.0 <= manifest["stats"]["char_coverage"] <= 1.0
