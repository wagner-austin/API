---
title: Model-Trainer's fine-tuning strategy seam — four strategies, one enum, capabilities that are read
tags: [services, model-trainer, finetuning, lora, qlora, cartridge, registry, protocol]
related:
  - "[[model-trainer-service-architecture]]"
  - "[[model-trainer-composition-ceiling]]"
  - "[[corpus-attachment-program]]"
  - "[[covenant-radar-backend-registry]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/core/contracts/strategy_names.py
  - services/Model-Trainer/src/model_trainer/core/contracts/finetuning.py
  - services/Model-Trainer/src/model_trainer/core/services/finetuning/registry.py
  - services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/full.py
  - services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/lora.py
  - services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/qlora.py
  - services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/cartridge.py
  - services/Model-Trainer/src/model_trainer/core/services/model/backends/hf_lm/prepare.py
  - services/Model-Trainer/src/model_trainer/core/services/model/backends/hf_lm/io.py
  - services/Model-Trainer/src/model_trainer/core/services/training/trainer_grad_utils.py
  - services/Model-Trainer/src/model_trainer/core/services/training/reload.py
  - services/Model-Trainer/src/model_trainer/api/validators/runs_cross_fields.py
  - services/Model-Trainer/docs/ERNIE_LORA_INTEGRATION.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/core/contracts/strategy_names.py": bffee4ddfbdf08b18bc3ac0dd580c331cd1331ee
  "services/Model-Trainer/src/model_trainer/core/contracts/finetuning.py": 073069eb62b9dcce288ae14044a08c6a19503bc8
  "services/Model-Trainer/src/model_trainer/core/services/finetuning/registry.py": 82bb1555c49e9ecf00ee198c7ff0375313e2bec6
  "services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/full.py": 1d2e80c0fcd39344c7b5ea940d507f7d60ab0b33
  "services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/lora.py": b26b21d8937d901ffa98cf382c3ddf9ca4092bb5
  "services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/qlora.py": 59b9453085178354925411ab0ea3bd419a13ebcb
  "services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/cartridge.py": dfb982b403fb576e9bb088e529eeae5bd0cea152
  "services/Model-Trainer/src/model_trainer/core/services/model/backends/hf_lm/prepare.py": ad10892b12fd6a76e0c9040fcb9c5d990351c3a0
  "services/Model-Trainer/src/model_trainer/core/services/model/backends/hf_lm/io.py": 89d980ed8b1fdf5da6bc267527ec2d07cc6cb62f
  "services/Model-Trainer/src/model_trainer/core/services/training/trainer_grad_utils.py": 264311fe70463f245583e33a3bd7ec41f61f7e3b
  "services/Model-Trainer/src/model_trainer/core/services/training/reload.py": 49695d7c93f1e8f78c9315575cc41fd0384d8deb
  "services/Model-Trainer/src/model_trainer/api/validators/runs_cross_fields.py": 2d69645eb16c1f064c5f7b524ca0ce5a6da700fe
  "services/Model-Trainer/docs/ERNIE_LORA_INTEGRATION.md": 684b7878ef1c804aeaae5b9c19de985a0a6a0c59
provenance:
  - "read from code at API commit 2e0511ee7 on 2026-10-05"
  - "git log -S'unsloth' -- services/Model-Trainer/src -> efe588d11, 2026-08-17, 'Remove the unsloth strategy and bind the finetuning hooks'"
fact_checked: "2026-10-05"
confidence: high
hubs: [services]
---

# Model-Trainer's fine-tuning strategy seam

A *strategy* decides how a pretrained HuggingFace model is adapted before
training. It is the seam the cartridge programme ([[corpus-attachment-program]])
was built through, and the one a new adaptation method plugs into.[^proto][^hf]

## Four strategies, named once

`StrategyName` is a `StrEnum` — `full`, `lora`, `qlora`, `cartridge` — and is
the **only** place the names are written.[^names] Its docstring records why:
the set had been spelled out nine times across seven files, and a stale copy
did not fail to compile; it either refused a valid request deep in the stack
or silently dropped the strategy from a checkpoint's metadata.[^names]
Untrusted strings narrow through `require_strategy_name`.

**There is no Unsloth.** `docs/ERNIE_LORA_INTEGRATION.md` still lists
"full, lora, qlora, unsloth" as done and a four-value literal including
`"unsloth"`;[^ernie] the strategy was removed in `efe588d11` on 2026-08-17
(provenance). That document is history, not a description.

## The protocol

`FineTuningStrategy` has five members: `name()`, `capabilities()`,
`adapt(model, model_id, cfg) -> AdaptedModel`, `save_adapted(adapted, out_dir)`,
and `load_adapted(base_model, model_id, adapter_path)`.[^proto]
`StrategyCapabilities` declares `supports_quantization`,
`supports_gradient_checkpointing`, `requires_peft` and an approximate
`trainable_param_fraction`.[^proto]

| strategy | quantization | grad checkpointing | PEFT | trainable fraction |
|---|---|---|---|---|
| `full` | no | yes | no | 1.0 |
| `lora` | no | yes | yes | ~0.01 |
| `qlora` | yes | yes | yes | ~0.01 |
| `cartridge` | no | **no** | no | ~0.017 |

Sources: each strategy's `capabilities()`.[^caps]

`default_registry()` registers all four through `__import__` + a
Protocol-annotated factory, the same Protocol-Registration-Registry pattern as
covenant's model backends ([[covenant-radar-backend-registry]]).[^reg] Its
callers are the `hf_lm` backend — `prepare_hf_lm_with_handle` calls
`strategy.adapt`, and `io.py` uses the strategy recorded in an artifact's
metadata to reload it — so **strategies apply only to `hf_lm`**; `gpt2` and
`char_lstm` predate them.[^hf]

## Capabilities are load-bearing, and the cartridge row is why

`supports_gradient_checkpointing` was declared by all four strategies and read
by nothing outside their own tests until `_enable_gradient_checkpointing_if_supported`
began asking the registry for it.[^grad] The cartridge's `False` is measured,
not copied: a checkpointed model discards the key-value cache it is handed
(transformers 4.46.3, 2026-09-03), so the trained prefix would never reach
attention — and it read `True` until a run showed otherwise.[^cartridge]
Checkpointing is enabled at `BaseTrainer.train` rather than in a strategy,
because a continuation run reaches the trainer through `load_adapted`, not
`adapt`; that asymmetry once cost a 124M model 22.84 GiB on a 24 GB card.[^grad]

## Adding a strategy touches four places

1. a member on `StrategyName`;
2. a module under `finetuning/strategies/` exposing a factory, registered in
   `default_registry`;
3. the request cross-field rules — `lora` config is required for
   `lora`/`qlora`, `cartridge` config is required for and exclusive to
   `cartridge`;[^xf]
4. **the reload dispatch**, which is the one that bites late:
   `reload_shipped_weights` chooses a reader by the strategy that wrote the
   artifact (cartridge, PEFT adapter, HuggingFace directory, char-LSTM
   checkpoint). Cartridge arrived as the fourth format and its first run
   trained to completion and then died in `_restore_best_checkpoint` with
   "Unrecognized model".[^reload]

[^names]: `services/Model-Trainer/src/model_trainer/core/contracts/strategy_names.py` / `StrategyName`, `require_strategy_name`, and the module docstring ("WHY THIS MODULE EXISTS").
[^ernie]: `services/Model-Trainer/docs/ERNIE_LORA_INTEGRATION.md` § "ERNIE + LoRA/Unsloth Integration — Refactor Document" (status table, Phase 3 row).
[^proto]: `services/Model-Trainer/src/model_trainer/core/contracts/finetuning.py` / `FineTuningStrategy`, `StrategyCapabilities`, `AdaptedModel`.
[^caps]: `services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/full.py`, `lora.py`, `qlora.py`, `cartridge.py` / `capabilities` in each; cartridge's fraction is `_REPRESENTATIVE_TRAINABLE_FRACTION = 0.017`.
[^reg]: `services/Model-Trainer/src/model_trainer/core/services/finetuning/registry.py` / `default_registry`, `FineTuningRegistry.get_capabilities`.
[^hf]: `services/Model-Trainer/src/model_trainer/core/services/model/backends/hf_lm/prepare.py` / `prepare_hf_lm_with_handle`; `services/Model-Trainer/src/model_trainer/core/services/model/backends/hf_lm/io.py` (two `default_registry()` lookups).
[^grad]: `services/Model-Trainer/src/model_trainer/core/services/training/trainer_grad_utils.py` / `_enable_gradient_checkpointing_if_supported` (docstring, "THE INVARIANT THIS OWNS" and "WHY IT ASKS THE STRATEGY").
[^cartridge]: `services/Model-Trainer/src/model_trainer/core/services/finetuning/strategies/cartridge.py` / `CartridgeStrategy.capabilities` (docstring).
[^xf]: `services/Model-Trainer/src/model_trainer/api/validators/runs_cross_fields.py` / `_LORA_STRATEGIES` and the three cartridge refusals.
[^reload]: `services/Model-Trainer/src/model_trainer/core/services/training/reload.py` / `reload_shipped_weights` and the module docstring ("So there are FOUR artifact formats").
