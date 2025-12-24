"""HuggingFace Language Model backend.

This backend loads pretrained models from HuggingFace Hub via hub_model_id
and applies finetuning strategies (full, lora, qlora, unsloth).

Note: Imports are done at function call time to avoid circular imports
with BaseTrainer. Use direct imports from submodules when needed:

    from model_trainer.core.services.model.backends.hf_lm.prepare import (
        prepare_hf_lm_with_handle,
    )
"""

from __future__ import annotations

# Re-exports are intentionally omitted to avoid circular imports.
# Import from submodules directly.

__all__: list[str] = []
