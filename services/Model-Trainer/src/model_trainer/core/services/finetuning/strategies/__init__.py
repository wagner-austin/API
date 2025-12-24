"""Fine-tuning strategy implementations.

Pluggable strategies following the FineTuningStrategy Protocol:
    - full: Train all parameters (no adapters)
    - lora: LoRA via PEFT library
    - qlora: Quantized LoRA (4-bit base + LoRA)
    - unsloth: Unsloth-optimized LoRA (2x-5x faster)
"""

from model_trainer.core.services.finetuning.strategies.full import (
    FullFineTuneStrategy,
    create_full_strategy,
)
from model_trainer.core.services.finetuning.strategies.lora import (
    LoRAStrategy,
    create_lora_strategy,
)
from model_trainer.core.services.finetuning.strategies.qlora import (
    QLoRAStrategy,
    create_qlora_strategy,
)
from model_trainer.core.services.finetuning.strategies.unsloth import (
    UnslothStrategy,
    create_unsloth_strategy,
)

__all__ = [
    "FullFineTuneStrategy",
    "LoRAStrategy",
    "QLoRAStrategy",
    "UnslothStrategy",
    "create_full_strategy",
    "create_lora_strategy",
    "create_qlora_strategy",
    "create_unsloth_strategy",
]
