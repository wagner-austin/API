"""Fine-tuning strategy implementations.

Pluggable strategies following the FineTuningStrategy Protocol:
    - full: Train all parameters (no adapters)
    - lora: LoRA via PEFT library
    - qlora: Quantized LoRA (4-bit base + LoRA)
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

__all__ = [
    "FullFineTuneStrategy",
    "LoRAStrategy",
    "QLoRAStrategy",
    "create_full_strategy",
    "create_lora_strategy",
    "create_qlora_strategy",
]
