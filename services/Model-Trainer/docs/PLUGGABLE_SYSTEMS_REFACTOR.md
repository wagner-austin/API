# Pluggable Systems Refactoring Plan

## Overview

This document outlines the implementation plan for adding missing pluggable systems to Model-Trainer, following the established patterns from the fine-tuning strategy system.

**Design Principles:**
- No `Any`, `cast`, `type: ignore`, `.pyi`, `noqa`
- No mocks in tests, no weak assertions
- Every TypedDict has encode/decode functions with `require_*` validation
- Every module has `_test_hooks.py` for dependency injection
- Production sets hooks to real implementations at startup
- Tests set hooks to fakes for isolation
- 100% test coverage for statements and branches
- Google-style docstrings
- No fallback, no best-effort, no try/except recovery

**Reference Pattern:** `src/model_trainer/core/services/finetuning/`

---

## Phase 1: Optimizer Strategy Protocol + Registry

### File Structure

```
src/model_trainer/core/
├── contracts/
│   └── optimizer.py                    # Protocol + TypedDicts
└── services/
    └── optimizer/
        ├── __init__.py
        ├── _test_hooks.py              # Dependency injection hooks
        ├── registry.py                 # OptimizerRegistry
        └── strategies/
            ├── __init__.py
            ├── _test_hooks.py          # Strategy-specific hooks
            ├── adamw.py                # AdamWStrategy
            ├── sgd.py                  # SGDStrategy
            └── adafactor.py            # AdaFactorStrategy

tests/core/services/optimizer/
├── __init__.py
├── conftest.py
├── testing.py                          # Fake implementations
├── test_registry.py
└── strategies/
    ├── __init__.py
    ├── testing.py
    ├── test_adamw.py
    ├── test_sgd.py
    └── test_adafactor.py
```

### contracts/optimizer.py

```python
"""Protocols and types for pluggable optimizer strategies.

Follows the covenant pattern of Protocol + Registry for extensibility.
Strict typing: no Any, cast, type: ignore, .pyi, or stubs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Protocol, TypedDict

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_float,
    require_str,
)

from model_trainer.core.types import NamedParameter

# Optimizer name as a literal type for strict typing
OptimizerName = Literal["adamw", "sgd", "adafactor"]


class OptimizerConfig(TypedDict):
    """Configuration for optimizer creation.

    Attributes:
        weight_decay: L2 regularization factor.
        betas: Adam beta parameters (momentum, RMSprop).
        eps: Numerical stability epsilon.
        amsgrad: Whether to use AMSGrad variant.
    """

    weight_decay: float
    betas: tuple[float, float]
    eps: float
    amsgrad: bool


class OptimizerCapabilities(TypedDict):
    """Declares what features an optimizer strategy supports.

    Attributes:
        supports_weight_decay: Whether weight decay is natively supported.
        supports_momentum: Whether momentum is supported.
        supports_adaptive_lr: Whether adaptive learning rates are used.
        memory_efficient: Whether optimizer uses less memory than Adam.
    """

    supports_weight_decay: bool
    supports_momentum: bool
    supports_adaptive_lr: bool
    memory_efficient: bool


def encode_optimizer_config(config: OptimizerConfig) -> JSONObject:
    """Encode OptimizerConfig to JSON-serializable dict.

    Args:
        config: Configuration to encode.

    Returns:
        JSON-serializable dictionary.
    """
    betas = config["betas"]
    return {
        "weight_decay": config["weight_decay"],
        "betas": [betas[0], betas[1]],
        "eps": config["eps"],
        "amsgrad": config["amsgrad"],
    }


def decode_optimizer_config(data: JSONObject) -> OptimizerConfig:
    """Decode JSON object to OptimizerConfig.

    Args:
        data: JSON dictionary to decode.

    Returns:
        Validated OptimizerConfig.

    Raises:
        JSONTypeError: If field types are incorrect.
    """
    from platform_core.json_utils import require_list

    weight_decay = require_float(data, "weight_decay")
    betas_list = require_list(data, "betas")
    if len(betas_list) != 2:
        from platform_core.json_utils import JSONTypeError

        raise JSONTypeError("Field 'betas' must have exactly 2 elements")
    beta1 = float(betas_list[0])
    beta2 = float(betas_list[1])
    eps = require_float(data, "eps")
    amsgrad = require_bool(data, "amsgrad")

    return OptimizerConfig(
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps,
        amsgrad=amsgrad,
    )


def encode_optimizer_capabilities(caps: OptimizerCapabilities) -> JSONObject:
    """Encode OptimizerCapabilities to JSON-serializable dict.

    Args:
        caps: Capabilities to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "supports_weight_decay": caps["supports_weight_decay"],
        "supports_momentum": caps["supports_momentum"],
        "supports_adaptive_lr": caps["supports_adaptive_lr"],
        "memory_efficient": caps["memory_efficient"],
    }


def decode_optimizer_capabilities(data: JSONObject) -> OptimizerCapabilities:
    """Decode JSON object to OptimizerCapabilities.

    Args:
        data: JSON dictionary to decode.

    Returns:
        Validated OptimizerCapabilities.

    Raises:
        JSONTypeError: If field types are incorrect.
    """
    return OptimizerCapabilities(
        supports_weight_decay=require_bool(data, "supports_weight_decay"),
        supports_momentum=require_bool(data, "supports_momentum"),
        supports_adaptive_lr=require_bool(data, "supports_adaptive_lr"),
        memory_efficient=require_bool(data, "memory_efficient"),
    )


class OptimizerProto(Protocol):
    """Protocol for PyTorch-compatible optimizer.

    Matches torch.optim.Optimizer interface for the methods we use.
    """

    def step(self) -> None:
        """Perform a single optimization step."""
        ...

    def zero_grad(self) -> None:
        """Zero out gradients."""
        ...


class OptimizerStrategy(Protocol):
    """Protocol for pluggable optimizer strategy implementations.

    Each strategy defines how to create an optimizer for training.
    Strategies are registered in OptimizerRegistry and selected by name.
    """

    def name(self) -> OptimizerName:
        """Return the strategy name identifier.

        Returns:
            Optimizer name as literal type.
        """
        ...

    def capabilities(self) -> OptimizerCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities describing what this optimizer supports.
        """
        ...

    def create_optimizer(
        self,
        parameters: Iterable[NamedParameter],
        lr: float,
        config: OptimizerConfig,
    ) -> OptimizerProto:
        """Create an optimizer for the given parameters.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate.
            config: Optimizer-specific configuration.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If required libraries are unavailable.
        """
        ...


class OptimizerStrategyFactory(Protocol):
    """Factory protocol to construct an optimizer strategy."""

    def __call__(self) -> OptimizerStrategy:
        """Create a new strategy instance.

        Returns:
            Strategy implementation.
        """
        ...


__all__ = [
    "OptimizerCapabilities",
    "OptimizerConfig",
    "OptimizerName",
    "OptimizerProto",
    "OptimizerStrategy",
    "OptimizerStrategyFactory",
    "decode_optimizer_capabilities",
    "decode_optimizer_config",
    "encode_optimizer_capabilities",
    "encode_optimizer_config",
]
```

### services/optimizer/_test_hooks.py

```python
"""Test hooks for optimizer strategies.

Follows the covenant pattern: production code sets hooks to real implementations,
tests set hooks to fakes for isolation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from model_trainer.core.types import NamedParameter


class TorchOptimizerCreator(Protocol):
    """Protocol for creating PyTorch optimizers."""

    def __call__(
        self,
        parameters: Iterable[NamedParameter],
        lr: float,
        *,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        amsgrad: bool,
    ) -> object:
        """Create a PyTorch optimizer.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate.
            weight_decay: L2 regularization factor.
            betas: Adam beta parameters.
            eps: Numerical stability epsilon.
            amsgrad: Whether to use AMSGrad variant.

        Returns:
            PyTorch optimizer instance.
        """
        ...


class SGDOptimizerCreator(Protocol):
    """Protocol for creating SGD optimizers."""

    def __call__(
        self,
        parameters: Iterable[NamedParameter],
        lr: float,
        *,
        momentum: float,
        weight_decay: float,
        nesterov: bool,
    ) -> object:
        """Create an SGD optimizer.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate.
            momentum: Momentum factor.
            weight_decay: L2 regularization factor.
            nesterov: Whether to use Nesterov momentum.

        Returns:
            SGD optimizer instance.
        """
        ...


class Hooks:
    """Container for test hooks.

    Production code sets these to real implementations.
    Tests set these to fakes for isolation.
    """

    create_adamw: TorchOptimizerCreator | None = None
    create_sgd: SGDOptimizerCreator | None = None
    create_adafactor: TorchOptimizerCreator | None = None


def reset_hooks() -> None:
    """Reset all hooks to None (for test cleanup)."""
    Hooks.create_adamw = None
    Hooks.create_sgd = None
    Hooks.create_adafactor = None


def init_production_hooks() -> None:
    """Initialize hooks with production implementations.

    Called at application startup to wire real PyTorch optimizers.
    """
    torch_optim = __import__("torch.optim", fromlist=["AdamW", "SGD"])

    def create_adamw_impl(
        parameters: Iterable[NamedParameter],
        lr: float,
        *,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        amsgrad: bool,
    ) -> object:
        adamw_cls: type[object] = getattr(torch_optim, "AdamW")
        return adamw_cls(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
        )

    def create_sgd_impl(
        parameters: Iterable[NamedParameter],
        lr: float,
        *,
        momentum: float,
        weight_decay: float,
        nesterov: bool,
    ) -> object:
        sgd_cls: type[object] = getattr(torch_optim, "SGD")
        return sgd_cls(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    Hooks.create_adamw = create_adamw_impl
    Hooks.create_sgd = create_sgd_impl


__all__ = [
    "Hooks",
    "SGDOptimizerCreator",
    "TorchOptimizerCreator",
    "init_production_hooks",
    "reset_hooks",
]
```

### services/optimizer/registry.py

```python
"""Registry for pluggable optimizer strategies.

Follows the covenant pattern: Protocol + Registration + Registry.
Strict typing; no optional fallbacks.
"""

from __future__ import annotations

from model_trainer.core.contracts.optimizer import (
    OptimizerCapabilities,
    OptimizerName,
    OptimizerStrategy,
    OptimizerStrategyFactory,
)


class OptimizerRegistration:
    """Registration record holding a factory and cached capabilities.

    Caches capabilities after first access to avoid repeated instantiation.
    """

    def __init__(self, factory: OptimizerStrategyFactory) -> None:
        """Initialize registration with a factory function.

        Args:
            factory: Callable that creates strategy instances.
        """
        self._factory = factory
        self._capabilities_cache: OptimizerCapabilities | None = None

    def factory(self) -> OptimizerStrategyFactory:
        """Return the factory function.

        Returns:
            Strategy factory callable.
        """
        return self._factory

    def capabilities(self) -> OptimizerCapabilities:
        """Return cached capabilities, creating strategy once if needed.

        Returns:
            Strategy capabilities.
        """
        if self._capabilities_cache is None:
            strategy = self._factory()
            self._capabilities_cache = strategy.capabilities()
        return self._capabilities_cache


class OptimizerRegistry:
    """Registry of optimizer strategies keyed by name.

    Provides lookup, enumeration, and capability querying.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._map: dict[OptimizerName, OptimizerRegistration] = {}

    def register(self, name: OptimizerName, registration: OptimizerRegistration) -> None:
        """Register a strategy by name.

        Args:
            name: Strategy identifier.
            registration: Registration containing factory and capabilities.
        """
        self._map[name] = registration

    def list_strategies(self) -> list[OptimizerName]:
        """List all registered strategy names.

        Returns:
            Sorted list of strategy names.
        """
        return sorted(self._map.keys())

    def get(self, name: OptimizerName) -> OptimizerStrategy:
        """Get a strategy instance by name.

        Args:
            name: Strategy identifier.

        Returns:
            Strategy instance.

        Raises:
            KeyError: If strategy is not registered.
        """
        reg = self._map[name]
        return reg.factory()()

    def get_capabilities(self, name: OptimizerName) -> OptimizerCapabilities:
        """Get capabilities for a strategy without full instantiation.

        Args:
            name: Strategy identifier.

        Returns:
            Strategy capabilities.

        Raises:
            KeyError: If strategy is not registered.
        """
        return self._map[name].capabilities()

    def is_registered(self, name: OptimizerName) -> bool:
        """Check if a strategy is registered.

        Args:
            name: Strategy identifier.

        Returns:
            True if strategy is registered.
        """
        return name in self._map


def default_optimizer_registry() -> OptimizerRegistry:
    """Build the default registry with all supported optimizer strategies.

    Includes:
        - adamw: AdamW with weight decay
        - sgd: Stochastic Gradient Descent
        - adafactor: Memory-efficient adaptive optimizer

    Returns:
        Registry with all strategies registered.
    """
    reg = OptimizerRegistry()

    adamw_mod = __import__(
        "model_trainer.core.services.optimizer.strategies.adamw",
        fromlist=["create_adamw_strategy"],
    )
    create_adamw: OptimizerStrategyFactory = adamw_mod.create_adamw_strategy
    reg.register("adamw", OptimizerRegistration(create_adamw))

    sgd_mod = __import__(
        "model_trainer.core.services.optimizer.strategies.sgd",
        fromlist=["create_sgd_strategy"],
    )
    create_sgd: OptimizerStrategyFactory = sgd_mod.create_sgd_strategy
    reg.register("sgd", OptimizerRegistration(create_sgd))

    adafactor_mod = __import__(
        "model_trainer.core.services.optimizer.strategies.adafactor",
        fromlist=["create_adafactor_strategy"],
    )
    create_adafactor: OptimizerStrategyFactory = adafactor_mod.create_adafactor_strategy
    reg.register("adafactor", OptimizerRegistration(create_adafactor))

    return reg


__all__ = [
    "OptimizerRegistration",
    "OptimizerRegistry",
    "default_optimizer_registry",
]
```

### services/optimizer/strategies/adamw.py

```python
"""AdamW optimizer strategy - Adam with decoupled weight decay.

Standard optimizer for transformer fine-tuning.
"""

from __future__ import annotations

from collections.abc import Iterable

from model_trainer.core.contracts.optimizer import (
    OptimizerCapabilities,
    OptimizerConfig,
    OptimizerName,
    OptimizerProto,
)
from model_trainer.core.services.optimizer._test_hooks import Hooks
from model_trainer.core.types import NamedParameter


class AdamWStrategy:
    """AdamW optimizer strategy.

    Uses Adam with decoupled weight decay regularization.
    Standard choice for transformer fine-tuning.

    Attributes:
        _name: Strategy identifier "adamw".
    """

    def __init__(self) -> None:
        """Initialize AdamW strategy."""
        self._name: OptimizerName = "adamw"

    def name(self) -> OptimizerName:
        """Return the strategy name identifier.

        Returns:
            Strategy name as literal "adamw".
        """
        return self._name

    def capabilities(self) -> OptimizerCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities showing adaptive LR with weight decay support.
        """
        return OptimizerCapabilities(
            supports_weight_decay=True,
            supports_momentum=True,
            supports_adaptive_lr=True,
            memory_efficient=False,
        )

    def create_optimizer(
        self,
        parameters: Iterable[NamedParameter],
        lr: float,
        config: OptimizerConfig,
    ) -> OptimizerProto:
        """Create an AdamW optimizer for the given parameters.

        Args:
            parameters: Model parameters to optimize.
            lr: Learning rate.
            config: Optimizer configuration.

        Returns:
            Configured AdamW optimizer instance.

        Raises:
            RuntimeError: If AdamW hook is not configured.
        """
        if Hooks.create_adamw is None:
            raise RuntimeError("AdamW hook not configured. Call init_production_hooks().")

        optimizer = Hooks.create_adamw(
            parameters,
            lr,
            weight_decay=config["weight_decay"],
            betas=config["betas"],
            eps=config["eps"],
            amsgrad=config["amsgrad"],
        )

        # Cast to protocol - optimizer implements step() and zero_grad()
        opt_proto: OptimizerProto = optimizer  # type narrowing via protocol
        return opt_proto


def create_adamw_strategy() -> AdamWStrategy:
    """Factory function to create an AdamWStrategy.

    Returns:
        New AdamWStrategy instance.
    """
    return AdamWStrategy()


__all__ = [
    "AdamWStrategy",
    "create_adamw_strategy",
]
```

---

## Phase 2: LR Scheduler Protocol + Registry

### File Structure

```
src/model_trainer/core/
├── contracts/
│   └── scheduler.py                    # Protocol + TypedDicts
└── services/
    └── scheduler/
        ├── __init__.py
        ├── _test_hooks.py
        ├── registry.py
        └── strategies/
            ├── __init__.py
            ├── _test_hooks.py
            ├── cosine.py
            ├── linear_warmup.py
            └── constant.py
```

### contracts/scheduler.py

```python
"""Protocols and types for pluggable learning rate schedulers.

Strict typing: no Any, cast, type: ignore, .pyi, or stubs.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from platform_core.json_utils import JSONObject, require_float, require_int

from model_trainer.core.contracts.optimizer import OptimizerProto

SchedulerName = Literal["cosine", "linear_warmup", "constant"]


class SchedulerConfig(TypedDict):
    """Configuration for scheduler creation.

    Attributes:
        warmup_steps: Number of warmup steps (0 for no warmup).
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as ratio of initial LR.
    """

    warmup_steps: int
    total_steps: int
    min_lr_ratio: float


class SchedulerCapabilities(TypedDict):
    """Declares what features a scheduler supports.

    Attributes:
        supports_warmup: Whether warmup is supported.
        supports_decay: Whether LR decay is supported.
        requires_total_steps: Whether total_steps must be known upfront.
    """

    supports_warmup: bool
    supports_decay: bool
    requires_total_steps: bool


def encode_scheduler_config(config: SchedulerConfig) -> JSONObject:
    """Encode SchedulerConfig to JSON-serializable dict.

    Args:
        config: Configuration to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "warmup_steps": config["warmup_steps"],
        "total_steps": config["total_steps"],
        "min_lr_ratio": config["min_lr_ratio"],
    }


def decode_scheduler_config(data: JSONObject) -> SchedulerConfig:
    """Decode JSON object to SchedulerConfig.

    Args:
        data: JSON dictionary to decode.

    Returns:
        Validated SchedulerConfig.

    Raises:
        JSONTypeError: If field types are incorrect.
    """
    return SchedulerConfig(
        warmup_steps=require_int(data, "warmup_steps"),
        total_steps=require_int(data, "total_steps"),
        min_lr_ratio=require_float(data, "min_lr_ratio"),
    )


class LRSchedulerProto(Protocol):
    """Protocol for PyTorch-compatible LR scheduler."""

    def step(self) -> None:
        """Advance scheduler by one step."""
        ...

    def get_last_lr(self) -> list[float]:
        """Get last computed learning rate.

        Returns:
            List of learning rates for each parameter group.
        """
        ...


class SchedulerStrategy(Protocol):
    """Protocol for pluggable LR scheduler implementations."""

    def name(self) -> SchedulerName:
        """Return the strategy name identifier.

        Returns:
            Scheduler name as literal type.
        """
        ...

    def capabilities(self) -> SchedulerCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities describing what this scheduler supports.
        """
        ...

    def create_scheduler(
        self,
        optimizer: OptimizerProto,
        config: SchedulerConfig,
    ) -> LRSchedulerProto:
        """Create an LR scheduler for the given optimizer.

        Args:
            optimizer: Optimizer to schedule.
            config: Scheduler-specific configuration.

        Returns:
            Configured scheduler instance.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If required libraries are unavailable.
        """
        ...


class SchedulerStrategyFactory(Protocol):
    """Factory protocol to construct a scheduler strategy."""

    def __call__(self) -> SchedulerStrategy:
        """Create a new strategy instance.

        Returns:
            Strategy implementation.
        """
        ...


__all__ = [
    "LRSchedulerProto",
    "SchedulerCapabilities",
    "SchedulerConfig",
    "SchedulerName",
    "SchedulerStrategy",
    "SchedulerStrategyFactory",
    "decode_scheduler_config",
    "encode_scheduler_config",
]
```

---

## Phase 3: CV Strategy Protocol + Registry

### File Structure

```
src/model_trainer/core/
├── contracts/
│   └── cv.py                           # Protocol + TypedDicts
└── services/
    └── cv/
        ├── __init__.py
        ├── _test_hooks.py
        ├── registry.py
        └── strategies/
            ├── __init__.py
            ├── holdout.py              # Current implementation
            ├── kfold.py
            ├── stratified_kfold.py
            └── timeseries.py
```

### contracts/cv.py

```python
"""Protocols and types for pluggable cross-validation strategies.

Strict typing: no Any, cast, type: ignore, .pyi, or stubs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal, Protocol, TypedDict

from platform_core.json_utils import JSONObject, require_float, require_int

CVStrategyName = Literal["holdout", "kfold", "stratified_kfold", "timeseries"]


class CVConfig(TypedDict):
    """Configuration for cross-validation.

    Attributes:
        n_splits: Number of folds (for k-fold strategies).
        test_size: Fraction of data for test set.
        shuffle: Whether to shuffle before splitting.
        random_state: Random seed for reproducibility.
    """

    n_splits: int
    test_size: float
    shuffle: bool
    random_state: int


class CVCapabilities(TypedDict):
    """Declares what features a CV strategy supports.

    Attributes:
        preserves_order: Whether data order is preserved.
        supports_stratification: Whether class-balanced splits are supported.
        supports_groups: Whether group-aware splitting is supported.
        n_iterations: Number of train/test iterations produced.
    """

    preserves_order: bool
    supports_stratification: bool
    supports_groups: bool
    n_iterations: int


class CVSplit(TypedDict):
    """A single train/validation split.

    Attributes:
        train_indices: Indices for training set.
        val_indices: Indices for validation set.
        fold_id: Identifier for this fold (0-indexed).
    """

    train_indices: Sequence[int]
    val_indices: Sequence[int]
    fold_id: int


def encode_cv_config(config: CVConfig) -> JSONObject:
    """Encode CVConfig to JSON-serializable dict.

    Args:
        config: Configuration to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "n_splits": config["n_splits"],
        "test_size": config["test_size"],
        "shuffle": config["shuffle"],
        "random_state": config["random_state"],
    }


def decode_cv_config(data: JSONObject) -> CVConfig:
    """Decode JSON object to CVConfig.

    Args:
        data: JSON dictionary to decode.

    Returns:
        Validated CVConfig.

    Raises:
        JSONTypeError: If field types are incorrect.
    """
    from platform_core.json_utils import require_bool

    return CVConfig(
        n_splits=require_int(data, "n_splits"),
        test_size=require_float(data, "test_size"),
        shuffle=require_bool(data, "shuffle"),
        random_state=require_int(data, "random_state"),
    )


class CVStrategy(Protocol):
    """Protocol for pluggable cross-validation strategy implementations."""

    def name(self) -> CVStrategyName:
        """Return the strategy name identifier.

        Returns:
            CV strategy name as literal type.
        """
        ...

    def capabilities(self) -> CVCapabilities:
        """Return strategy capabilities for discovery.

        Returns:
            Capabilities describing what this CV strategy supports.
        """
        ...

    def split(
        self,
        n_samples: int,
        config: CVConfig,
    ) -> Iterator[CVSplit]:
        """Generate train/validation splits.

        Args:
            n_samples: Total number of samples.
            config: CV configuration.

        Yields:
            CVSplit for each fold.

        Raises:
            ValueError: If configuration is invalid for this strategy.
        """
        ...


class CVStrategyFactory(Protocol):
    """Factory protocol to construct a CV strategy."""

    def __call__(self) -> CVStrategy:
        """Create a new strategy instance.

        Returns:
            Strategy implementation.
        """
        ...


__all__ = [
    "CVCapabilities",
    "CVConfig",
    "CVSplit",
    "CVStrategy",
    "CVStrategyFactory",
    "CVStrategyName",
    "decode_cv_config",
    "encode_cv_config",
]
```

---

## Phase 4: Warm-Start Protocol

### Extension to contracts/model.py

```python
# Add to existing contracts/model.py


class WarmStartConfig(TypedDict):
    """Configuration for warm-starting from checkpoints.

    Attributes:
        checkpoint_path: Path to checkpoint directory (None for fresh start).
        freeze_layers: Layer name patterns to freeze initially.
        unfreeze_at_epoch: Epoch at which to unfreeze frozen layers.
        layer_lr_multipliers: Per-layer learning rate multipliers.
    """

    checkpoint_path: str | None
    freeze_layers: Sequence[str]
    unfreeze_at_epoch: int
    layer_lr_multipliers: Mapping[str, float]


def encode_warm_start_config(config: WarmStartConfig) -> JSONObject:
    """Encode WarmStartConfig to JSON-serializable dict.

    Args:
        config: Configuration to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "checkpoint_path": config["checkpoint_path"],
        "freeze_layers": list(config["freeze_layers"]),
        "unfreeze_at_epoch": config["unfreeze_at_epoch"],
        "layer_lr_multipliers": dict(config["layer_lr_multipliers"]),
    }


def decode_warm_start_config(data: JSONObject) -> WarmStartConfig:
    """Decode JSON object to WarmStartConfig.

    Args:
        data: JSON dictionary to decode.

    Returns:
        Validated WarmStartConfig.

    Raises:
        JSONTypeError: If field types are incorrect.
    """
    from platform_core.json_utils import (
        require_int,
        require_list,
        require_str_or_none,
    )

    checkpoint_path = require_str_or_none(data, "checkpoint_path")
    freeze_layers_raw = require_list(data, "freeze_layers")
    freeze_layers: list[str] = [str(x) for x in freeze_layers_raw]
    unfreeze_at_epoch = require_int(data, "unfreeze_at_epoch")

    multipliers_raw = data.get("layer_lr_multipliers", {})
    if not isinstance(multipliers_raw, dict):
        from platform_core.json_utils import JSONTypeError

        raise JSONTypeError("Field 'layer_lr_multipliers' must be an object")
    multipliers: dict[str, float] = {str(k): float(v) for k, v in multipliers_raw.items()}

    return WarmStartConfig(
        checkpoint_path=checkpoint_path,
        freeze_layers=freeze_layers,
        unfreeze_at_epoch=unfreeze_at_epoch,
        layer_lr_multipliers=multipliers,
    )
```

---

## Integration Points

### BaseTrainer Updates

The `BaseTrainer` class in `services/training/base_trainer.py` needs updates to:

1. Accept `OptimizerRegistry` and `SchedulerRegistry` as dependencies
2. Use strategy pattern for optimizer/scheduler creation
3. Support warm-start configuration

```python
# In BaseTrainer.__init__
def __init__(
    self,
    *,
    optimizer_registry: OptimizerRegistry,
    scheduler_registry: SchedulerRegistry,
    # ... existing params
) -> None:
    self._optimizer_registry = optimizer_registry
    self._scheduler_registry = scheduler_registry
```

### Worker Entry Updates

The `worker_entry.py` needs to call `init_production_hooks()` for all systems:

```python
# In worker_entry.py or app startup
from model_trainer.core.services.optimizer._test_hooks import (
    init_production_hooks as init_optimizer_hooks,
)
from model_trainer.core.services.scheduler._test_hooks import (
    init_production_hooks as init_scheduler_hooks,
)


def init_all_hooks() -> None:
    """Initialize all production hooks at application startup."""
    init_optimizer_hooks()
    init_scheduler_hooks()
    # ... existing hooks
```

---

## Test Requirements

### Test File Template

```python
"""Tests for {module} module."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from model_trainer.core.services.{module}._test_hooks import (
    Hooks,
    reset_hooks,
)


@pytest.fixture(autouse=True)
def _reset_all_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    yield
    reset_hooks()


class TestStrategyName:
    """Tests for {Strategy} class."""

    def test_name_returns_correct_literal(self) -> None:
        """Test that name() returns the correct literal type."""
        strategy = {Strategy}()
        assert strategy.name() == "{name}"

    def test_capabilities_returns_valid_dict(self) -> None:
        """Test that capabilities() returns valid TypedDict."""
        strategy = {Strategy}()
        caps = strategy.capabilities()
        # Assert specific capability values
        assert type(caps["supports_weight_decay"]) is bool

    def test_create_raises_when_hook_not_configured(self) -> None:
        """Test that RuntimeError is raised when hook is None."""
        strategy = {Strategy}()
        with pytest.raises(RuntimeError, match="hook not configured"):
            strategy.create_optimizer([], 0.001, config)
```

### Testing.py Template (for tests/ directory)

```python
"""Test utilities for {module} tests.

Provides fake implementations for testing without requiring
actual PyTorch optimizers or external dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable

from model_trainer.core.contracts.optimizer import (
    OptimizerCapabilities,
    OptimizerConfig,
    OptimizerProto,
)
from model_trainer.core.types import NamedParameter


class FakeOptimizer(OptimizerProto):
    """Fake optimizer for testing."""

    def __init__(self) -> None:
        """Initialize fake optimizer."""
        self._step_count = 0

    def step(self) -> None:
        """Record step call."""
        self._step_count += 1

    def zero_grad(self) -> None:
        """No-op for testing."""
        pass


class FakeOptimizerCreator:
    """Fake optimizer creator for hook testing."""

    def __init__(self) -> None:
        """Initialize with call tracking."""
        self.calls: list[tuple[float, OptimizerConfig]] = []

    def __call__(
        self,
        parameters: Iterable[NamedParameter],
        lr: float,
        *,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        amsgrad: bool,
    ) -> FakeOptimizer:
        """Create fake optimizer and track call."""
        config = OptimizerConfig(
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
        )
        self.calls.append((lr, config))
        return FakeOptimizer()
```

---

## Implementation Checklist

### Phase 1: Optimizer Strategy
- [ ] Create `contracts/optimizer.py` with Protocol and TypedDicts
- [ ] Create `services/optimizer/__init__.py`
- [ ] Create `services/optimizer/_test_hooks.py`
- [ ] Create `services/optimizer/registry.py`
- [ ] Create `services/optimizer/strategies/__init__.py`
- [ ] Create `services/optimizer/strategies/_test_hooks.py`
- [ ] Create `services/optimizer/strategies/adamw.py`
- [ ] Create `services/optimizer/strategies/sgd.py`
- [ ] Create `services/optimizer/strategies/adafactor.py`
- [ ] Create `tests/core/services/optimizer/` with full test coverage
- [ ] Update `BaseTrainer` to use `OptimizerRegistry`
- [ ] Run `make check` - all tests pass, 100% coverage

### Phase 2: LR Scheduler Strategy
- [ ] Create `contracts/scheduler.py`
- [ ] Create `services/scheduler/` module structure
- [ ] Create scheduler strategies (cosine, linear_warmup, constant)
- [ ] Create tests with 100% coverage
- [ ] Update `BaseTrainer` to use `SchedulerRegistry`
- [ ] Run `make check` - all tests pass, 100% coverage

### Phase 3: CV Strategy
- [ ] Create `contracts/cv.py`
- [ ] Create `services/cv/` module structure
- [ ] Create CV strategies (holdout, kfold, stratified_kfold, timeseries)
- [ ] Create tests with 100% coverage
- [ ] Update dataset builder to use `CVRegistry`
- [ ] Run `make check` - all tests pass, 100% coverage

### Phase 4: Warm-Start Protocol
- [ ] Add `WarmStartConfig` to `contracts/model.py`
- [ ] Add encode/decode functions
- [ ] Update `prepare_hf_lm_with_handle` to support warm-start
- [ ] Update `BaseTrainer` with staged unfreezing
- [ ] Create tests with 100% coverage
- [ ] Run `make check` - all tests pass, 100% coverage

---

## Verification Commands

After each phase:

```bash
# Run full check suite
make check

# Verify no violations
# Guard rule summary: 0 violations in all categories

# Verify coverage
# TOTAL coverage: 100.00%
```
