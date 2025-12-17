"""CLI argument parsing and types for optimization script.

Provides argument parsing for multi-backend hyperparameter optimization.
Supports XGBoost, MLP, LightGBM, and LSTM backends.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from covenant_ml.types import BackendName
from platform_core.logging import get_rich_console

# Type aliases
FeaturePreset = Literal["none", "log_only", "ratios_only", "full"]
DatasetName = Literal["taiwan", "us", "polish"]

# Feature preset descriptions
PRESET_DESCRIPTIONS: dict[str, str] = {
    "none": "Original features only",
    "log_only": "Original + log transforms",
    "ratios_only": "Original + pairwise ratios (capped at 500)",
    "full": "Original + log + ratios + products (max ~800 features)",
}

# Backend descriptions for help text
BACKEND_DESCRIPTIONS: dict[BackendName, str] = {
    "xgboost": "Gradient boosted trees (XGBoost DMatrix API)",
    "mlp": "Multi-layer perceptron (PyTorch)",
    "lightgbm": "Gradient boosted trees (LightGBM)",
    "lstm": "Long short-term memory network (PyTorch)",
}


class OptimizeArgs:
    """Parsed command line arguments.

    Attributes:
        backend: ML backend to use for optimization.
        dataset: Dataset name to optimize on.
        n_trials: Number of Optuna trials to run.
        feature_preset: Feature engineering preset.
        device: Compute device (cpu, cuda, auto).
        timeout: Optional timeout in seconds.
        compare_presets: Run all presets and compare.
        all_datasets: Run on all datasets.
        verbose: Enable verbose logging.
        save_model: Train and save the best model after optimization.
    """

    backend: BackendName
    dataset: DatasetName
    n_trials: int
    feature_preset: FeaturePreset
    device: str
    timeout: int | None
    compare_presets: bool
    all_datasets: bool
    verbose: bool
    save_model: bool

    def __init__(self) -> None:
        """Initialize with defaults."""
        self.backend = "xgboost"
        self.dataset = "taiwan"
        self.n_trials = 300
        self.feature_preset = "full"
        self.device = "cuda"
        self.timeout = None
        self.compare_presets = False
        self.all_datasets = False
        self.verbose = False
        self.save_model = True


def print_help() -> None:
    """Print help message with all available options."""
    console = get_rich_console()
    help_text = """
[bold]Usage:[/bold] python -m scripts.optimize [OPTIONS]

[bold]Options:[/bold]
  -b, --backend         Backend: xgboost, mlp, lightgbm, lstm (default: xgboost)
  -d, --dataset         Dataset: taiwan, us, polish (default: taiwan)
  -n, --n-trials        Number of trials (default: 300)
  -f, --feature-preset  Preset: none, log_only, ratios_only, full (default: full)
  --device              Device: auto, cpu, cuda (default: cuda)
  -t, --timeout         Timeout in seconds (optional)
  -c, --compare-presets Run all presets on one dataset and compare
  -a, --all-datasets    Run on all three datasets
  -s, --save-model      Train and save the best model after optimization (default: on)
  --no-save-model       Skip training and saving the best model
  -v, --verbose         Show Optuna trial logs (default: quiet)
  -h, --help            Show this help

[bold]Backends:[/bold]
  xgboost   Gradient boosted trees (XGBoost DMatrix API)
  mlp       Multi-layer perceptron (PyTorch)
  lightgbm  Gradient boosted trees (LightGBM)
  lstm      Long short-term memory network (PyTorch)
"""
    console.print(help_text)


def _parse_backend(val: str) -> BackendName:
    """Parse backend value.

    Args:
        val (str): Backend name string from CLI.

    Returns:
        BackendName: Validated backend name literal.

    Raises:
        SystemExit: If backend name is invalid.
    """
    console = get_rich_console()
    if val == "xgboost":
        return "xgboost"
    if val == "mlp":
        return "mlp"
    if val == "lightgbm":
        return "lightgbm"
    if val == "lstm":
        return "lstm"
    console.print(f"[red]Invalid backend: {val}. Must be xgboost, mlp, lightgbm, or lstm.[/red]")
    raise SystemExit(1)


def _parse_dataset(val: str) -> DatasetName:
    """Parse dataset value.

    Args:
        val (str): Dataset name string from CLI.

    Returns:
        DatasetName: Validated dataset name literal.

    Raises:
        SystemExit: If dataset name is invalid.
    """
    console = get_rich_console()
    if val == "taiwan":
        return "taiwan"
    if val == "us":
        return "us"
    if val == "polish":
        return "polish"
    console.print(f"[red]Invalid dataset: {val}. Must be taiwan, us, or polish.[/red]")
    raise SystemExit(1)


def _parse_preset(val: str) -> FeaturePreset:
    """Parse feature preset value.

    Args:
        val (str): Feature preset string from CLI.

    Returns:
        FeaturePreset: Validated feature preset literal.

    Raises:
        SystemExit: If preset is invalid.
    """
    console = get_rich_console()
    if val == "none":
        return "none"
    if val == "log_only":
        return "log_only"
    if val == "ratios_only":
        return "ratios_only"
    if val == "full":
        return "full"
    console.print(f"[red]Invalid preset: {val}. Must be none, log_only, ratios_only, full.[/red]")
    raise SystemExit(1)


def _handle_flag(result: OptimizeArgs, arg: str) -> bool:
    """Handle boolean flags.

    Args:
        result (OptimizeArgs): OptimizeArgs instance to update.
        arg (str): Argument string to check.

    Returns:
        bool: True if flag was handled, False otherwise.

    Raises:
        SystemExit: If --help flag is provided (exits with code 0).
    """
    if arg in ("--compare-presets", "-c"):
        result.compare_presets = True
        return True
    if arg in ("--all-datasets", "-a"):
        result.all_datasets = True
        return True
    if arg in ("--save-model", "-s"):
        result.save_model = True
        return True
    if arg == "--no-save-model":
        result.save_model = False
        return True
    if arg in ("--verbose", "-v"):
        result.verbose = True
        return True
    if arg in ("--help", "-h"):
        print_help()
        raise SystemExit(0)
    return False


def parse_args(argv: Sequence[str]) -> OptimizeArgs:
    """Parse command line arguments.

    Args:
        argv (Sequence[str]): Command line argument sequence.

    Returns:
        OptimizeArgs: Parsed arguments with all settings.
    """
    args = list(argv)
    result = OptimizeArgs()

    i = 0
    while i < len(args):
        arg = args[i]
        if _handle_flag(result, arg):
            i += 1
        elif arg in ("--backend", "-b") and i + 1 < len(args):
            result.backend = _parse_backend(args[i + 1])
            i += 2
        elif arg in ("--dataset", "-d") and i + 1 < len(args):
            result.dataset = _parse_dataset(args[i + 1])
            i += 2
        elif arg in ("--n-trials", "-n") and i + 1 < len(args):
            result.n_trials = int(args[i + 1])
            i += 2
        elif arg in ("--feature-preset", "-f") and i + 1 < len(args):
            result.feature_preset = _parse_preset(args[i + 1])
            i += 2
        elif arg in ("--device",) and i + 1 < len(args):
            result.device = args[i + 1]
            i += 2
        elif arg in ("--timeout", "-t") and i + 1 < len(args):
            result.timeout = int(args[i + 1])
            i += 2
        else:
            i += 1

    return result
