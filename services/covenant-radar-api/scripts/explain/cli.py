"""CLI argument parsing and types for explain script.

Provides argument parsing for feature importance explanation.
Supports all backends (XGBoost, MLP, LightGBM, LSTM) and explainers.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from covenant_ml.explainers.types import SupportedExplainer
from covenant_ml.types import BackendName
from platform_core.logging import get_rich_console

# Type aliases
DatasetName = Literal["taiwan", "us", "polish"]

# Explainer descriptions for help text
EXPLAINER_DESCRIPTIONS: dict[SupportedExplainer, str] = {
    "permutation": "Permutation importance (all backends)",
    "gradient": "Gradient-based (MLP, LSTM only)",
    "integrated_gradients": "Integrated gradients (MLP, LSTM only)",
    "shap_tree": "SHAP TreeExplainer (XGBoost, LightGBM only)",
}

# Backend to compatible explainers mapping
BACKEND_EXPLAINERS: dict[BackendName, list[SupportedExplainer]] = {
    "xgboost": ["permutation", "shap_tree"],
    "lightgbm": ["permutation", "shap_tree"],
    "mlp": ["permutation", "gradient", "integrated_gradients"],
    "lstm": ["permutation", "gradient", "integrated_gradients"],
}


class ExplainArgs:
    """Parsed command line arguments.

    Attributes:
        backend: ML backend of the trained model.
        dataset: Dataset name to explain on.
        explainer: Explainer method to use.
        model_path: Path to trained model file (optional, uses default).
        n_samples: Number of samples to use for explanation.
        target_class: Target class for importance computation.
        top_n: Number of top features to display.
        verbose: Enable verbose logging.
    """

    backend: BackendName
    dataset: DatasetName
    explainer: SupportedExplainer
    model_path: str | None
    n_samples: int
    target_class: int
    top_n: int
    verbose: bool

    def __init__(self) -> None:
        """Initialize with defaults."""
        self.backend = "xgboost"
        self.dataset = "taiwan"
        self.explainer = "permutation"
        self.model_path = None
        self.n_samples = 1000
        self.target_class = 1
        self.top_n = 20
        self.verbose = False


def print_help() -> None:
    """Print help message with all available options."""
    console = get_rich_console()
    help_text = """
[bold]Usage:[/bold] python -m scripts.explain [OPTIONS]

[bold]Options:[/bold]
  -b, --backend         Backend: xgboost, mlp, lightgbm, lstm (default: xgboost)
  -d, --dataset         Dataset: taiwan, us, polish (default: taiwan)
  -e, --explainer       Explainer method (default: permutation)
  -m, --model-path      Path to model file (optional, uses saved best model)
  -n, --n-samples       Number of samples to use (default: 1000)
  -c, --target-class    Target class for importance (default: 1)
  -t, --top-n           Number of top features to show (default: 20)
  -v, --verbose         Enable verbose logging
  -h, --help            Show this help

[bold]Explainers:[/bold]
  permutation           Permutation importance (all backends)
  gradient              Gradient-based attribution (MLP, LSTM)
  integrated_gradients  Integrated gradients (MLP, LSTM)
  shap_tree             SHAP TreeExplainer (XGBoost, LightGBM)

[bold]Backend Compatibility:[/bold]
  xgboost   permutation, shap_tree
  lightgbm  permutation, shap_tree
  mlp       permutation, gradient, integrated_gradients
  lstm      permutation, gradient, integrated_gradients
"""
    console.print(help_text)


def _parse_backend(val: str) -> BackendName:
    """Parse backend value.

    Args:
        val: Backend name string from CLI.

    Returns:
        Validated backend name literal.

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
        val: Dataset name string from CLI.

    Returns:
        Validated dataset name literal.

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


def _parse_explainer(val: str) -> SupportedExplainer:
    """Parse explainer value.

    Args:
        val: Explainer name string from CLI.

    Returns:
        Validated explainer name literal.

    Raises:
        SystemExit: If explainer name is invalid.
    """
    console = get_rich_console()
    if val == "permutation":
        return "permutation"
    if val == "gradient":
        return "gradient"
    if val == "integrated_gradients":
        return "integrated_gradients"
    if val == "shap_tree":
        return "shap_tree"
    console.print(
        f"[red]Invalid explainer: {val}. "
        f"Must be permutation, gradient, integrated_gradients, or shap_tree.[/red]"
    )
    raise SystemExit(1)


def _handle_flag(result: ExplainArgs, arg: str) -> bool:
    """Handle boolean flags.

    Args:
        result: ExplainArgs instance to update.
        arg: Argument string to check.

    Returns:
        True if flag was handled, False otherwise.

    Raises:
        SystemExit: If --help flag is provided (exits with code 0).
    """
    if arg in ("--verbose", "-v"):
        result.verbose = True
        return True
    if arg in ("--help", "-h"):
        print_help()
        raise SystemExit(0)
    return False


def validate_explainer_backend(explainer: SupportedExplainer, backend: BackendName) -> None:
    """Validate that explainer is compatible with backend.

    Args:
        explainer: Selected explainer.
        backend: Selected backend.

    Raises:
        SystemExit: If explainer is incompatible with backend.
    """
    console = get_rich_console()
    compatible = BACKEND_EXPLAINERS[backend]
    if explainer not in compatible:
        compatible_str = ", ".join(compatible)
        console.print(
            f"[red]Explainer '{explainer}' is not compatible with backend '{backend}'.[/red]\n"
            f"[yellow]Compatible explainers for {backend}: {compatible_str}[/yellow]"
        )
        raise SystemExit(1)


def parse_args(argv: Sequence[str]) -> ExplainArgs:
    """Parse command line arguments.

    Args:
        argv: Command line argument sequence.

    Returns:
        Parsed arguments with all settings.
    """
    args = list(argv)
    result = ExplainArgs()

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
        elif arg in ("--explainer", "-e") and i + 1 < len(args):
            result.explainer = _parse_explainer(args[i + 1])
            i += 2
        elif arg in ("--model-path", "-m") and i + 1 < len(args):
            result.model_path = args[i + 1]
            i += 2
        elif arg in ("--n-samples", "-n") and i + 1 < len(args):
            result.n_samples = int(args[i + 1])
            i += 2
        elif arg in ("--target-class", "-c") and i + 1 < len(args):
            result.target_class = int(args[i + 1])
            i += 2
        elif arg in ("--top-n", "-t") and i + 1 < len(args):
            result.top_n = int(args[i + 1])
            i += 2
        else:
            i += 1

    # Validate explainer/backend compatibility after parsing
    validate_explainer_backend(result.explainer, result.backend)

    return result


__all__ = [
    "BACKEND_EXPLAINERS",
    "EXPLAINER_DESCRIPTIONS",
    "DatasetName",
    "ExplainArgs",
    "parse_args",
    "print_help",
    "validate_explainer_backend",
]
