"""Import-order regression tests.

The suite normally imports backend modules directly, which initialises the
gpt2 package before anything reaches ``base_trainer``. Production does the
opposite: the hf_lm backend defers its ``BaseTrainer`` import until
``_default_create_trainer`` runs inside the worker, so ``base_trainer`` is the
first of the two to execute.

That ordering used to raise ``ImportError: cannot import name 'BaseTrainer'
from partially initialized module`` on every hf_lm training run, because
``base_trainer`` imported the DataLoader from inside the gpt2 backend, whose
package ``__init__`` imported ``gpt2.train``, which imported ``base_trainer``
again. A green suite proved nothing about it.

These tests import in a fresh interpreter so module state cannot be inherited
from another test, and assert the production order works.
"""

from __future__ import annotations

import subprocess
import sys


def _run_in_fresh_interpreter(source: str) -> subprocess.CompletedProcess[str]:
    """Execute source in a clean interpreter and capture the outcome.

    A subprocess is required rather than a plain import: once any test in the
    session has imported the gpt2 package, the ordering under test is no
    longer reachable in-process.

    Args:
        source: Python source to execute.

    Returns:
        The completed process, with stdout and stderr captured as text.
    """
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_base_trainer_imports_before_any_backend() -> None:
    """base_trainer must import standalone, as the hf_lm worker path does."""
    result = _run_in_fresh_interpreter(
        "from model_trainer.core.services.training.base_trainer import BaseTrainer\n"
        "print(BaseTrainer.__name__)\n"
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert result.stdout.strip() == "BaseTrainer"


def test_hf_lm_deferred_trainer_import_resolves() -> None:
    """Reproduce the exact deferred import the hf_lm backend performs.

    ``_default_create_trainer`` imports BaseTrainer at call time. Importing the
    hf_lm hooks module first and then resolving BaseTrainer is the production
    sequence that previously failed.
    """
    result = _run_in_fresh_interpreter(
        "import model_trainer.core.services.model.backends.hf_lm._test_hooks as h\n"
        "from model_trainer.core.services.training.base_trainer import BaseTrainer\n"
        "print(h.__name__.split('.')[-1], BaseTrainer.__name__)\n"
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert result.stdout.strip() == "_test_hooks BaseTrainer"


def test_dataloader_is_importable_without_the_gpt2_package() -> None:
    """The shared DataLoader must not drag a model backend in with it."""
    result = _run_in_fresh_interpreter(
        "import sys\n"
        "from model_trainer.core.services.training.dataloader import DataLoader\n"
        "gpt2 = 'model_trainer.core.services.model.backends.gpt2'\n"
        "print(DataLoader.__name__, gpt2 in sys.modules)\n"
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert result.stdout.strip() == "DataLoader False"
