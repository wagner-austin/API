"""Name the exact weights a hub model loads as, for an edit to be written against.

Board task 7964f9f6, criterion A2, the step before the edit. A weight program
names the weights its writer read (``written_against``), and the measuring
run refuses it on any others. For the small-network arm the training run
reports that digest itself; for a hub model nothing else does, so this loads
the model the measuring run will load, from the same offline cache, and
writes its digest and its parameter inventory -- the names and shapes a
writer addresses its operations to.

Usage:
    python -m model_trainer.cli.weight_space_lm_digest --model gpt2 --out OUT.json
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.logging import get_logger

from model_trainer.cli.weight_space_lm_edit import load_editable_lm
from model_trainer.core.services.model.editing.verify import parameter_digests, weights_digest

_log = get_logger(__name__)

MODEL_FLAG = "--model"
OUT_FLAG = "--out"

_FLAGS = (MODEL_FLAG, OUT_FLAG)


def describe_weights(model_id: str) -> str:
    """Load a model and render its digest and parameter inventory.

    Args:
        model_id: The hub id.

    Returns:
        Canonical JSON: ``model``, ``weights_digest``, and ``parameters`` as
        ``[name, shape]`` pairs in the model's own order.

    Raises:
        AppError: Propagated from the loader.
    """
    model, _encoder = load_editable_lm(model_id)
    inventory: list[JSONValue] = []
    for name, parameter in model.named_parameters():
        shape: list[JSONValue] = [int(size) for size in parameter.detach().shape]
        inventory.append([name, shape])
    return dump_json_str(
        {
            "model": model_id,
            "weights_digest": weights_digest(parameter_digests(model)),
            "parameters": inventory,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Describe one model's weights.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the description is written.

    Raises:
        ValueError: When a flag is unknown, repeated or missing.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    model_id = cli_args.require_flag(parsed, MODEL_FLAG)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(describe_weights(model_id), encoding="utf-8")
    _log.info("described %s -> %s", model_id, out)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = ["describe_weights", "entrypoint", "main"]


# Without this, `python -m model_trainer.cli.weight_space_lm_digest` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
