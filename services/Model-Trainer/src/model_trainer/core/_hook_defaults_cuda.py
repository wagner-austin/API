"""Hook defaults for the CUDA surface - production defaults, tests override.

Split from :mod:`model_trainer.core._hook_defaults` when that module crossed
the 600-line ceiling; the CUDA adapters move together because they share one
contract the rest of the defaults do not have: **whether the call is safe is
the caller's question.** Every function here either initialises a CUDA
context or requires one, so callers gate on ``cuda_is_available`` (or on the
operands' device) first, and none of these repeats the check -- a repeated
check is a branch no caller can reach and no machine with a GPU can execute.
"""

from __future__ import annotations

import torch


def _default_cuda_is_available() -> bool:
    """Production cuda_is_available - used as default hook."""
    return torch.cuda.is_available()


def _default_cuda_device_name() -> str:
    """Production cuda_device_name - used as default hook.

    Callers gate on the run's device being "cuda" (which _setup_device has
    already proven available); repeating the check here would hide a caller
    that forgot the gate. Calling this initialises a CUDA context in the
    process, which is exactly why cpu-device runs must not reach it.
    """
    return torch.cuda.get_device_name(0)


def _default_sdpa_cuda_eligibility(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> dict[str, bool]:
    """Production sdpa_cuda_eligibility - used as default hook.

    Callers gate on the operands being CUDA tensors: under torch 2.7,
    ``can_use_cudnn_attention`` initialises a CUDA context even for CPU
    operands, so reaching this on a driverless host is fatal rather than
    merely wasteful.
    """
    import torch.backends.cuda as backends_cuda

    params = backends_cuda.SDPAParams(query, key, value, None, 0.0, True, False)
    return {
        "flash": backends_cuda.can_use_flash_attention(params),
        "efficient": backends_cuda.can_use_efficient_attention(params),
        "cudnn": backends_cuda.can_use_cudnn_attention(params),
    }


def _default_cuda_driver_version() -> str:
    """Production cuda_driver_version - used as default hook.

    Read from ``nvidia-smi`` rather than from torch. ``torch.version.cuda``
    is the CUDA runtime the wheel was BUILT against (12.4 here) and is not
    the driver; reporting it as one would put a wrong value in a field whose
    whole purpose is telling two otherwise-identical configurations apart.
    torch 2.6 exposes no public driver accessor -- everything NVML-side under
    ``torch.cuda`` is underscore-private.

    Callers gate on the run's device being "cuda", which means CUDA
    initialised, which means the driver answered. A failure here is therefore
    a real fault and propagates: a fingerprint that quietly records "unknown"
    for a run that HAD a driver would make two different configurations
    compare equal, which is the one outcome this field exists to prevent.

    Returns:
        The NVIDIA driver version, e.g. ``"591.86"``.

    Raises:
        CalledProcessError: When nvidia-smi exits non-zero.
        FileNotFoundError: When nvidia-smi is not present.
    """
    import subprocess as _sp

    out = _sp.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        stderr=_sp.DEVNULL,
    )
    return out.decode("utf-8").strip().splitlines()[0].strip()


def _default_torch_cuda_max_memory_allocated() -> int:
    """Production torch.cuda.max_memory_allocated - used as default hook.

    A thin adapter over torch. Whether CUDA is present is the caller's
    question, and `_default_gpu_max_memory_allocated` already asks it through
    the `cuda_is_available` hook before delegating here; repeating the check
    added a branch that no caller can reach and that no machine with a GPU can
    execute.

    Returns:
        Peak GPU memory allocated in bytes.
    """
    return torch.cuda.max_memory_allocated()


def _default_torch_cuda_reset_peak_memory_stats() -> None:
    """Production torch.cuda.reset_peak_memory_stats - used as default hook.

    A thin adapter over torch; `_default_gpu_reset_peak_memory_stats` owns the
    availability check.
    """
    torch.cuda.reset_peak_memory_stats()


def _default_torch_cuda_get_rng_state_all() -> list[torch.Tensor]:
    """Production torch.cuda.get_rng_state_all - used as default hook.

    A thin adapter over torch; the checkpoint capture owns the
    availability check through the ``cuda_is_available`` hook.
    """
    return list(torch.cuda.get_rng_state_all())


def _default_torch_cuda_set_rng_state_all(states: list[torch.Tensor]) -> None:
    """Production torch.cuda.set_rng_state_all - used as default hook.

    Args:
        states: States previously returned by
            ``torch.cuda.get_rng_state_all``.
    """
    torch.cuda.set_rng_state_all(states)


def _default_torch_device(device_str: str) -> torch.device:
    """Production torch.device - used as default hook."""
    return torch.device(device_str)
