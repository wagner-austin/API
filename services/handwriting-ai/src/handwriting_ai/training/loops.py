from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as functional
from platform_core.logging import get_logger
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from handwriting_ai.monitoring import check_memory_pressure, get_memory_snapshot

from .memory_diagnostics import log_memory_diagnostics
from .progress import emit_batch as _emit_batch
from .safety import get_memory_guard_config, on_batch_check
from .train_utils import bytes_of_model_and_grads, torch_allocator_stats


class _BatchIterator(Protocol):
    """Iterator protocol for batch iteration."""

    def __iter__(self) -> _BatchIterator: ...
    def __next__(self) -> tuple[Tensor, Tensor]: ...


class _BatchLoader(Protocol):
    """Protocol for data loaders that yield (tensor, tensor) batches."""

    def __iter__(self) -> _BatchIterator: ...
    def __len__(self) -> int: ...


def evaluate(model: Module, loader: _BatchLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        batch: tuple[Tensor, Tensor]
        for batch in loader:
            x: Tensor = batch[0]
            y: Tensor = batch[1]
            logits: Tensor = model(x.to(device))
            preds: Tensor = logits.argmax(dim=1)
            correct += int((preds.cpu() == y).sum().item())
            total += y.size(0)
    return (correct / total) if total > 0 else 0.0


def train_epoch(
    model: Module,
    train_loader: _BatchLoader,
    device: torch.device,
    optimizer: Optimizer,
    *,
    ep: int,
    ep_total: int,
    total_batches: int,
) -> float:
    import time as _time

    log = get_logger("handwriting_ai")
    model.train()
    total = 0
    loss_sum = 0.0
    batch_idx: int = 0
    batch: tuple[Tensor, Tensor]
    for batch_idx, batch in enumerate(train_loader):
        t0 = _time.perf_counter()
        x: Tensor = batch[0]
        y: Tensor = batch[1]
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits: Tensor = model(x)
        loss: Tensor = functional.cross_entropy(logits, y)
        torch.autograd.backward((loss,))
        optimizer.step()
        # Proactive memory guard: check every batch (not only on log cadence)
        if on_batch_check():
            log.info("mem_guard_abort e=%s b=%s", ep, (batch_idx + 1))
            raise RuntimeError("memory_pressure_guard_triggered")
        total += y.size(0)
        loss_sum += float(loss.item()) * y.size(0)
        # Local cadence check optimized for performance (avoids expensive ops every batch)
        # Aligned with batch NUMBER not batch_idx for consistency with progress.emit_batch()
        batch_num = batch_idx + 1
        log_every = 10
        if (batch_num % log_every == 0) or (batch_num == 1) or (batch_num == total_batches):
            avg_loss: float = loss_sum / total if total > 0 else 0.0
            with torch.no_grad():
                preds: Tensor = logits.argmax(dim=1)
                batch_acc = float((preds == y).float().mean().item())
            dt = _time.perf_counter() - t0
            ips = (int(y.size(0)) / dt) if dt > 0 else 0.0
            snap = get_memory_snapshot()
            _mg = get_memory_guard_config()
            thr = float(_mg["threshold_percent"])
            pressed = check_memory_pressure(threshold_percent=thr)
            press = "true" if pressed else "false"

            # Model and gradient memory
            param_b, grad_b = bytes_of_model_and_grads(model)  # bytes
            model_param_mb = int(param_b // (1024 * 1024))
            grad_param_mb = int(grad_b // (1024 * 1024))

            # CUDA allocator stats (CPU-only envs report zeros)
            cuda_ok, cuda_alloc_b, cuda_reserved_b, cuda_max_alloc_b = torch_allocator_stats()
            cuda_alloc_mb = int(cuda_alloc_b // (1024 * 1024)) if cuda_ok else 0
            cuda_reserved_mb = int(cuda_reserved_b // (1024 * 1024)) if cuda_ok else 0
            cuda_max_alloc_mb = int(cuda_max_alloc_b // (1024 * 1024)) if cuda_ok else 0

            main_mb = snap["main_process"]["rss_bytes"] // (1024 * 1024)
            workers_mb = sum(w["rss_bytes"] for w in snap["workers"]) // (1024 * 1024)
            cgroup_usage_mb = snap["cgroup_usage"]["usage_bytes"] // (1024 * 1024)
            cgroup_limit_mb = snap["cgroup_usage"]["limit_bytes"] // (1024 * 1024)
            anon_mb = snap["cgroup_breakdown"]["anon_bytes"] // (1024 * 1024)
            file_mb = snap["cgroup_breakdown"]["file_bytes"] // (1024 * 1024)

            log.info(
                f"train_batch_done epoch={ep}/{ep_total} "
                f"batch={batch_num}/{total_batches} "
                f"batch_loss={float(loss.item()):.4f} batch_acc={batch_acc:.4f} "
                f"avg_loss={avg_loss:.4f} samples_per_sec={ips:.1f} "
                f"main_rss_mb={main_mb} workers_rss_mb={workers_mb} "
                f"worker_count={len(snap['workers'])} "
                f"cgroup_usage_mb={cgroup_usage_mb} cgroup_limit_mb={cgroup_limit_mb} "
                f"cgroup_pct={snap['cgroup_usage']['percent']:.1f} "
                f"anon_mb={anon_mb} file_mb={file_mb} "
                f"mem_pressure={press} guard_threshold={thr:.1f} "
                f"model_param_mb={model_param_mb} grad_param_mb={grad_param_mb} "
                f"cuda_alloc_mb={cuda_alloc_mb} cuda_reserved_mb={cuda_reserved_mb} "
                f"cuda_max_alloc_mb={cuda_max_alloc_mb}"
            )
            # Supplementary diagnostics for trend and prediction using the same snapshot
            log_memory_diagnostics(context="train_batch", snapshot=snap)
            _emit_batch(
                {
                    "epoch": ep,
                    "total_epochs": ep_total,
                    "batch": batch_num,
                    "total_batches": total_batches,
                    "batch_loss": float(loss.item()),
                    "batch_acc": batch_acc,
                    "avg_loss": avg_loss,
                    "samples_per_sec": ips,
                    "main_rss_mb": main_mb,
                    "workers_rss_mb": workers_mb,
                    "worker_count": len(snap["workers"]),
                    "cgroup_usage_mb": cgroup_usage_mb,
                    "cgroup_limit_mb": cgroup_limit_mb,
                    "cgroup_pct": snap["cgroup_usage"]["percent"],
                    "anon_mb": anon_mb,
                    "file_mb": file_mb,
                }
            )
    return loss_sum / total if total > 0 else 0.0
