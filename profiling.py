"""Model and system profiling: params, FLOPs, activation memory, latency.

Used by ``python profiling.py`` (standalone) and by the auto-downgrader.

FLOPs convention: one FLOP = one multiply-accumulate counted as 2 ops is NOT
used; we count multiply-accumulates (MACs) and call them FLOPs, which is the
convention most CNN papers use.  The number is an *estimate* for comparison
between profiles, not a cycle-accurate cost model.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import torch
import torch.nn as nn

from metrics import stats
from models import PROFILES, DuelingDQN, count_trainable_params


def set_cpu_threads(threads: int = 1) -> None:
    """Pin torch to a single thread so Chrome/capture keep CPU headroom."""
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError as exc:
        # Already initialised after work started in this process; safe to skip.
        logging.getLogger("ssbot.profiling").debug(
            "interop threads already set: %s", exc)


def estimate_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Estimate multiply-accumulate count for one forward pass via hooks."""
    flops = [0]
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(module: nn.Conv2d, inputs: tuple, output: torch.Tensor) -> None:
        out_h, out_w = output.shape[2], output.shape[3]
        kh, kw = module.kernel_size
        cin = module.in_channels // module.groups
        per_out = kh * kw * cin
        flops[0] += int(per_out * out_h * out_w * output.shape[1])

    def linear_hook(module: nn.Linear, inputs: tuple, output: torch.Tensor) -> None:
        flops[0] += int(module.in_features * module.out_features)

    def pool_hook(module: nn.AdaptiveAvgPool2d, inputs: tuple, output: torch.Tensor) -> None:
        # GAP over HxW with C channels: ~C*H*W accumulations.
        x = inputs[0]
        flops[0] += int(x.shape[1] * x.shape[2] * x.shape[3])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            handles.append(m.register_forward_hook(pool_hook))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, *input_shape, device=device))
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return flops[0]


def estimate_activation_memory_mb(model: nn.Module, input_shape: tuple[int, ...]) -> float:
    """Peak intermediate activation bytes for batch=1 (via autograd graph)."""
    act_bytes = [0]

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        if isinstance(output, torch.Tensor):
            act_bytes[0] += output.numel() * output.element_size()
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    act_bytes[0] += o.numel() * o.element_size()

    # DEEP-FIX: hook LEAF modules only.  The old selector matched the composite
    # containers (ConvBlock / DepthwiseSeparableConv, matched by class name)
    # AND their inner Conv2d children (matched by isinstance).  A container's
    # output tensor is identical to its last child's output, so it was counted
    # twice -- inflating the activation-memory estimate used to size profiles
    # for a 12 GB / i5-7200U box.  Leaves only yields a clean sum of every
    # intermediate activation.
    for m in model.modules():
        if next(m.children(), None) is not None:
            continue                      # container; its leaves are hooked below
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.GroupNorm, nn.ReLU,
                          nn.AdaptiveAvgPool2d)):
            handles.append(m.register_forward_hook(hook))
    device = next(model.parameters()).device
    was_training = model.training
    model.train()
    x = torch.zeros(1, *input_shape, device=device, requires_grad=True)
    out = model(x)
    if isinstance(out, tuple):
        loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor))
    else:
        loss = out.sum()
    loss.backward()
    for h in handles:
        h.remove()
    if not was_training:
        model.eval()
    return act_bytes[0] / (1024 ** 2)


def measure_inference_latency_ms(
    model: nn.Module,
    input_shape: tuple[int, ...],
    warmup: int = 10,
    iters: int = 60,
) -> dict[str, float]:
    """Single-thread CPU inference latency (ms) — p50/p95/p99 over iters."""
    set_cpu_threads(1)
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, *input_shape, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        samples: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return stats(samples)


def profile_model(profile_name: str, frame_stack: int = 4, size: int = 84) -> dict[str, Any]:
    """Full report for one model profile: params, FLOPs, memory, latency."""
    torch.manual_seed(0)
    model = DuelingDQN.from_profile(profile_name, in_frames=frame_stack, size=size)
    shape = (frame_stack, size, size)
    params = count_trainable_params(model)
    report: dict[str, Any] = {
        "profile": profile_name,
        "params": params,
        "flops": estimate_flops(model, shape),
        "activation_mb_train": estimate_activation_memory_mb(model, shape),
        "inference": measure_inference_latency_ms(model, shape),
    }
    del model
    return report


def profile_all_profiles(frame_stack: int = 4, size: int = 84) -> list[dict[str, Any]]:
    return [profile_model(p, frame_stack, size) for p in PROFILES]


def measure_training_update_ms(
    model: nn.Module,
    batch: int = 32,
    in_frames: int = 4,
    size: int = 84,
    iters: int = 10,
) -> dict[str, float]:
    """Rough per-update cost (forward+backward+optimizer) at batch size."""
    set_cpu_threads(1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(batch, in_frames, size, size)
    target = torch.randn(batch, 5)
    samples: list[float] = []
    model.train()
    for i in range(iters + 2):
        t0 = time.perf_counter()
        out = model(x)
        loss = torch.nn.functional.smooth_l1_loss(out, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i >= 2:
            samples.append((time.perf_counter() - t0) * 1000.0)
    return stats(samples)


def format_profile_table(reports: list[dict[str, Any]]) -> str:
    header = (
        f"{'profile':<14}{'params':>10}{'FLOPs':>12}"
        f"{'act MB':>8}{'p50 ms':>9}{'p95 ms':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        lines.append(
            f"{r['profile']:<14}{r['params']:>10,d}{r['flops']:>12,d}"
            f"{r['activation_mb_train']:>8.3f}"
            f"{r['inference']['p50']:>9.3f}{r['inference']['p95']:>9.3f}"
        )
    return "\n".join(lines)


class SystemProfiler:
    """Context manager that samples CPU/RAM and reports the peak."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.peak_ram_gb = 0.0
        self.cpu_samples: list[float] = []
        self._stop = False

    def __enter__(self) -> "SystemProfiler":
        import threading

        from metrics import system_usage

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._system_usage = system_usage  # keep reference for typing
        return self

    def _run(self) -> None:
        import time as _time

        from metrics import system_usage

        while not self._stop:
            usage = system_usage()
            self.peak_ram_gb = max(self.peak_ram_gb, usage["ram_process_gb"])
            self.cpu_samples.append(usage["cpu_process"])
            _time.sleep(self.interval_s)

    def __exit__(self, *exc: Any) -> None:
        self._stop = True
        if getattr(self, "_thread", None) is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, float]:
        return {
            "peak_ram_gb": self.peak_ram_gb,
            "cpu_mean": sum(self.cpu_samples) / len(self.cpu_samples)
            if self.cpu_samples
            else 0.0,
            "cpu_max": max(self.cpu_samples) if self.cpu_samples else 0.0,
        }


def main() -> int:
    """CLI: ``python profiling.py [--batch 32]``."""
    import argparse

    from logging_utils import setup_logging

    parser = argparse.ArgumentParser(description="Model/system profiling")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    setup_logging("profiling", "logs")
    set_cpu_threads(1)
    print("Profiling model profiles (CPU, torch num_threads=1)...")
    reports = profile_all_profiles()
    print(format_profile_table(reports))
    if not args.skip_training:
        print(f"\nTraining-update cost at batch={args.batch} (ms):")
        from models import DuelingDQN

        for name in PROFILES:
            model = DuelingDQN.from_profile(name)
            tstats = measure_training_update_ms(model, batch=args.batch)
            print(
                f"  {name:<14} p50={tstats['p50']:8.2f}  p95={tstats['p95']:8.2f}"
            )
            del model
    print("\nNOTE: numbers above are measured on THIS machine.")
    print("The target machine is an i5-7200U; re-run this script there for")
    print("authoritative numbers before selecting the default profile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
