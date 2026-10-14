"""Where Chameleon runs: CUDA on Kaggle/Colab (unchanged behaviour), else Apple MPS or the CPU (PLAN §16).

RR_DEVICE=cuda|mps|cpu overrides the automatic choice. On CUDA every helper is exactly the call it replaces
(torch.cuda.synchronize, max_memory_allocated, ...). Elsewhere `use_device()` redirects the starter's hard-coded
`.cuda()` calls (starter_utils.load_chameleon, its predict helpers) to the chosen device.
"""
import os

import torch


def _auto():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = os.environ.get('RR_DEVICE') or _auto()


def use_device():
    """Route Module.cuda() / Tensor.cuda() to DEVICE when it is not CUDA (no-op on CUDA)."""
    if DEVICE == 'cuda' or getattr(torch.Tensor, '_rr_device', None) == DEVICE:
        return
    to_dev = lambda self, device=None: self.to(DEVICE)         # noqa: E731
    torch.nn.Module.cuda = to_dev
    try:                                       # Lightning modules override .cuda() (DeviceDtypeModuleMixin)
        from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
        _DeviceDtypeModuleMixin.cuda = to_dev
    except ImportError:
        pass
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False, **kw: self.to(DEVICE, non_blocking=non_blocking)
    torch.Tensor._rr_device = DEVICE


def sync():
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elif DEVICE == 'mps':
        torch.mps.synchronize()


def reset_peak():
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    """peak allocated memory (CUDA); MPS: memory the driver holds now; CPU: 0"""
    if DEVICE == 'cuda':
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    if DEVICE == 'mps':
        return torch.mps.driver_allocated_memory() / 1024 ** 3
    return 0.0


def empty_cache():
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE == 'mps':
        torch.mps.empty_cache()


def name():
    if DEVICE == 'cuda':
        return torch.cuda.get_device_name(0)
    return {'mps': 'Apple MPS', 'cpu': 'CPU'}[DEVICE]


def total_gb():
    if DEVICE == 'cuda':
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    import subprocess
    try:
        return int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout) / 1024 ** 3
    except (OSError, ValueError):
        return 0.0
