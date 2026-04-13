"""Determinism mode control for the MountainSort5 PyTorch pipeline.

Provides three determinism levels that can be applied globally before
running a pipeline:

- ``"none"``    — default GPU behavior; outputs may vary run-to-run.
- ``"relaxed"`` — seeds fixed, cuDNN deterministic, but TF32 allowed.
                  Goal: final spike times + labels are stable across runs,
                  intermediate tensors may differ bitwise.
- ``"full"``    — strict bitwise reproducibility.  All known sources of
                  nondeterminism are locked down (TF32 off, single-threaded
                  BLAS/OMP, ``use_deterministic_algorithms(True)``).

Usage::

    from torched_mountainsort5.determinism import set_determinism
    set_determinism("full")
    # ... run pipeline ...

The relaxed/full distinction mirrors the framework in:
    Lucman et al., "Porting spike sorting algorithms to PyTorch" (ASPLOS 2027).

See also the ``set_deterministic()`` block from the companion thesis
on Kilosort4/3/RT-Sort, which this module is modeled after.
"""

import os
import random
from typing import Literal

import numpy as np
import torch

# cuBLAS reads CUBLAS_WORKSPACE_CONFIG only once, at initialization time.
# Setting it here guarantees it is visible before any CUDA context is created,
# regardless of which determinism mode is selected later.  The variable is
# harmless when use_deterministic_algorithms is False.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

DeterminismMode = Literal["none", "relaxed", "full"]


def set_seeds(seed: int = 1) -> None:
    """Fix all RNG seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CuPy is optional — seed it only if available.
    try:
        import cupy as cp
        cp.random.seed(seed)
    except ImportError:
        pass


def set_determinism(mode: DeterminismMode, seed: int = 1) -> None:
    """Apply a determinism level globally before running a pipeline.

    Parameters
    ----------
    mode : ``"none"`` | ``"relaxed"`` | ``"full"``
        Determinism level.  See module docstring for details.
    seed : int
        RNG seed used for ``"relaxed"`` and ``"full"`` modes.
    """
    if mode == "none":
        _apply_none()
    elif mode == "relaxed":
        _apply_relaxed(seed)
    elif mode == "full":
        _apply_full(seed)
    else:
        raise ValueError(f"Unknown determinism mode: {mode!r}. "
                         f"Expected 'none', 'relaxed', or 'full'.")


# ── Mode implementations ────────────────────────────────────────────────


def _apply_none() -> None:
    """Restore default (non-deterministic) GPU behavior.

    Note: CUBLAS_WORKSPACE_CONFIG is set at module level and cannot be
    undone — cuBLAS reads it once at init.  This pins one source of
    non-determinism even in "none" mode, but all other sources (cuDNN
    benchmarking, TF32, unseeded RNGs, threading) remain active, so the
    pipeline is still non-deterministic overall.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Restore default threading.
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ.pop("MKL_NUM_THREADS", None)
    os.environ.pop("NUMEXPR_NUM_THREADS", None)


def _apply_relaxed(seed: int) -> None:
    """Seed everything and stabilise cuDNN, but leave TF32 on.

    This targets *final-output* reproducibility: spike times and labels
    should be identical across runs even though intermediate floating-point
    values may not be bitwise equal.
    """
    set_seeds(seed)

    # Deterministic cuDNN algorithm selection (no benchmark search).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Do NOT force use_deterministic_algorithms — some ops may lack a
    # deterministic CUDA implementation and we don't want to error out.
    torch.use_deterministic_algorithms(False)

    # TF32 stays on — small rounding differences are acceptable as long as
    # the final decision boundaries (argmax, threshold) are not affected.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Fix cuBLAS workspace so matmul kernel selection is stable.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _apply_full(seed: int) -> None:
    """Lock down every known source of GPU nondeterminism.

    After calling this, ``torch.use_deterministic_algorithms(True)`` is
    active — any op without a deterministic implementation will raise a
    ``RuntimeError``.  The caller must handle or work around those errors
    (e.g. by moving the offending op to CPU).
    """
    set_seeds(seed)

    # cuDNN: deterministic algorithm, no benchmarking.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Global enforcement — errors on nondeterministic ops.
    torch.use_deterministic_algorithms(True)

    # Disable TF32 everywhere.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Fix cuBLAS workspace.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"

    # Single-thread CPU-side parallelism to eliminate tie-break races.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads can only be called once; ignore if
        # already set from a previous call.
        pass
