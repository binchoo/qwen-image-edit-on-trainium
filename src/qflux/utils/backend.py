"""Hardware backend abstraction layer.

This module centralizes all hardware/device specific logic so that the rest of
the codebase does not need to call ``torch.cuda.*`` directly. It supports three
families of backends:

- ``cuda``  : NVIDIA GPU (the original, unchanged behaviour)
- ``xla``   : AWS Trainium / Inferentia via ``torch-neuronx`` (torch_xla)
- ``mps`` / ``cpu`` : fallbacks (mainly for local development / static checks)

Design goals:

1. **The CUDA path must be byte-for-byte unchanged.** When the backend resolves
   to ``cuda`` every helper forwards to the exact ``torch.cuda.*`` call it
   replaced.
2. **Importable without torch_xla.** All ``torch_xla`` imports are lazy and
   guarded so this module imports cleanly on macOS / CPU-only machines.
3. **Single source of truth.** Backend selection happens once and is cached.

Backend selection order:
    1. ``QFLUX_BACKEND`` env var (explicit override: cuda|xla|mps|cpu)
    2. Autodetect XLA: ``torch_xla`` importable AND a Neuron runtime env var set
       (``NEURON_RT_VISIBLE_CORES``) -> ``xla``
    3. ``torch.cuda.is_available()`` -> ``cuda``
    4. ``torch.backends.mps.is_available()`` -> ``mps``
    5. ``cpu``
"""

from __future__ import annotations

import contextlib
import importlib.util
import os

import torch


# Valid backend identifiers.
CUDA = "cuda"
XLA = "xla"
MPS = "mps"
CPU = "cpu"

_VALID_BACKENDS = {CUDA, XLA, MPS, CPU}

# Cached resolved backend (resolved lazily on first use).
_backend: str | None = None


def _xla_available() -> bool:
    """Return True if torch_xla can be imported."""
    return importlib.util.find_spec("torch_xla") is not None


def _detect_backend() -> str:
    """Resolve the active backend following the documented selection order."""
    override = os.environ.get("QFLUX_BACKEND", "").strip().lower()
    if override:
        if override not in _VALID_BACKENDS:
            raise ValueError(
                f"Invalid QFLUX_BACKEND={override!r}. Expected one of {sorted(_VALID_BACKENDS)}."
            )
        return override

    # Autodetect Neuron/XLA: torch_xla present and a Neuron runtime hint set.
    if _xla_available() and (
        os.environ.get("NEURON_RT_VISIBLE_CORES") is not None
        or os.environ.get("NEURON_RT_NUM_CORES") is not None
    ):
        return XLA

    if torch.cuda.is_available():
        return CUDA

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return MPS

    return CPU


def get_backend() -> str:
    """Return the resolved backend string, caching the result."""
    global _backend
    if _backend is None:
        _backend = _detect_backend()
    return _backend


def set_backend(name: str) -> None:
    """Override the backend explicitly (mainly for tests / static checks)."""
    name = name.strip().lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend {name!r}. Expected one of {sorted(_VALID_BACKENDS)}.")
    global _backend
    _backend = name


def is_xla() -> bool:
    return get_backend() == XLA


def is_cuda() -> bool:
    return get_backend() == CUDA


def _xm():
    """Lazily import torch_xla.core.xla_model. Only call when backend is xla."""
    import torch_xla.core.xla_model as xm  # noqa: PLC0415  (intentional lazy import)

    return xm


def device(index: int | None = None) -> torch.device:
    """Return the canonical training device for the active backend.

    On XLA the index is ignored (the runtime assigns the local NeuronCore via
    the ordinal); callers that previously hardcoded ``cuda:1`` etc. should pass
    no index so the backend resolves the correct device.
    """
    backend = get_backend()
    if backend == XLA:
        return _xm().xla_device()
    if backend == CUDA:
        return torch.device("cuda" if index is None else f"cuda:{index}")
    if backend == MPS:
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache() -> None:
    """Free cached device memory.

    No-op on XLA (the XLA runtime manages its own memory and ``torch.cuda.*``
    would raise). Forwards to the real CUDA call on GPU so behaviour is
    unchanged there.
    """
    if get_backend() == CUDA:
        torch.cuda.empty_cache()


def synchronize() -> None:
    """Block until queued device work completes.

    On XLA this maps to ``mark_step()`` (the closest equivalent sync point).
    No-op on CPU/MPS. Forwards to ``torch.cuda.synchronize`` on GPU.
    """
    backend = get_backend()
    if backend == CUDA:
        torch.cuda.synchronize()
    elif backend == XLA:
        _xm().mark_step()


def mark_step() -> None:
    """Execute the accumulated XLA graph. No-op on every other backend.

    This is the critical graph-execution boundary for XLA training and should
    be called exactly once per optimizer step.
    """
    if get_backend() == XLA:
        _xm().mark_step()


def manual_seed_all(seed: int) -> None:
    """Seed all devices for the active backend."""
    backend = get_backend()
    if backend == CUDA:
        torch.cuda.manual_seed_all(seed)
    elif backend == XLA:
        # torch_xla seeds via the global torch RNG forwarded to the device.
        _xm().set_rng_state(seed)


def device_context(index_or_device=None):
    """Context manager analogous to ``torch.cuda.device(idx)``.

    Accepts either an integer index or a ``torch.device``. Returns a real CUDA
    device context on GPU; a null context everywhere else (XLA has no per-index
    device context manager).
    """
    if get_backend() == CUDA and index_or_device is not None:
        return torch.cuda.device(index_or_device)
    return contextlib.nullcontext()
