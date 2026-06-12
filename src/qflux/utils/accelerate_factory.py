"""Factory for constructing the right Accelerator for the active backend.

On the CUDA/CPU path this returns a vanilla ``accelerate.Accelerator`` with the
exact same arguments the codebase used before (behaviour unchanged).

On the XLA (AWS Trainium) path it returns ``optimum.neuron.NeuronAccelerator``,
a drop-in subclass of ``accelerate.Accelerator`` that maps ``backward`` /
``accumulate`` / ``gather`` / ``prepare_*`` onto torch_xla. The ``optimum``
import is lazy so this module imports cleanly on machines without it.

NeuronAccelerator differences worth noting:
- ``mixed_precision="bf16"`` lets it drive bf16 autocast (the CUDA path keeps
  ``mixed_precision="no"`` and pre-casts weights instead).
- ``zero_1=True`` shards optimizer state (ZeRO stage 1) — optional, cheap win
  for larger optimizer states; LoRA states are tiny so it defaults off.
"""

from __future__ import annotations

from qflux.utils import backend


def build_accelerator(
    *,
    gradient_accumulation_steps: int,
    project_config,
    neuron_config=None,
):
    """Return an Accelerator appropriate for the resolved backend.

    Parameters mirror the subset of ``accelerate.Accelerator`` arguments this
    codebase relies on. ``neuron_config`` is a ``qflux.data.config.NeuronConfig``
    (only consulted on the XLA path).
    """
    if backend.is_xla():
        # Lazy import: optimum-neuron is only installed in the Neuron env.
        from optimum.neuron import NeuronAccelerator  # noqa: PLC0415

        mixed_precision = "bf16" if (neuron_config is None or neuron_config.bf16) else "no"
        zero_1 = bool(neuron_config.zero1) if neuron_config is not None else False
        return NeuronAccelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            zero_1=zero_1,
            project_config=project_config,
        )

    # CUDA / CPU path — identical to the original construction.
    from accelerate import Accelerator  # noqa: PLC0415

    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="no",  # ← 关键: weights are pre-cast to bf16 on the GPU path
        project_config=project_config,
    )
