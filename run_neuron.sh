#!/bin/bash
#
# Launch Qwen-Image-Edit LoRA training on AWS Trainium (Neuron).
#
# Two phases:
#   1. neuron_parallel_compile  — traces the graph and pre-populates the compile
#      cache WITHOUT running real training (fast, no weight updates). This avoids
#      a multi-minute stall on the first real step.
#   2. torchrun                 — the actual training run, reusing the warm cache.
#
# Set NUM_CORES to the number of NeuronCores to use for data-parallel training
# (e.g. 2 on trn1.2xlarge, up to 32 on trn1.32xlarge).

set -euo pipefail

CONFIG=${CONFIG:-configs/qwen_image_edit_neuron_bf16.yaml}
NUM_CORES=${NUM_CORES:-1}

export QFLUX_BACKEND=xla
export WANDB_MODE=${WANDB_MODE:-offline}
# Keep compiled graphs cached across runs.
export NEURON_COMPILE_CACHE_URL=${NEURON_COMPILE_CACHE_URL:-/var/tmp/neuron-compile-cache}

echo "=== Phase 1/2: neuron_parallel_compile (graph warm-up, no weight updates) ==="
neuron_parallel_compile \
  torchrun --nproc_per_node="${NUM_CORES}" \
  -m qflux.main --config "${CONFIG}" || {
    echo "neuron_parallel_compile not available or failed; skipping warm-up."
    echo "(The first real step will compile instead — slower but correct.)"
  }

echo "=== Phase 2/2: training ==="
torchrun --nproc_per_node="${NUM_CORES}" \
  -m qflux.main --config "${CONFIG}"
