#!/bin/bash
#
# Qwen Image Finetune — AWS Trainium (Neuron) environment setup.
# Usage: ./setup_neuron.sh [HF_TOKEN]
#
# Run this on a trn1 / trn2 instance launched from the AWS Neuron DLAMI (or with
# the Neuron driver/runtime already installed). It creates a Python venv and
# installs the Neuron PyTorch stack (torch-neuronx) plus this project's
# Neuron-compatible requirements. It does NOT install CUDA/bitsandbytes/TE.

set -euo pipefail

HF_TOKEN=${1:-${HF_TOKEN:-}}

# Pin versions to a known-good Neuron SDK release. Bump together when upgrading.
TORCH_NEURONX_VERSION="2.5.*"
NEURONX_CC_VERSION="2.*"
NEURON_PIP_INDEX="https://pip.repos.neuron.amazonaws.com"

echo "=== Setting up Qwen Image Finetune for AWS Trainium (Neuron) ==="

# 1) Verify the Neuron runtime is present (driver + neuron-rt).
if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "WARNING: 'neuron-ls' not found. Make sure the Neuron driver/runtime is installed"
    echo "         (use the AWS Neuron DLAMI, or follow the Neuron setup guide)."
else
    echo "Detected NeuronCores:"
    neuron-ls || true
fi

# 2) Python venv.
PYTHON=${PYTHON:-python3}
if [ ! -d ".venv-neuron" ]; then
    echo "Creating venv .venv-neuron ..."
    "$PYTHON" -m venv .venv-neuron
fi
# shellcheck disable=SC1091
source .venv-neuron/bin/activate
pip install --upgrade pip setuptools wheel

# 3) Install the Neuron PyTorch stack from the AWS Neuron pip index.
echo "Installing torch-neuronx / neuronx-cc from the Neuron pip index ..."
pip install --upgrade \
    "torch-neuronx==${TORCH_NEURONX_VERSION}" \
    "neuronx-cc==${NEURONX_CC_VERSION}" \
    libneuronxla \
    --extra-index-url "${NEURON_PIP_INDEX}"

# 4) optimum-neuron (NeuronAccelerator) + the project's Neuron requirements.
pip install --upgrade optimum-neuron --extra-index-url "${NEURON_PIP_INDEX}"
pip install -r requirements-neuron.txt
pip install -e .

# 5) Optional Hugging Face auth.
if [ -n "${HF_TOKEN}" ]; then
    python - <<PY
from huggingface_hub import login
login(token="${HF_TOKEN}")
print("Hugging Face auth OK")
PY
else
    echo "No HF_TOKEN provided; run 'huggingface-cli login' later if needed."
fi

cat <<'EOF'

=== Setup complete ===

Next steps (on the trn instance):

  source .venv-neuron/bin/activate
  export QFLUX_BACKEND=xla            # or rely on autodetect (NEURON_RT_VISIBLE_CORES)

  # 1) Build the embedding cache once (encode VAE/text on a fixed resolution):
  python -m qflux.main --config configs/qwen_image_edit_neuron_bf16.yaml --cache

  # 2) Precompile the training graph (avoids slow first-step compiles), then train:
  ./run_neuron.sh

NOTE: actual compilation and training require Trainium hardware. The first
real training step triggers neuronx-cc compilation (minutes). Keep input shapes
fixed (batch size, image resolution, text length) so the graph is compiled once.
EOF
