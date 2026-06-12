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

NEURON_PIP_INDEX="https://pip.repos.neuron.amazonaws.com"

# We do NOT pin torch / torch-neuronx / neuronx-cc by hand. Installing them
# individually is what caused the earlier breakage (a CUDA torch '+cu124' got
# pulled from PyPI, which then forced an OLD optimum-neuron without
# NeuronAccelerator). Instead we let `optimum-neuron[neuronx]` resolve the whole
# Neuron stack in ONE consolidated, mutually-compatible install (the approach the
# HF docs prescribe). It pins torch (Neuron build), torch-neuronx, neuronx-cc,
# libneuronxla, transformers and accelerate together.

echo "=== Setting up Qwen Image Finetune for AWS Trainium (Neuron) ==="

# 1) Verify the Neuron runtime is present (driver + neuron-rt).
if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "WARNING: 'neuron-ls' not found. Make sure the Neuron driver/runtime is installed"
    echo "         (use the AWS Neuron DLAMI, or follow the Neuron setup guide)."
else
    echo "Detected NeuronCores:"
    neuron-ls || true
fi

# 2) Python venv — fresh every time, to avoid inheriting a broken/CUDA torch.
PYTHON=${PYTHON:-python3}
echo "Python interpreter: $("$PYTHON" --version 2>&1) at $(command -v "$PYTHON")"
if [ -d ".venv-neuron" ]; then
    echo "Removing existing .venv-neuron for a clean install ..."
    rm -rf .venv-neuron
fi
"$PYTHON" -m venv .venv-neuron
# shellcheck disable=SC1091
source .venv-neuron/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Point pip at the Neuron index via an env var (scoped to this script run only,
# not written to the user's global pip config) so the [neuronx] extra can resolve
# torch-neuronx / neuronx-cc / libneuronxla from the AWS repo.
export PIP_EXTRA_INDEX_URL="${NEURON_PIP_INDEX}"

# 3) Install the ENTIRE Neuron stack in one consolidated, compatible resolve.
#    --upgrade-strategy eager is what the HF docs prescribe; it lets pip pull the
#    matching Neuron torch instead of leaving a stale CUDA torch in place.
echo "Installing optimum-neuron[neuronx] (resolves torch-neuronx + neuronx-cc + transformers + accelerate) ..."
python -m pip install --upgrade --upgrade-strategy eager "optimum-neuron[neuronx]"

# 4) Freeze the Neuron-owned packages into a pip CONSTRAINTS file. Installing the
#    app requirements under this constraint lets pip resolve legitimate transitive
#    deps (numpy, regex, filelock, ...) while making it IMPOSSIBLE to change
#    torch / torch-neuronx / transformers / accelerate / torch_xla — i.e. it can
#    never silently pull a CUDA torch again. This is more robust than a hand-kept
#    backfill list (which drifts from requirements-neuron.txt).
echo "Freezing Neuron stack into constraints.txt ..."
CONSTRAINTS="$(mktemp)"
python -m pip freeze | grep -iE '^(torch|torch-xla|torch-neuronx|torchvision|neuronx-cc|libneuronxla|transformers|tokenizers|accelerate|optimum-neuron|safetensors|huggingface-hub|numpy)==' \
    > "${CONSTRAINTS}" || true
echo "--- constraints ---"; cat "${CONSTRAINTS}"; echo "-------------------"

# 5) Install the app requirements (with deps, but constrained) and the project.
echo "Installing project requirements under Neuron constraints ..."
python -m pip install -c "${CONSTRAINTS}" -r requirements-neuron.txt
python -m pip install --no-deps -e .
rm -f "${CONSTRAINTS}"

# 6) Sanity: assert we ended up on a NEURON torch (NOT a +cuXXX build) and that
#    NeuronAccelerator is importable. Fail loudly here rather than at train time.
python - <<'PY'
import sys
import torch
v = torch.__version__
print(f"torch: {v}")
assert "+cu" not in v, (
    f"FATAL: a CUDA torch ({v}) got installed. The Neuron stack was overwritten. "
    "Re-run after `pip uninstall -y torch torchvision`."
)
try:
    import torch_xla  # noqa: F401
    print("torch_xla: OK")
except Exception as e:
    print(f"WARNING: torch_xla import failed: {e!r}")
from optimum.neuron import NeuronAccelerator  # noqa: F401
print("optimum.neuron.NeuronAccelerator: OK")
import optimum.neuron as on
print(f"optimum-neuron: {getattr(on, '__version__', 'unknown')}")
print("=== Neuron stack sanity check PASSED ===")
PY

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
