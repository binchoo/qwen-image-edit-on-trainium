"""Pre-quantize the Qwen-Image-Edit transformer (DiT) to NF4 and save to disk.

Run once per machine. After this, training loads the NF4 model directly:
no bf16 CPU-RAM peak (~20 GB saved on a typical AI PC), 9.5 GB on disk.

Trigger condition for using this script: see SKILL.md §7.1.

Prerequisites: qflux Python package importable (clone qwen-image-finetune and
either `pip install -e .` or run with the project's `src/` on PYTHONPATH).
See SKILL.md §6 Framework Adaptation.

Usage:
    python templates/quantize_transformer_nf4.py \\
        --src <path-to-qwen-image-edit-pipeline> \\
        --dst <path-to-save-nf4-transformer>

Example:
    python templates/quantize_transformer_nf4.py \\
        --src <path/to/Qwen-Image-Edit-pipeline> \\
        --dst <path/to/Qwen-Image-Edit-nf4-transformer>

On Windows, prefer the launcher (which activates oneAPI + conda):
    templates/launchers/quantize_xpu.bat <SRC> <DST>

bnb's Triton-backed quantize_4bit needs the oneAPI runtime to JIT-compile
the quantize kernel — running without setvars.bat will fail.

After completion, set in your training config:
    model:
      transformer_path: <DST>
The qflux model loader auto-detects the saved `quantization_config` in
`config.json` and skips online quantization on subsequent runs.
"""
import argparse
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

from qflux.models.transformer_qwenimage import QwenImageTransformer2DModel

try:
    from qflux.utils.memory_probe import MemoryProbe
except ImportError:
    MemoryProbe = None


def main():
    p = argparse.ArgumentParser(
        description="Pre-quantize Qwen-Image-Edit transformer to NF4 (one-time)."
    )
    p.add_argument("--src", required=True,
                   help="Source pipeline dir (contains transformer/ subfolder)")
    p.add_argument("--dst", required=True,
                   help="Destination dir for the NF4-quantized transformer")
    p.add_argument("--compute-dtype", default="bfloat16",
                   choices=["bfloat16", "float16"],
                   help="bnb_4bit_compute_dtype (default: bfloat16)")
    p.add_argument("--probe-interval", type=float, default=0.5,
                   help="Memory sampling interval (seconds); ignored if MemoryProbe unavailable")
    args = p.parse_args()

    weight_dtype = getattr(torch, args.compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=weight_dtype,
        bnb_4bit_use_double_quant=True,
    )

    def _quantize():
        print(f"[quantize] streaming-load & quantize: {args.src}/transformer")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.src,
            subfolder="transformer",
            quantization_config=bnb_config,
            torch_dtype=weight_dtype,
        )
        Path(args.dst).mkdir(parents=True, exist_ok=True)
        print(f"[quantize] saving NF4 weights to {args.dst}")
        transformer.save_pretrained(args.dst, safe_serialization=True)
        print(f'[quantize] done. Set in config:  model.transformer_path: "{args.dst}"')

    if MemoryProbe is not None:
        with MemoryProbe(interval=args.probe_interval, label="quantize"):
            _quantize()
    else:
        print("[quantize] (qflux.utils.memory_probe not installed; running without probe)")
        _quantize()


if __name__ == "__main__":
    main()
