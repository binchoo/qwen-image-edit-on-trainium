"""Pre-quantize the Qwen2.5-VL text_encoder to NF4 and save to disk.

Run once per machine. After this, training cache phase loads the NF4 text_encoder
directly — saves ~8 GB CPU RAM vs bf16 load, which is the margin that prevents
the 31.5 GB ceiling crash on 32 GB AI PC machines.

Trigger condition: `ram_gb <= 32`. See SKILL.md §7.2.

Prerequisites: qflux Python package importable (clone qwen-image-finetune and
either `pip install -e .` or run with the project's `src/` on PYTHONPATH).
See SKILL.md §6 Framework Adaptation.

Usage:
    python templates/quantize_text_encoder_nf4.py \\
        --src <path-to-qwen-image-edit-pipeline> \\
        --dst <path-to-save-nf4-text-encoder>

Example:
    python templates/quantize_text_encoder_nf4.py \\
        --src <path/to/Qwen-Image-Edit-pipeline> \\
        --dst <path/to/Qwen-Image-Edit-nf4-text-encoder>

On Windows, activate the oneAPI + conda env before running:
    setvars.bat
    conda activate qwen-image-edit-xpu

After completion, set in your training config:
    model:
      text_encoder_path: <DST>
The qflux model loader auto-detects quantization_config in <DST>/config.json
and skips online quantization, avoiding the text_encoder bf16 CPU-RAM peak.

Notes:
- The text_encoder is Qwen2_5_VLForConditionalGeneration loaded from the
  pipeline's `text_encoder/` subfolder (~15 GB bf16 on disk, ~7.5 GB NF4).
- NF4 quantization requires bitsandbytes XPU backend and the oneAPI runtime.
  Run setvars.bat before this script (Windows) or oneAPI on PATH (Linux).
- On LNL 32 GB: text_encoder quantization runs on XPU (bitsandbytes auto-selects);
  expect ~10 GB XPU peak during quantize + save.
"""
import argparse
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

try:
    from qflux.utils.memory_probe import MemoryProbe
except ImportError:
    MemoryProbe = None


def main():
    p = argparse.ArgumentParser(
        description="Pre-quantize Qwen2.5-VL text_encoder to NF4 (one-time)."
    )
    p.add_argument("--src", required=True,
                   help="Source pipeline dir (contains text_encoder/ subfolder)")
    p.add_argument("--dst", required=True,
                   help="Destination dir for the NF4-quantized text_encoder")
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
        print(f"[quantize_te] streaming-load & quantize: {args.src}/text_encoder")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.src,
            subfolder="text_encoder",
            quantization_config=bnb_config,
            torch_dtype=weight_dtype,
        )
        Path(args.dst).mkdir(parents=True, exist_ok=True)
        print(f"[quantize_te] saving NF4 text_encoder to {args.dst}")
        model.save_pretrained(args.dst, safe_serialization=True)
        print(f'[quantize_te] done. Set in config:  model.text_encoder_path: "{args.dst}"')

    if MemoryProbe is not None:
        with MemoryProbe(interval=args.probe_interval, label="quantize_te"):
            _quantize()
    else:
        print("[quantize_te] (qflux.utils.memory_probe not installed; running without probe)")
        _quantize()


if __name__ == "__main__":
    main()
