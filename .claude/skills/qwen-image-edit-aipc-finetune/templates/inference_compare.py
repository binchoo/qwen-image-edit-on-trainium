"""Run inference on a test split with and without LoRA for visual comparison.

Run twice with the same config — once for the base model, once with LoRA weights.
Inspect the outputs side-by-side using the §9.3 visual evaluation checklist.

Usage:
    # Step 1 — base model (no LoRA)
    python templates/inference_compare.py \\
        --config <path/to/config.yaml> \\
        --test-parquet <path/to/test.parquet> \\
        --output-dir outputs/compare/<run>/base \\
        --num-samples 5

    # Step 2 — LoRA-applied
    python templates/inference_compare.py \\
        --config <path/to/config.yaml> \\
        --test-parquet <path/to/test.parquet> \\
        --output-dir outputs/compare/<run>/lora \\
        --num-samples 5 \\
        --lora-weight outputs/<run>/checkpoint-<step>/pytorch_lora_weights.safetensors

Output per directory:
    sample_NN_<id>.png          — model output for each test sample
    sample_NN_<id>__target.png  — ground-truth target (if present in dataset)
    manifest.json               — metadata (prompts, paths, any errors)

Prerequisite: run from within the qwen-image-finetune project root, with the
qflux package importable. Set PYTHONPATH to include the project's `src/`
directory, or run from inside `src/` directly.

Network policy: this script does NOT touch the network. All models and data
must already be on disk. Run with HF_HUB_OFFLINE=1 etc. in your launcher.

(Adapted from scripts/inference_compare.py in the qwen-image-finetune project)
"""
import argparse
import io
import json
import sys
from pathlib import Path

import PIL.Image
import pyarrow.parquet as pq
import torch

from qflux.data.config import Config, TrMode, load_config_from_yaml

try:
    from qflux.utils.memory_probe import MemoryProbe
except ImportError:
    MemoryProbe = None


def _read_test_samples(parquet_path: Path, n: int) -> list:
    """Pull first n rows from a test parquet file."""
    table = pq.read_table(parquet_path)
    rows = table.slice(0, n).to_pylist()
    samples = []
    for i, row in enumerate(rows):
        ctrl_imgs = []
        for entry in row.get("control_images") or []:
            if isinstance(entry, dict) and "bytes" in entry and entry["bytes"]:
                ctrl_imgs.append(PIL.Image.open(io.BytesIO(entry["bytes"])).convert("RGB"))
            elif isinstance(entry, (bytes, bytearray)):
                ctrl_imgs.append(PIL.Image.open(io.BytesIO(entry)).convert("RGB"))
        target_entry = row.get("target_image")
        tgt = None
        if isinstance(target_entry, dict) and target_entry.get("bytes"):
            tgt = PIL.Image.open(io.BytesIO(target_entry["bytes"])).convert("RGB")
        samples.append({
            "id": row.get("id", str(i)),
            "control_images": ctrl_imgs,
            "target_image": tgt,
            "prompt": row.get("prompt", ""),
        })
    return samples


def _instantiate_trainer(config: Config):
    if config.trainer_type == "QwenImageEdit":
        from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
        return QwenImageEditTrainer(config)
    raise NotImplementedError(
        f"inference_compare.py supports QwenImageEdit only, got {config.trainer_type}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Run inference on a test split for visual LoRA comparison."
    )
    p.add_argument("--config", required=True,
                   help="Path to training config YAML")
    p.add_argument("--test-parquet", required=True,
                   help="Path to test split parquet file")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write output images and manifest.json")
    p.add_argument("--num-samples", type=int, default=5,
                   help="Number of test samples to run (default: 5)")
    p.add_argument("--num-inference-steps", type=int, default=20,
                   help="Diffusion inference steps (default: 20)")
    p.add_argument("--lora-weight", default=None,
                   help="Path to pytorch_lora_weights.safetensors; omit for base model")
    p.add_argument("--seed", type=int, default=42,
                   help="Fixed seed for reproducibility (default: 42)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config_from_yaml(args.config)
    # Inference mode: force text_encoder load (needed for new prompts) and
    # disable cache-first optimization (cache embeddings are training-time only)
    config.mode = TrMode.fit
    config.cache.use_cache = False
    config.data.init_args.use_cache = False
    config.validation.enabled = False
    # Keep text_encoder on CPU to avoid XPU memory pressure during inference
    config.predict.devices.text_encoder = "cpu"

    if args.lora_weight:
        config.model.lora.pretrained_weight = args.lora_weight
        print(f"[inference_compare] LoRA mode: {args.lora_weight}")
    else:
        config.model.lora.pretrained_weight = None
        print("[inference_compare] Base model mode (no LoRA)")

    samples = _read_test_samples(Path(args.test_parquet), args.num_samples)
    print(f"[inference_compare] Loaded {len(samples)} test samples")

    torch.manual_seed(args.seed)

    label = "lora" if args.lora_weight else "base"

    def _run():
        trainer = _instantiate_trainer(config)
        trainer.setup_predict()
        manifest = {
            "mode": label,
            "lora_weight": args.lora_weight,
            "config": args.config,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "samples": [],
        }
        for i, s in enumerate(samples):
            sid = s["id"]
            print(f"[inference_compare] {i + 1}/{len(samples)}: id={sid}")
            try:
                # Derive height/width from the first control image so the
                # trainer's prepare_embeddings path receives non-None values.
                # (PIL.Image.size returns (width, height))
                ctrl_w, ctrl_h = s["control_images"][0].size
                out_imgs = trainer.predict(
                    image=s["control_images"],
                    prompt=s["prompt"],
                    height=ctrl_h,
                    width=ctrl_w,
                    num_inference_steps=args.num_inference_steps,
                )
                out_img = out_imgs[0] if isinstance(out_imgs, list) else out_imgs
                fname = f"sample_{i:02d}_{sid}.png"
                out_img.save(out_dir / fname)
                if s["target_image"] is not None:
                    s["target_image"].save(out_dir / f"sample_{i:02d}_{sid}__target.png")
                manifest["samples"].append({
                    "id": sid,
                    "prompt": s["prompt"],
                    "output": fname,
                    "has_target": s["target_image"] is not None,
                })
            except Exception as e:
                import traceback
                print(f"[inference_compare] sample {i} FAILED: {e}")
                manifest["samples"].append({
                    "id": sid,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[inference_compare] Done. Output: {out_dir}")

    if MemoryProbe is not None:
        with MemoryProbe(label=f"inference-{label}"):
            _run()
    else:
        _run()


if __name__ == "__main__":
    main()
