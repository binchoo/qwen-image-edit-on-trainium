"""Recommend a training config given hardware probe output.

Reads probe JSON (from probe_hw.py) and applies the rule table to emit a
complete training YAML config that fits the user's machine.

Rules applied (see SKILL.md §5.1 for the policy table):

    Memory (NF4 + 8-bit optimizer):
      ram_gb <= 64    → model.quantize_type = "nf4"          (NF4 transformer)
      ram_gb <= 32    → recommend NF4 text_encoder (§7.2)
      ram_gb <= 64    → optimizer = bitsandbytes.optim.Adam8bit
    Resolution (default [384,672]; user can override for speed/quality):
      [256, 448] — fast iteration: lower XPU demand, significantly faster per step
      [384, 672] — balanced default (emitted by this script): moderate XPU demand
      [512, 768] — full quality: higher XPU demand, tighter headroom on 32 GB AI PC,
                   significantly slower per step; requires gradient_checkpointing=true
      This script emits [384,672]; user can manually override in the output YAML.
    Always-on for AI PC tier:
      data.batch_size = 1
      data.num_workers = 1                  (Windows spawn-mode)
      train.mixed_precision = "no"          (XPU stability)
      train.gradient_accumulation_steps = 2
      train.gradient_checkpointing = true   (REQUIRED at [384,672]+; measured OOM without)
      cache.use_cache = true                (REQUIRED at 32GB; live encode crashes)
      lora.r = 16  (8-32 freely adjustable; memory-neutral at this LoRA scale)
      optimizer = Adam8bit  (memory-neutral at 24M params; 30% faster step vs torch Adam)

Usage:
    python templates/probe_hw.py | python templates/recommend_config.py
    python templates/recommend_config.py --probe probe.json --out config.yaml
    python templates/recommend_config.py --probe probe.json --dataset-path <path/to/dataset>

Output: a complete training YAML on stdout (or --out file). Selected §7
catalog patterns printed to stderr as a comment list.

The user must fill in remaining TODO placeholders before training:
- model.pretrained_model_name_or_path
- model.transformer_path       (optional: pre-quantized NF4 dir; omit to use online NF4 — memory-equivalent, slightly slower startup)
- model.text_encoder_path      (emitted only for ram_gb <= 32; optional: pre-quantized NF4 text_encoder from §7.2)
- data.init_args.dataset_path  (or pass --dataset-path)
- logging.output_dir
- cache.cache_dir

Network policy: this script does NOT touch the network. All inputs are local.
"""
import argparse
import json
import sys


def recommend(probe: dict, dataset_path: str | None = None) -> tuple[dict, list[str]]:
    """Return (config_dict, selected_patterns_list).

    This skill is NF4-QLoRA-only: NF4 transformer + bnb Adam8bit are
    always emitted. RAM tier (system memory; iGPUs share unified RAM)
    only gates the optional NF4 text_encoder layer (§7.2) and is
    informational for the agent.

    Resolution defaults to [384, 672] across all tiers — this is the only
    value characterized in practice. Higher / lower resolutions may fit
    different VRAM budgets but the trade-offs have not been measured.
    """
    ram_gb = probe.get("ram_gb", -1.0)

    # NF4 transformer + Adam8bit: always-on for this skill's NF4 QLoRA recipe.
    use_nf4_transformer = True
    use_adam8bit = True
    # NF4 text_encoder is the one tier-dependent optimization.
    use_nf4_text_encoder_recommended = (ram_gb > 0 and ram_gb <= 32)

    target_size = [384, 672]
    controls_size = [[384, 672], [512, 512]]

    use_grad_ckpt = True

    patterns = []
    if use_nf4_transformer:
        patterns.append("§7.1 Pre-quantize transformer (NF4)")
        patterns.append("§6.6 NF4 QLoRA patches (A1 + A2; apply before training)")
    if use_nf4_text_encoder_recommended:
        patterns.append("§7.2 Pre-quantize text_encoder (recommended for 32 GB machines — see §7.2)")
    patterns.append("§7.3 Mode-aware loading")
    patterns.append("§7.4 Cache-first workflow")
    patterns.append("§5.1 YAML config patterns (this script's output applies them)")

    cfg = {
        "trainer": "QwenImageEdit",
        "model": {
            "pretrained_model_name_or_path": "<TODO: original Qwen-Image-Edit pipeline dir>",
            "quantize": False,
            "lora": {
                "r": 16,
                "lora_alpha": 16,
                "init_lora_weights": "gaussian",
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "pretrained_weight": None,
            },
        },
        "data": {
            "class_path": "qflux.data.dataset.ImageDataset",
            "init_args": {
                "dataset_path": dataset_path if dataset_path else "<TODO: dataset path; see SKILL.md §3>",
                "caption_dropout_rate": 0.05,
                "prompt_image_dropout_rate": 0.05,
                "selected_control_indexes": [1],
                "cache_dir": "${cache.cache_dir}",
                "use_cache": "${cache.use_cache}",
                "processor": {
                    "class_path": "qflux.data.preprocess.ImageProcessor",
                    "init_args": {
                        "process_type": "center_crop",
                        "target_size": target_size,
                        "controls_size": controls_size,
                    },
                },
            },
            "batch_size": 1,
            "num_workers": 1,
            "shuffle": True,
        },
        "logging": {
            "output_dir": "<TODO: outputs/<run_name>>",
            "report_to": "tensorboard",
            "tracker_project_name": "<TODO: run name>",
        },
        "lr_scheduler": {
            "scheduler_type": "cosine",
            "warmup_steps": 50,
            "num_cycles": 0.5,
            "power": 1.0,
        },
        "train": {
            "gradient_accumulation_steps": 2,
            "max_train_steps": 1000,
            "num_epochs": 100,
            "checkpointing_steps": 100,
            "checkpoints_total_limit": 10,
            "max_grad_norm": 1.0,
            "mixed_precision": "no",
            "gradient_checkpointing": use_grad_ckpt,
            "low_memory": True,
        },
        "cache": {
            "devices": {"vae": "xpu:0", "text_encoder": "xpu:0"},
            "cache_dir": "<TODO: outputs/<run_name>/cache>",
            "use_cache": True,
        },
        "predict": {
            "devices": {"vae": "xpu:0", "text_encoder": "xpu:0", "dit": "xpu:0"},
        },
        "resume": None,
        "validation": {"enabled": False},
    }

    # NF4 + Adam8bit are always emitted (skill scope is NF4 QLoRA only).
    cfg["model"]["transformer_path"] = "<TODO: pre-quantized NF4 dir from §7.1>"
    if use_nf4_text_encoder_recommended:
        cfg["model"]["text_encoder_path"] = "<TODO: pre-quantized NF4 text_encoder dir from §7.2>"
    cfg["model"]["quantize_type"] = "nf4"
    cfg["optimizer"] = {
        "class_path": "bitsandbytes.optim.Adam8bit",
        "init_args": {"lr": 0.0001, "betas": [0.9, 0.999]},
    }

    return cfg, patterns


def _format_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        # Quote strings that look like config-vars, paths, or contain special chars
        if (v.startswith("${") or v.startswith("<") or "/" in v or " " in v
                or ":" in v or v == "no" or v == "yes" or v == "null"):
            return f'"{v}"'
        return v
    return str(v)


def _to_yaml(obj, indent: int = 0) -> str:
    """Minimal YAML serializer. Sufficient for the schema we emit."""
    sp = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return f"{sp}{{}}"
        lines = []
        for k, v in obj.items():
            if isinstance(v, dict) and v:
                lines.append(f"{sp}{k}:")
                lines.append(_to_yaml(v, indent + 1))
            elif isinstance(v, list) and v and any(isinstance(x, (dict,)) for x in v):
                lines.append(f"{sp}{k}:")
                lines.append(_to_yaml(v, indent + 1))
            elif isinstance(v, list):
                # Inline scalar list (or list of lists)
                if all(isinstance(x, list) for x in v):
                    inner = ", ".join(
                        "[" + ", ".join(_format_scalar(y) for y in x) + "]" for x in v
                    )
                    lines.append(f"{sp}{k}: [{inner}]")
                else:
                    inner = ", ".join(_format_scalar(x) for x in v)
                    lines.append(f"{sp}{k}: [{inner}]")
            else:
                lines.append(f"{sp}{k}: {_format_scalar(v)}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        lines = []
        for x in obj:
            if isinstance(x, dict):
                lines.append(f"{sp}-")
                lines.append(_to_yaml(x, indent + 1))
            else:
                lines.append(f"{sp}- {_format_scalar(x)}")
        return "\n".join(lines)
    else:
        return f"{sp}{_format_scalar(obj)}"


def to_yaml(obj) -> str:
    """Serialize to YAML. Prefer pyyaml when available; fall back to manual."""
    try:
        import yaml
        return yaml.safe_dump(obj, sort_keys=False, default_flow_style=None,
                              allow_unicode=True).rstrip() + "\n"
    except ImportError:
        return _to_yaml(obj) + "\n"


def main():
    p = argparse.ArgumentParser(
        description="Recommend a training config given hardware probe output."
    )
    p.add_argument("--probe",
                   help="Path to probe JSON file (default: read stdin)")
    p.add_argument("--out",
                   help="Path to write YAML (default: stdout)")
    p.add_argument("--dataset-path", default=None,
                   help="Dataset path; embedded into output (default: TODO placeholder)")
    args = p.parse_args()

    if args.probe:
        with open(args.probe) as f:
            probe = json.load(f)
    else:
        probe = json.load(sys.stdin)

    cfg, patterns = recommend(probe, dataset_path=args.dataset_path)
    yaml_text = to_yaml(cfg)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(yaml_text)
    else:
        sys.stdout.write(yaml_text)

    print("\n# Selected §7 catalog patterns to apply:", file=sys.stderr)
    for pat in patterns:
        print(f"#  - {pat}", file=sys.stderr)


if __name__ == "__main__":
    main()
