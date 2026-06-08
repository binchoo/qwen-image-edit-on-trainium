# templates/

Scripts and launchers the agent runs as part of the SKILL.md workflow.

| File | SKILL.md | Purpose |
|---|---|---|
| `probe_hw.py` | §5 | Probe XPU / RAM / oneAPI; emit JSON to stdout (consumed by `recommend_config.py`) |
| `recommend_config.py` | §5 | Read probe JSON → emit a training YAML + list of §7 patterns to apply |
| `dataset_validate.py` | §3 | Sanity-check dataset format + sample count before training |
| `quantize_transformer_nf4.py` | §7.1 | One-time NF4 pre-quantization of the DiT transformer |
| `quantize_text_encoder_nf4.py` | §7.2 | One-time NF4 pre-quantization of the text_encoder |
| `inference_compare.py` | §9.2 | Run inference on a test split (base + LoRA); saves PNGs + `manifest.json` for visual comparison |
| `monitor_xpu_memory.py` | §5.1 / §9.2 | Poll XPU memory at a fixed interval; run from a second terminal during training or inference |
| `launchers/quantize_xpu.bat` | §7.1 | Windows launcher for transformer NF4 quantization (handles oneAPI + conda activation) |
| `launchers/quantize_text_encoder_xpu.bat` | §7.2 | Windows launcher for text_encoder NF4 quantization |
| `launchers/train_xpu.bat` | §8 | Windows launcher: oneAPI `setvars.bat` + `conda activate` + `accelerate launch` / `--cache` mode |

All scripts use only stdlib + commonly-installed deps (torch, transformers,
psutil where applicable). Each script's docstring documents its CLI and
expected outputs in detail.
