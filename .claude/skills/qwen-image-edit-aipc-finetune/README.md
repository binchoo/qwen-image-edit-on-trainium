# Qwen-Image-Edit Fine-Tune on Intel AI PC — Quick Start

A Claude Code skill for fine-tuning **Qwen-Image-Edit** with **NF4 QLoRA**
on an Intel AI PC (Core Ultra processor) Windows laptop.

This skill is consumed by an **agent** (Claude Code or compatible) acting
on your behalf. The agent reads `SKILL.md` and walks you through every
step interactively — including the human-only pre-flight (driver,
oneAPI, conda, etc.).

## How to use this skill

1. **Clone the framework** (original CUDA-only version) to a local directory:

   ```bat
   git clone --no-checkout https://github.com/tsiendragon/qwen-image-finetune <local-path>
   cd <local-path>
   git checkout main -- . ":(exclude)docs/plan/v1.6.0:sampling_during_train.md" ":(exclude)tests/src/trainer/ /"
   ```

   > **Why not a plain `git clone`?** The upstream repository contains two files
   > with Windows-incompatible names (a colon and a space-only directory). A plain
   > `git clone` fails the checkout entirely and leaves the working tree empty.
   > The commands above use git's `:(exclude)` pathspec to skip those two paths.
   >
   > After cloning, `git status` will show those two paths as `deleted` — this is
   > expected and does not affect the training workflow. If the upstream repository
   > removes those files, the `:(exclude)` patterns simply match nothing and all
   > files are checked out normally.

   This is the unmodified upstream repo. The agent will apply all XPU
   adaptations (§6 of `SKILL.md`) during setup.

2. **Open your agent (Claude Code, etc.) inside that directory.** This skill
   is pre-installed in `.claude/skills/` of the repository — no extra setup
   needed. If you are using the skill from a separate source, place this
   directory under `.claude/skills/` in your project.

3. **Ask the agent for help**, for example:

   > "I have a Core Ultra AI PC and want to fine-tune Qwen-Image-Edit
   > with LoRA on my own dataset."

   The agent loads `SKILL.md` and walks you through everything else.
   First-time setup requires working through several installers and
   downloading large model files; subsequent runs reuse the result.

## What you'll do versus what the agent does

The agent will guide you through each item, but here's the rough split:

- **You (one-time, hands-on)**: Intel Arc driver install · Miniforge /
  conda install · Visual Studio Community + C++ workload · Intel oneAPI
  Base Toolkit · Windows long-path registry tweak · GPU shared-memory
  registry tweak · model + dataset on disk ·
  visual inspection of training results.
- **The agent**: hardware probe · training config recommendation · conda
  env creation · `torch+xpu` install · framework XPU adaptations · NF4
  pre-quantization · cache build · training launch · inference image
  generation (base + LoRA) for your review.

## Skill contents

- `SKILL.md` — the agent's full runbook (11 sections). Don't edit unless
  maintaining the skill.
- `templates/` — scripts the agent invokes:
  - `probe_hw.py` + `recommend_config.py` — hardware-tiered config
  - `quantize_transformer_nf4.py` + `quantize_text_encoder_nf4.py` — one-time NF4 pre-quantization (DiT and text_encoder)
  - `dataset_validate.py` — dataset sanity check
  - `inference_compare.py` — post-training visual comparison (base vs LoRA)
  - `monitor_xpu_memory.py` — XPU memory monitor for training / inference
  - `launchers/quantize_xpu.bat` — Windows launcher for transformer NF4 quantization
  - `launchers/quantize_text_encoder_xpu.bat` — Windows launcher for text_encoder NF4 quantization
  - `launchers/train_xpu.bat` — Windows launcher for training (cache + fit)


## When something goes wrong

`SKILL.md` §10 covers known XPU-on-Windows issues with fixes (Triton
cache staleness, DataLoader pickling, `accelerator.device` selecting
the wrong device, dataset-split errors, cache-phase OOM, etc.). The
agent will reach for §10 entries automatically as symptoms appear.
