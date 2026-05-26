---
name: qwen-image-edit-aipc-finetune
description: >
  End-to-end runbook for fine-tuning Qwen-Image-Edit with LoRA on a single Intel
  AI PC (Core Ultra processor) Windows laptop. Use this skill whenever a user
  wants to fine-tune or personalize Qwen-Image-Edit on their own dataset using
  Intel AI PC hardware, mentions XPU training or NF4 QLoRA on Windows — even if
  they don't use those exact terms. Covers: dataset preparation, hardware probe
  → recommended training config, oneAPI / conda environment setup, XPU framework
  adaptation, NF4 QLoRA memory optimizations, training execution, and
  post-training visual comparison.
---

# Qwen-Image-Edit LoRA Fine-Tuning on Intel AI PC (Windows)

## 1. Background & Scope

This skill is the end-to-end runbook for fine-tuning **Qwen-Image-Edit** with
LoRA on a single Intel AI PC (Core Ultra processor) Windows laptop. It is
written to be consumed by an agent acting on behalf of a non-expert user.

**Scope**: this skill targets **NF4 QLoRA only**. Other recipes (bf16 /
fp16 LoRA, full fine-tuning, etc.) are out of scope — 4-bit quantization
of the transformer is what makes the workload fit on AI PC unified memory.

**Recipe**: NF4-quantized transformer + LoRA +
`bitsandbytes.optim.Adam8bit` + cache-first workflow + mode-aware component
loading. Single-card XPU training under `accelerate launch` with
`distributed_type: NO` and `mixed_precision: 'no'`. The recommender (§5)
selects a sensible default config matched to AI PC hardware, with RAM-tier
adjustments where applicable.

**Workflow**: §2 human pre-flight → §3 dataset prep → §4 conda env →
§5 hardware probe + recommended config → §6 framework adaptation →
§7 conditional memory optimizations → §8 training → §9 validation.
§11 walks the full sequence with PASS signals at each step.

The skill assumes the user has already cloned `qwen-image-finetune`. The
agent walks them through §6 to apply XPU adaptations to the clone — these
are not auto-applied. After §6, the rest is largely automatable.

## 2. Pre-flight (human prerequisites)

The agent **cannot** complete the items below unattended. The user works
through them once before signaling the agent to proceed at §3. Order matters
where noted (e.g., Visual Studio is a prerequisite for oneAPI).

### 2.1 Hardware sanity

- Intel AI PC with **Core Ultra** processor and Intel iGPU.

### 2.2 Disk space

The estimates below cover **training-related components only** (model, data,
outputs, conda env). Prerequisite tools required by §2 (Intel Arc driver,
Intel oneAPI Base Toolkit, Visual Studio, etc.) need additional space on top
of this — consult their respective installers for current sizes.

Estimate at minimum **~65 GB free** on the target drive for training
components; **~80 GB** if both NF4 pre-quantizations (§7.1 + §7.2) are
performed. Sizes below are as displayed in Windows File Explorer (binary,
GiB — e.g. Windows shows a 9.82 GiB file as "9.82 GB"):

| Item | Approx. size |
|---|---|
| Qwen-Image-Edit model repository (the original, unmodified) | ~54 GB |
| NF4-quantized transformer output (§7.1, optional) | ~10 GB |
| NF4-quantized text encoder output (§7.2, optional) | ~5.5 GB |
| Dataset (depends on yours; character-composition reference is small) | ~1-5 GB |
| Embedding cache (proportional to dataset size) | ~0.5 MB/sample (e.g. ~18 MB for 35 samples) |
| Training checkpoints (LoRA adapters only; lightweight) | ~50 MB × N checkpoints |
| Conda env | ~6 GB |

### 2.3 Windows registry settings (admin Command Prompt)

Two registry tweaks make AI PC fine-tuning materially smoother. Open an
**admin Command Prompt** (right-click Start → "Terminal (Admin)" or search
"cmd" and run as administrator):

**Long-path support** — model file paths can exceed Windows' 260-char limit:

```bat
powershell -Command "Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1"
```

**Shared GPU memory ceiling** — AI PC iGPUs use unified memory; raising this
ceiling lets the OS commit more memory to the GPU process during the NF4 cache
phase. Open Registry Editor (Win+R → `regedit`), navigate to:

```
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager
```

Change `SystemPartitionCommitLimitPercentage` to a higher value (e.g. **75**
from the default ~57). Reboot for both changes to take effect.

### 2.4 Intel GPU driver

Install the latest Intel Arc Graphics driver:

https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html

Verify after install: Windows Device Manager → Display adapters → "Intel
Arc Graphics" appears without warnings.

### 2.5 Conda / Python

Install Miniforge from conda-forge:

https://conda-forge.org/download/

Confirm `conda --version` works in CMD after install (may require new shell
session).

### 2.6 Visual Studio Community (oneAPI prerequisite)

Required to provide the C++ build toolchain that oneAPI uses for Triton
JIT kernel compilation:

https://visualstudio.microsoft.com/vs/community/

During install, check the **"Desktop development with C++"** workload.
Other workloads are not required.

### 2.7 Intel oneAPI Base Toolkit

Required for bitsandbytes Triton-backed XPU kernels (NF4 quantize on first
load, `Adam8bit.step()` every step). Install **after** §2.6 (oneAPI's
installer detects the C++ toolchain).

Match the oneAPI version to the `torch+xpu` version you plan to install
in §4:

| `torch+xpu` | Triton package | oneAPI version |
|---|---|---|
| `2.9.0+xpu` | `pytorch-triton-xpu` 3.5 | 2025.2 |
| `2.10.0+xpu` | `triton-xpu` 3.6 | 2025.3 |
| `2.11.0+xpu` | `triton-xpu` 3.7 | 2025.3 |
| `2.12.0+xpu` | `triton-xpu` 3.7.1 | 2025.3 |

Download from:

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

Default install path: `C:\Program Files (x86)\Intel\oneAPI`. The training
launcher (`templates/launchers/train_xpu.bat`) invokes `setvars.bat` from
this path; if you install elsewhere, update the launcher.

> **If your `torch+xpu` version isn't listed above** (e.g. pip pulls a
> newer release than this skill anticipates): infer the matching oneAPI
> from pip's installed Intel transitive dependencies. Complete §2.4
> (conda) + §4.1-§4.2 first (create env + `pip install torch --index-url
> https://download.pytorch.org/whl/xpu`), then in the activated env run:
>
> ```bat
> pip list | findstr /I "intel- onemkl-sycl-"
> ```
>
> The major version of the `intel-*` and `onemkl-sycl-*` packages tells
> you which oneAPI release to install (e.g. seeing `2025.3.x` → install
> Intel oneAPI Base Toolkit **2025.3**). After oneAPI installs, return to
> §4.3 to install the remaining project dependencies.

**Already have oneAPI installed?** If oneAPI is already on your machine,
consider installing the `torch+xpu` version that matches your existing oneAPI
rather than upgrading oneAPI. Check which oneAPI version you have (see below),
then install the corresponding `torch+xpu` in §4.2 — this is often less work
than reinstalling oneAPI.

Verify after install (fresh CMD):

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
where icpx
icpx --version
```

`where icpx` must print a real path. To read the exact toolkit version (e.g.
`2025.3`), look at the `InstalledDir` line in the `icpx --version` output —
the version directory in the path is the toolkit version:

```
InstalledDir: C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\compiler
                                                           ^^^^^
                                             this is the toolkit version
```

(`icpx --version` also prints `Compiler 2025.3.3 ...` — note the three-part
compiler build version may differ slightly from the two-part toolkit version
`2025.3` in the path. Use the path number to match against the §2.7 table.)

Alternatively, open `C:\Program Files (x86)\Intel\oneAPI\Installer\installer.exe`
to see the installed toolkit version in the UI.

If `setvars.bat` reports `'vars.bat' is not recognized`, see §10.

### 2.8 Download Qwen-Image-Edit model

Qwen-Image-Edit is an open model — no HuggingFace account or token is required.
Download to a local directory that the agent will use as
`pretrained_model_name_or_path` (~54 GB):

```bat
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir <path\to\model>
```

(https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

Earlier versions of Qwen-Image-Edit (e.g., Qwen-Image-Edit-2509) should also
work with this skill in principle; replace the repo ID and local-dir path
accordingly.

### 2.9 Pre-flight checklist

Confirm each item before signaling the agent to proceed:

- [ ] AI PC with Core Ultra processor confirmed
- [ ] ≥ 80 GB free disk space on target drive
- [ ] Long-path registry enabled; GPU shared memory ceiling adjusted; reboot done
- [ ] Intel Arc Graphics driver installed (Device Manager shows iGPU clean)
- [ ] conda installed (`conda --version` works)
- [ ] Visual Studio Community installed with "Desktop development with C++" workload
- [ ] Intel oneAPI Base Toolkit installed (`where icpx` works in a fresh CMD after `setvars.bat`)
- [ ] Qwen-Image-Edit model downloaded to a local directory
- [ ] `qwen-image-finetune` project cloned to a local directory
- [ ] Training dataset prepared per §3 (or ready to be prepared at §3)

## 3. Dataset Preparation

The qwen-image-finetune framework (`qflux.data.dataset.ImageDataset`) supports
three dataset formats. Pick whichever fits the user's workflow; all three are
validated by `templates/dataset_validate.py`.

### 3.1 Format options

| Format | When to use | Schema |
|---|---|---|
| **Local directory** | Quick iteration on a custom dataset | Stem-paired files: `<stem>.png` (target) + `<stem>.png` (control) + `<stem>.txt` (prompt), under `training_images/` and `control_images/` subdirs |
| **HuggingFace dataset** | Sharing / public datasets | Repo or local parquet dir; columns: `target_image`, `control_images` (list), `prompt`, optional `control_mask` |
| **CSV** | Migrating from another pipeline | Columns: `path_target`, `path_control` (one or more `path_control_N`), `prompt`, optional `path_mask` |

Detailed schemas live in `src/qflux/data/dataset.py` (`ImageDataset.__init__`
docstring). The §5 recommender emits a `data:` block matching whichever format
the user has.

### 3.2 Local directory layout

```
dataset_root/
├── training_images/
│   ├── sampleA.png              # target image — the desired OUTPUT after the edit
│   └── sampleA.txt              # required: prompt text (the edit instruction)
└── control_images/
    ├── sampleA.png              # main control — INPUT condition (e.g. background scene)
    ├── sampleA_control_1.png    # optional: 2nd control (e.g. character reference image)
    └── sampleA_mask.png         # optional: mask of the edited region (character silhouette)
```

**Naming rules** (from `qflux.data.dataset.ImageDataset` documentation):
- Extra control images: append `_control_1`, `_control_2`, … to the base name
- Mask: `<base>_mask.png` in either `control_images/` or `training_images/`
- Prompt: `.txt` file with the same base name; if present in both dirs, `training_images/` wins

**Alternative directory names** accepted: `images` / `target_images` / `target` for targets; `control` / `condition_images` / `controls` for controls.

**Multi-control example** (character composition task): `sampleA.png` = background scene, `sampleA_control_1.png` = character on white background, `sampleA_mask.png` = character silhouette in the final composition, prompt = "Add the character to the image". This is the pattern used by the `TsienDragon/character-composition` reference dataset.

**Mask usage**: when a mask is present, `edit_mask_loss` applies higher training weight to the masked region, focusing the LoRA on the area that changed between control and target. Useful when only part of the image is edited.

### 3.3 Sample size guidance

| Sample count | Expected outcome |
|---|---|
| < 5 | Hard floor — `dataset_validate.py` rejects; too few to cache meaningfully |
| 5–19 | Smoke test only; rapid overfitting expected within the first tens of steps |
| 20–50 | **Practical starting point for a narrow task** (single subject, single edit type, single viewpoint set). Community experience confirms this range works for specific character/product LoRAs. ² |
| 50–100 | Better generalization; recommended if the edit should work across varying scenes or lighting. A 50-image rendered dataset has been validated for complex spatial transformations. ³ |
| 100–200+ | For more general edit styles that should work across many different scenes and subjects. ² |

**Quality beats quantity.** Adding low-quality or inconsistent pairs actively harms the LoRA — they introduce noise the model cannot learn a clean pattern from. ¹ ²

**Single-task LoRA preferred.** If you want to train multiple distinct edits (e.g., style + object replacement), train separate LoRAs rather than one mixed dataset. Multi-task LoRA often causes the tasks to interfere with each other. ²

`templates/dataset_validate.py` enforces the floor of 5 samples and warns below 30. The framework caches embeddings per-sample (§7.4), so larger datasets cost more upfront cache time but no extra per-step cost during fit.

---
¹ FlyMyAI LoRA Trainer, https://github.com/FlyMyAI/flymyai-lora-trainer, Aug 2025. Uses a different training framework; cited for model-level image quality guidance applicable to Qwen-Image-Edit.  
² HuggingFace Forums: "Question about lora fine tune qwen-image-edit" (John6666, Nov 2025), https://discuss.huggingface.co/t/question-about-lora-fine-tune-qwen-image-edit/170633. Qwen-Image-Edit-specific community guidance; cited for dataset size ranges, quality advice, and single-task LoRA recommendation.  
³ とりにく, "vast.AIでQwen image Edit 2509のLoRA学習", https://note.com/tori29umai/n/n256f30d51669, Sept 2025. Uses Musubi Tuner (a separate training framework); dataset construction and spatial transformation advice is model-level and framework-independent.

### 3.4 Control image and target image — the edit pair

Qwen-Image-Edit is an **image editing** model, not an image generation model. Every training sample teaches the model one specific edit:

| Part | Role | Example |
|---|---|---|
| **Control image(s)** | The input condition(s) — what the user provides | Background scene, character reference, source style |
| **Target image** | The desired output — what the model should produce | The same scene after the edit is applied |
| **Prompt** | The edit instruction | "Add the character to the image" |

The key difference from caption-based LoRA training (e.g. for image generation): the prompt describes **what to do**, not **what the result looks like**.

**Multi-control paradigm** — the framework natively supports multiple control images per sample, which is useful when the edit requires more than one reference:
- Main control (`sampleA.png`): the primary input scene or reference
- Additional controls (`sampleA_control_1.png`, `_control_2.png`, …): supplementary references (e.g. a character sheet, a style reference)
- Mask (`sampleA_mask.png`): region of interest — tells the model which part of the image the edit targets

**Verified example** (character-composition task): the `TsienDragon/character-composition` dataset is one verified instance of this setup — it uses two controls (a background image and a character-on-white-background image) plus a mask of the character's silhouette, with a fixed prompt "Add the character to the image". The target is the character correctly composited into the background. User-created datasets following the same format work equally well.

**General task examples** (for reference, not exhaustive):
- *Viewpoint change*: control = source angle; target = desired angle; prompt = describe the viewpoint transformation ³
- *Object addition*: control = scene; target = scene with object added; prompt = "Add [object] to the image"
- *Style change*: control = original; target = restyled; prompt = describe the style change

**Important**: control and target must share the same subject/context. Unrelated images in a pair prevent the model from learning a coherent edit mapping.

**Practical tip**: synthetic or rendered control images (e.g. 3D, game engine) produce very consistent results because they eliminate photographic variation that the model might otherwise try to replicate. ³

---
³ とりにく, "vast.AIでQwen image Edit 2509のLoRA学習", https://note.com/tori29umai/n/n256f30d51669, Sept 2025. Uses Musubi Tuner (a separate training framework); dataset construction advice is model-level and framework-independent.

### 3.5 Image quality guidelines

- **Resolution**: source images should be **at least as large as your `target_size`** in each dimension. The training preprocessor crops/resizes source images down to `target_size` — if the source is smaller, it gets upscaled first (quality loss). Practical guide by training tier:
  - `target_size [384, 672]` (default): source images ≥ 672px in the longer dimension. 512px+ short-side is fine.
  - `target_size [512, 768]`: source images ≥ 768px in the longer dimension. 512×512 would require upscaling and is not ideal.
  - Higher source resolution than your `target_size` is always fine; the crop just has more to choose from.
  *(Recommended range from `docs/guide/data-preparation.md`: 512×512 to 1024×1024 — this is appropriate for the default `target_size [384, 672]` and most configurations used with this skill.)*
- **Format**: JPG, JPEG, PNG, WebP — all supported by the framework. RGB (3-channel) required.
- **Consistency within dataset**: maintain consistent image quality, lighting style, and subject framing across pairs. Inconsistency is a common cause of blurry or incoherent model outputs. ¹
- **Quality over quantity**: more low-quality pairs actively harm the LoRA — they introduce noise the model cannot learn a clean pattern from. ¹
- **Avoid**: watermarks or text overlays in control/target images; heavy JPEG artefacts; mixing very different visual styles (e.g., anime and photorealistic) in the same dataset without a compelling reason.

---
¹ FlyMyAI LoRA Trainer, https://github.com/FlyMyAI/flymyai-lora-trainer, Aug 2025. Uses a different training framework; cited for model-level image quality guidance applicable to Qwen-Image-Edit.

### 3.6 Prompt strategy

The framework expects prompts to be **descriptive editing instructions** — text that tells the model what to do, not what the result looks like.  
*(From `docs/guide/data-preparation.md`: "Descriptive editing instructions"; recommended length 10–200 words)*

| Prompt type | Example | Use |
|---|---|---|
| **Edit instruction** (correct) | "Add the character to the image" | For image editing LoRA training |
| **Target image caption** (incorrect for editing) | "A character standing in a room" | Trains the model to generate, not edit |

**Key guidelines:**

- **Describe the transformation**: "Add the character to the image" teaches *what to do* (placement); "A character in a room" only describes *what exists* in the result — the model cannot learn the editing operation from it.
- **Consistency within a dataset**: if all samples share the same edit type (e.g., character composition), a fixed or near-fixed prompt like "Add the character to the image" is valid and has been shown to work well. The model learns the edit from the image pairs; the prompt anchors what operation is being requested.
- **Specificity when edits vary**: if your dataset covers multiple edit types or subjects, prompts should distinguish them — e.g. "Add the character to the outdoor scene" vs "Add the character to the interior scene" — so the model learns to condition on the instruction, not just pattern-match visually.
- **Length**: 10–200 words accepted by the framework. Short, clear instructions work well for specific tasks; longer prompts are appropriate when the edit is complex or context-dependent.
- **Avoid ambiguity**: "Edit the image" teaches nothing — a prompt must specify what kind of edit.

### 3.7 Validation step

Before committing to a training run, validate the dataset:

```bash
python templates/dataset_validate.py <path>
# Local dir:    python templates/dataset_validate.py <path/to/dataset>
# HF parquet:   python templates/dataset_validate.py <path/to/parquet-dir>
# CSV:          python templates/dataset_validate.py <path/to/dataset.csv>
```

JSON verdict format:
```json
{
  "valid": true,
  "format": "local",
  "sample_count": 35,
  "issues": [],
  "warnings": []
}
```

Exit code 0 = valid (proceed to §4); 1 = invalid (fix before proceeding).
Issues are blockers; warnings are advisory (e.g., small sample count fine for
smoke test, problematic for real training).

Common issues caught: missing `training_images/` or `control_images/` subdir;
target images without matching control or prompt; sample count too low.

**Manual check after the script passes** — `dataset_validate.py` only checks
format and structure; it cannot verify semantic correctness. Before starting
a full training run, inspect 3–5 random pairs by eye:

- Does the target image look like a plausible result of applying the prompt to the control image(s)?
- Is it clear what changed between control and target, and does the prompt describe that change?
- Is the subject/scene the same in control and target (they are before/after, not unrelated images)?

If any pair fails this check, the data has a consistency problem that will reduce LoRA quality.
Finding and fixing a few bad pairs early is much cheaper than diagnosing a poorly-trained model later.

### 3.8 Hold out a test set

Before finalising your dataset, **set aside at least 3 samples that will not
be used for training**. These become your held-out test set for §9.2
post-training visual comparison (base model vs. trained LoRA).

Choosing which samples to hold out:
- Pick samples that are **representative** of the edit you are teaching —
  the comparison is only meaningful if the test images resemble the training
  distribution.
- Do **not** reuse training samples as test samples. The model has seen those
  images during training, so any apparent quality difference is unreliable.

**For HuggingFace / parquet datasets**: the test split parquet
(`<dataset_dir>/data/test-*.parquet`) is usually provided automatically.
Use it directly in §9.2; no extra step needed here.

**For local directory datasets**: physically move the held-out pairs to a
separate folder (e.g. `test/`) before running `dataset_validate.py`. Point
`--test-parquet` in §9.2 at this folder (or adapt `inference_compare.py`
to load local images directly).

If you are working from a small dataset where holding out samples would
leave too few for training (see §3.3 thresholds), collect a few additional
pairs specifically for testing rather than reducing the training set.

## 4. Environment Setup

§2 covered the system-level prerequisites (oneAPI, driver, conda). §4
sets up the Python training environment. **One-time per machine**; reuse
across runs.

### 4.1 Create the conda env

```bat
conda create -n qwen-image-edit-xpu python=3.12 -y
conda activate qwen-image-edit-xpu
```

(If a torch+xpu base env already exists locally, prefer cloning it —
`conda create -n qwen-image-edit-xpu --clone <existing-xpu-env>` — to avoid
re-downloading the torch wheel.)

### 4.2 Install torch+xpu

**Recommended: latest wheel (let pip pick the version)**

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

Minimum supported: `torch >= 2.9.0+xpu`. Confirm the installed version
matches your oneAPI version (§2.7 version table). For example,
`torch 2.11.0+xpu` requires oneAPI 2025.3.

**Pin a specific version** (if your oneAPI is already installed and you want
to match it, rather than upgrade oneAPI):

```bat
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/xpu
```

`torchvision` follows `torch` automatically from the same XPU channel — no
need to pin it separately.

### 4.3 Install project dependencies

Before running `pip install`, the upstream `requirements.txt` needs two
edits (a fresh clone of `tsiendragon/qwen-image-finetune` ships with both
problems):

**1. Comment out `transformer_engine[pytorch]`** — CUDA-only, no XPU equivalent.
Installing it will fail on a machine without NVCC:

```diff
# requirements.txt
- transformer_engine[pytorch]
+ # XPU: transformer_engine is CUDA-only — comment out
+ # transformer_engine[pytorch]
```

**2. Pin `bitsandbytes>=0.48.2`** — the unversioned `bitsandbytes` line may
resolve to an older build that lacks the XPU backend:

```diff
- bitsandbytes
+ bitsandbytes>=0.48.2
```

Then install:

```bat
pip install -r requirements.txt
```

If §6 adaptation has already been applied (the repo has `device_utils.py`
and the XPU patches in place), this is the only required pre-install edit.

### 4.4 Activate oneAPI in the current shell

The training launcher (`templates/launchers/train_xpu.bat`) handles this
automatically; for ad-hoc verification:

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
```

If this reports `'vars.bat' is not recognized` errors, see §10.1: clear
`NoDefaultCurrentDirectoryInExePath` across the call. The launcher template
does this; for a manual session, run from a CMD opened directly (not from
inside another wrapper shell).

### 4.5 Verify

```bat
python templates\probe_hw.py
```

PASS criteria — JSON output must show:
- `xpu_available: true`
- `torch_version` ending in `+xpu`
- `bnb_version` >= `0.48.2`
- `bnb_xpu_available: true`
- `oneapi_version` non-null (only if §4.4 setvars activated; expected null
  for fresh sessions before launcher runs)

If any of the first four fail, do NOT proceed to §6 — fix the environment first.

## 5. Hardware Probe → Config Recommendation

Two scripts produce a training YAML matched to the user's machine:

```bash
python templates/probe_hw.py > probe.json
python templates/recommend_config.py --probe probe.json --out config.yaml
```

`probe_hw.py` emits hardware/software JSON (schema in its own header).
`recommend_config.py` reads that JSON, applies the rules table below,
and emits a complete training YAML.

The YAML output contains TODO placeholders for machine-specific paths the
user (or agent) must fill in:
- `model.pretrained_model_name_or_path` (original Qwen-Image-Edit model directory)
- `model.transformer_path` (pre-quantized NF4 dir from §7.1 — optional; omit to use online NF4 quantization)
- `data.init_args.dataset_path` (validated dataset; see §3)
- `logging.output_dir` and `cache.cache_dir`

Pass `--dataset-path <path>` to the recommender to fill the dataset_path
placeholder inline.

The recommender also prints a list of selected §7 catalog patterns to stderr
— these are the conditional optimizations from §7 that apply to this machine.

### 5.1 Recommended settings

> **Minimum required: 32 GB AI PC.** 16 GB is not supported: the NF4 DiT
> alone requires ~10 GB XPU allocation, leaving insufficient headroom for
> training activations on a 16 GB unified-memory machine. The minimum
> practical tier is **32 GB**.

The recommender emits the following settings. Each row includes the reason
and test evidence where available.

| Setting | Value | Why |
|---|---|---|
| `model.quantize_type` | `nf4` | Scope of this skill; NF4 QLoRA only |
| `model.transformer_path` | `<pre-quant dir or null>` | **Optional.** Online NF4 uses shard-by-shard streaming so memory impact is negligible. Value of pre-quantizing is startup time (saves one quantization pass per run). Recommended for multiple training iterations. See §7.1. |
| Pre-quantize text_encoder (`model.text_encoder_path`) | optional; see §7.2 | Reduces text_encoder from ~15 GB to ~5.5 GB (NF4). Reduces cache-phase RAM by ~8 GB on 32 GB machines — eliminates heavy paging. See §7.2 for measured effect and prep steps. |
| `cache.use_cache` (§7.4) | `true` | **Required.** Without caching, the trainer keeps text_encoder active during fit to encode prompts per batch while the NF4 DiT is also loaded — combined memory pressure crashes on 32 GB machines. Cache-first runs text_encoder and VAE at cache time, writes pre-computed embeddings and image latents to disk, then frees text_encoder before the NF4 DiT loads for fit. |
| Mode-aware loading (§7.3) | always | Enforces the above separation at the code level |
| `data.processor.target_size` | `[384, 672]` default | Balanced default; three-tier options: |
| ↳ `[256, 448]` | Lower XPU demand | Significantly faster per step; fast iteration and prototyping |
| ↳ `[384, 672]` | **Default** | Balanced default |
| ↳ `[512, 768]` | Higher (tight on 32 GB) | Significantly slower; system RAM reaches the 32 GB ceiling on 32 GB machines |
| `train.gradient_checkpointing` | `true` | **Recommended at all resolutions; required at ≥[384,672].** Without it, activation memory causes XPU out-of-memory during attention at the default and larger resolutions. Reducing LoRA rank does not resolve this. |
| `train.max_train_steps` | `1000` | Practical starting point for a narrow task on a ~35-sample dataset. Reduce to **50–200** for a quick pipeline sanity check; increase for larger datasets or more complex edits. Monitor smooth loss — stop when it plateaus across consecutive checkpoints. |
| `train.gradient_accumulation_steps` | `2` | Effective batch-size > 1 without extra per-step memory |
| `optimizer.class_path` | `bitsandbytes.optim.Adam8bit` | Memory difference vs `torch.optim.Adam` is negligible at this LoRA scale. Real value: measurably faster per step via Triton-backed optimizer. |
| `lora.dtype` | `"bf16"` | Optional; stores LoRA adapter weights in bf16 vs fp32 default (~48 MB savings at r=16 — negligible). §6.6 NF4-A2 autocast handles forward compute dtype; this is a complementary parameter-storage setting. |
| `lora.r` | `16` | Memory-neutral — activation memory dominates, not LoRA parameters. Quality choice; adjust freely between 8–32. |
| `data.batch_size` | `1` | Single GPU, minimum |
| `data.num_workers` | `1` | Windows spawn-mode; qflux pydantic schema rejects 0 |
| `train.mixed_precision` | `"no"` | accelerate's bf16 path wraps `torch.autocast("cuda")` — does not dispatch to XPU. qflux wraps the NF4 forward pass in `torch.autocast("xpu", ...)` internally (see §6). |

### 5.2 Manual override

If the probe is incorrect (e.g. `oneapi_version = null` because `setvars.bat`
hasn't run yet — see §4) or different settings are desired:
1. Edit the generated YAML directly. §5.1 documents what each
   key does and when to set it.
2. Or hand-edit `probe.json` before piping into the recommender.

The recommender is **conservative** — it picks the safest config that fits.
Loosen settings (larger batch, larger LoRA rank, lower grad-accum) only after
the conservative config trains successfully and you verify XPU memory has
headroom (via `templates/monitor_xpu_memory.py`).

> **HuggingFace / parquet dataset path**: `--dataset-path` expects a **local
> directory** with `training_images/` and `control_images/` subdirectories
> (see §3.2). If your dataset is in HuggingFace parquet format (downloaded
> via HF Hub), the recommender will emit a plain string path that the qflux
> loader cannot parse. Edit the generated YAML's `data.init_args.dataset_path`
> to use the dict format before training:
>
> ```yaml
> data:
>   init_args:
>     dataset_path:
>       - repo_id: <path/to/local-parquet-dir>
>         split: train
> ```


## 6. Framework Adaptation

> **Scope**: §6 describes adapting the **original `tsiendragon/qwen-image-finetune`
> repository** (CUDA-only, unmodified). If you cloned an already-adapted fork,
> some or all of these changes may already be present — run the §6.6 import
> test to confirm before re-applying anything.

The `qflux` codebase (qwen-image-finetune's package) hardcodes `torch.cuda.*`
calls — it's a custom training loop, not built on HF Trainer's auto-XPU
detection. The adaptations below are required before training will run on XPU.

### 6.1 Device-agnostic helpers (`device_utils.py`)

Replace each `torch.cuda.*` call in the framework with a device-agnostic
wrapper that checks XPU first. This confines the XPU adaptation to the
framework's own source files and leaves third-party libraries (Triton,
bitsandbytes, accelerate) untouched.

In qwen-image-finetune the wrappers live at `src/qflux/utils/device_utils.py`.
Verify the following helpers are present; if adapting a fresh clone, create
this file before any other change.

```python
# src/qflux/utils/device_utils.py
# enable XPU: device-agnostic helpers — call these instead of torch.cuda.* directly.
import contextlib
import torch

def device_empty_cache() -> None:
    if torch.xpu.is_available():           # enable XPU
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def device_synchronize() -> None:
    if torch.xpu.is_available():           # enable XPU
        torch.xpu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

def device_manual_seed_all(seed: int) -> None:
    if torch.xpu.is_available():           # enable XPU
        torch.xpu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)

def device_context(device):
    """Device context manager replacing torch.cuda.device().

    torch.xpu.device("cpu") raises; this helper returns nullcontext() for
    CPU and the correct accelerator context otherwise.
    """
    device_str = str(device) if device is not None else "cpu"
    if device_str == "cpu":
        return contextlib.nullcontext()
    if torch.xpu.is_available():           # enable XPU
        return torch.xpu.device(device)
    return torch.cuda.device(device)
```

Replace each direct `torch.cuda.*` call in the framework:

| Original | Replacement | Import |
|---|---|---|
| `torch.cuda.empty_cache()` | `device_empty_cache()` | `from qflux.utils.device_utils import device_empty_cache` |
| `torch.cuda.synchronize()` | `device_synchronize()` | `from qflux.utils.device_utils import device_synchronize` |
| `torch.cuda.manual_seed_all(s)` | `device_manual_seed_all(s)` | `from qflux.utils.device_utils import device_manual_seed_all` |
| `with torch.cuda.device(dev):` | `with device_context(dev):` | `from qflux.utils.device_utils import device_context` |

### 6.2 Find and fix residual patterns

The `device_utils.py` helpers cover functional calls. A separate scan is
needed for string-based device comparisons and CUDA-only library imports,
which helpers cannot address:

```bash
grep -rn "torch\.cuda\.\|flash_attention_2\|transformer_engine\|device\.type.*cuda" src/qflux/
```

| Pattern | Fix |
|---|---|
| `flash_attention_2` (in `from_pretrained` calls or `config.json`) | Replace with `"sdpa"`. Flash Attention 2 is CUDA-only |
| `transformer_engine` import / use | Comment out — CUDA-only library |
| `device.type == "cuda"` literal compare | Add an `xpu` branch (string compare can't be patched) |
| `mp.set_start_method("spawn", force=True)` | Guard with `if not torch.xpu.is_available(): ...`. Windows defaults to `spawn` already; `force=True` raises `RuntimeError: context has already been set` |
| `torch.cuda.device(device)` context manager | Replace with `device_context(device)` from `device_utils.py`. `torch.xpu.device("cpu")` raises when `device="cpu"` (e.g. text_encoder placed on CPU for inference); `device_context()` returns `nullcontext()` for CPU, the correct accelerator context otherwise |
| `train_epoch()` ignores `max_train_steps` — runs `num_epochs × dataset_size` steps instead | Add a guard at the top of the per-batch loop in `train_epoch()`: `if self.global_step >= self.config.train.max_train_steps: return` |

Example `device.type` fix in a config validator:

```python
if d.type == "cuda":
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA not available but got device={d}.")
# enable XPU: validate XPU device availability
if d.type == "xpu" and not torch.xpu.is_available():
    raise ValueError(f"XPU not available but got device={d}.")
```

### 6.3 accelerate single-card XPU config

`qflux.main` launches via `accelerate launch --config_file <path>`. Ship an
XPU-specific config alongside the CUDA one (e.g., `accelerate_config_xpu.yaml`):

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
mixed_precision: 'no'
num_machines: 1
num_processes: 1
main_training_function: main
dynamo_backend: 'no'
use_cpu: false
deepspeed_config: {}
```

Key differences from a typical CUDA multi-GPU config:

| Field | CUDA multi-GPU | XPU single-card | Why |
|---|---|---|---|
| `distributed_type` | `MULTI_GPU` | `NO` | Single process; no collective ops |
| `mixed_precision` | `bf16` | `'no'` | accelerate's bf16 wraps `torch.autocast("cuda", ...)` — does not dispatch to XPU. Apply `torch.autocast("xpu", ...)` in the trainer instead (§6.6 NF4-A2) |
| `dynamo_backend` | `inductor` | `'no'` | XPU `torch.compile` is experimental; accelerate's wrapper often fails through it |

> **YAML gotcha**: `dynamo_backend: no` unquoted parses as YAML boolean
> `False`; accelerate then calls `.upper()` on it and raises
> `AttributeError: 'bool' object has no attribute 'upper'`. Always quote
> `'no'` (and same for `mixed_precision: 'no'`).

### 6.4 Pydantic schema relaxation for `num_workers`

Many CUDA-origin frameworks (qflux included) declare `num_workers` with a
`> 0` constraint. On Windows under `spawn` start-method, classes registered
via `trust_remote_code` are not picklable across processes — workers crash
on dataset iteration. Setting `num_workers = 0` would fix it but the schema
rejects 0.

Relax the constraint to `>= 0` in the dataset config validator. The
recommender (§5) defaults to `1` (since 0 disables multiprocessing entirely
and slows data loading), but `0` should at least be acceptable as an escape
hatch for the picklability case.

### 6.5 BitsAndBytes XPU backend (informational)

Knowing which bnb operations dispatch to which backend on XPU helps the
agent diagnose failures and explains why §2 requires Visual Studio + oneAPI.
For `bitsandbytes >= 0.48.2`:

| Operation | Backend | When it runs |
|---|---|---|
| `dequantize_4bit` | **SYCL** | Every NF4 forward and backward pass |
| `dequantize_blockwise` | **SYCL** | Used alongside NF4 dequant |
| `gemv_4bit` | **SYCL** | Inference-only single-token path |
| `quantize_4bit`, `quantize_blockwise` | **Triton** | Initial quantization (e.g. §7.1 pre-quantize step) |
| `optimizer_update_8bit_blockwise`, `optimizer_update_32bit` | **Triton** | Every `bnb.optim.Adam8bit.step()` call |

**Key consequence**: NF4 forward/backward does not use a dedicated NF4
gemm kernel — bnb dequantizes the 4-bit weight via SYCL and hands off to
PyTorch's native XPU matmul. Only the **optimizer step** and
**initial quantization** actually go through Triton.

**Practical impact**: if a training run loads a pre-quantized NF4
checkpoint (§7.1 output) and uses `Adam8bit`, Triton kernel JIT compile
runs on the first optimizer step — this is why §2.7 oneAPI + §2.6 Visual
Studio C++ workload are required.

The dispatch above is empirical for `bitsandbytes 0.49.2`. Newer versions
may shift; if odd failures appear, check `bitsandbytes/backends/xpu/ops.py`
in your installed bnb version for the current registration.

### 6.6 NF4 QLoRA required patches

When adapting the framework for NF4 QLoRA, two additional patches are
required beyond the general XPU adaptation in §6.1-§6.4. In
qwen-image-finetune these are already applied; verify they are present
after applying §6.

#### NF4-A1. `torch_dtype` on `from_pretrained`

**Symptom if missing**: `RuntimeError` from attention op with a dtype
mismatch between fp16/bf16/fp32 tensors on the first forward pass.

**Why**: without `torch_dtype`, non-quantized params (norms, biases) stay at
the checkpoint's stored dtype. PEFT LoRA adapters default to fp32. The
cross-dtype attention kernel rejects mixed inputs.

**Fix**: pass `torch_dtype` to the NF4 `from_pretrained` call:
```python
transformer = QwenImageTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="transformer",
    quantization_config=bnb_config,
    torch_dtype=weight_dtype,   # aligns non-quantized params to training dtype
    attn_implementation="sdpa",
)
```

In qwen-image-finetune: `src/qflux/models/load_model.py` — search for
`torch_dtype=weight_dtype` in the `load_transformer` function.

#### NF4-A2. `torch.autocast` wrap for LoRA forward pass

**Symptom if missing**: dtype mismatch persists after NF4-A1. PEFT LoRA
adapters remain fp32 by default; in DiT joint attention their fp32 outputs
collide with bf16 outputs from other branches.

**Why**: `bf16_input @ fp32_lora_weight` promotes to fp32. The mismatch
only surfaces after NF4-A1 resolves the norm dtype issue.

**Fix**: wrap the trainable model's forward in `torch.autocast`:
```python
_autocast_device = "xpu" if torch.xpu.is_available() else "cuda"
with torch.autocast(_autocast_device, dtype=weight_dtype):
    output = model(inputs, ...)
```

This coerces all ops to `weight_dtype` without changing effective training
precision. Also eliminates the need to set `lora.dtype: bf16` for memory
savings — autocast handles the computation dtype at runtime.

In qwen-image-finetune: `src/qflux/trainer/qwen_image_edit_trainer.py` —
search for `torch.autocast(_autocast_device, dtype=self.weight_dtype)`.

### 6.7 Verify the adaptation before training

Write a throwaway import-test that loads each module you touched:

```python
# _test_residual_cuda.py — delete after use
import os, sys
sys.path.insert(0, "src")
os.environ["QFLUX_DOTENV_LOADED"] = "1"  # skip HF login on package import
import torch

# Add one line per modified module — examples:
from qflux.utils.device_utils import device_empty_cache  # noqa: F401
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer  # noqa: F401
print("[OK] all modified modules imported cleanly")

# If you touched attn_implementation logic, also assert:
# from qflux.models.load_model import _attn_implementation
# assert _attn_implementation() == "sdpa"
```

A clean run = no syntax errors and no broken references introduced by your
changes. End-to-end smoke is in §11.

## 7. Memory Optimization Catalog

> Each subsection 7.x starts with a **Trigger condition** — machine-judgable
> from §5 probe JSON / recommender output. Apply ONLY the subset §5 selected.
> Patterns are independent unless explicitly noted.

### 7.1 Pre-quantize transformer (NF4)

**Trigger condition**: optional; recommended when running training more than
once. A single exploratory run can skip this step (use online NF4 instead).

This skill always uses NF4 for the transformer — the question is *how* NF4
weights are loaded at each training start:

| | Pre-quantized (`transformer_path` set) | Online NF4 (`transformer_path` null) |
|---|---|---|
| Disk space | +~10 GB for the NF4 checkpoint dir | No extra |
| Per-run startup | Fast — load ~10 GB NF4 directly | Slower — stream bf16 shards from the original repo and quantize on the fly; Triton `quantize_4bit` kernel JIT-compiles on first use |
| Training-time memory | Essentially the same as online NF4 | Essentially the same as pre-quantized |
| Original model needed on disk | Still required for text_encoder, VAE, scheduler | Still required |

The bf16 transformer checkpoint on disk is ~38 GB; the NF4 output is ~10 GB
(~4× smaller). During online quantization the model is loaded shard-by-shard
so the peak RAM is much less than 38 GB, but the process still takes
meaningful time on every run start.

**One-time prep** — output is reused indefinitely:

```bat
templates\launchers\quantize_xpu.bat <SRC> <DST>
```

or manually (after activating conda + setvars.bat):

```bat
python templates\quantize_transformer_nf4.py --src <SRC> --dst <DST>
```

**After quantizing**, set `model.transformer_path` in your training config:

```yaml
model:
  pretrained_model_name_or_path: <SRC>   # still needed for text_encoder, VAE, etc.
  transformer_path: <DST>                # NF4-quantized DiT
  quantize_type: "nf4"
```

The qflux model loader reads `quantization_config` from `<DST>/config.json`
and skips online quantization automatically.

### 7.2 Pre-quantize text_encoder (NF4)

**Trigger condition** (machine-judgable):
- `ram_gb <= 32` → recommended. At 32 GB, the cache phase loads the ~15 GB
  bf16 text_encoder simultaneously with the dataset and VAE, pushing system
  RAM close to its physical limit and triggering virtual memory paging —
  slowing cache significantly.
- `32 < ram_gb <= 64` → optional (extra headroom).
- `ram_gb > 64` → not needed.

The Qwen2.5-VL text_encoder weighs ~15 GB at bf16 (15.4 GiB as shown in
Windows Explorer). Pre-quantizing to NF4 drops it to ~5.5 GB on disk.

**One-time prep** (run once per machine):

```bat
templates\launchers\quantize_text_encoder_xpu.bat <SRC> <DST>
```

or manually (after activating conda + setvars.bat):

```bat
python templates\quantize_text_encoder_nf4.py ^
    --src <path\to\Qwen-Image-Edit-pipeline> ^
    --dst <path\to\Qwen-Image-Edit-nf4-text-encoder>
```

**Add to config**:

```yaml
model:
  text_encoder_path: "<path\to\Qwen-Image-Edit-nf4-text-encoder>"
```

The qflux loader auto-detects `quantization_config` in the saved `config.json`
and skips online quantization on subsequent runs — identical to the
`transformer_path` pattern in §7.1.

**Measured effect** (32 GB AI PC, validated):

| Phase | bf16 text_encoder | NF4 text_encoder |
|---|---|---|
| Cache phase RAM | near physical limit — may trigger virtual memory paging | well within limit — no paging |
| Disk size | ~15 GB | ~5.5 GB |
| Cache throughput | may slow significantly if paging occurs | normal |
| Fit phase RAM | unchanged | unchanged — fit does not load text_encoder |

**Note on cache compatibility**: NF4 embeddings differ slightly from bf16.
If you have an existing cache built with bf16 text_encoder, rebuild it after
switching to `text_encoder_path`.

### 7.3 Mode-aware loading

**Trigger condition** (machine-judgable):
- Cache phase (`--cache`): always apply `skip_dit` — DiT is not invoked during
  embedding pre-compute, regardless of machine.
- Fit phase: apply `skip_text_encoder` when `cache.use_cache == true`, cache
  files exist on disk, AND `validation.enabled == false`. Recommended for all
  AI PC tiers; effectively mandatory at 32 GB and below.

The trainer (`qflux.trainer.qwen_image_edit_trainer`) loads only the components used
in the current run mode. Two skip-paths reclaim memory peaks that would otherwise
force quantization or smaller batches.

| Phase / mode | Skipped component | Measured savings | Why it's safe |
|---|---|---|---|
| `--cache` (TrMode.cache) | DiT (transformer, ~10 GB) | **REQUIRED** — loading DiT alongside text_encoder (~15 GB) during cache phase exceeds CPU RAM on 32 GB AI PC and crashes. Cache phase only invokes text_encoder + VAE; DiT is not needed. | Cache phase only forwards through text_encoder + VAE; DiT is never invoked |
| `fit` + `cache.use_cache=true` + cache populated + validation disabled | text_encoder (Qwen2.5-VL bf16, ~15 GB) | **REQUIRED** — loading text_encoder alongside DiT during fit phase exceeds CPU RAM on 32 GB AI PC and crashes. With mode-aware load, CPU RAM stays within budget. | All training samples have pre-computed embeddings; text_encoder is only needed for live encoding (e.g., validation samples) |

VAE is **always** loaded and is not subject to any skip logic. Unlike
text_encoder (~15 GB), VAE is only ~250 MB — unloading it saves negligible
memory. Additionally, its config attributes (`scale_factor`, `latent_mean`,
`latent_std`, `z_dim`) are used throughout the trainer's forward pass
regardless of mode, so it must remain resident.

Implementation pattern (in
`src/qflux/trainer/qwen_image_edit_trainer.py`'s `load_model` after
applying §6 framework adaptation):

```python
from qflux.data.config import TrMode

is_cache_mode = self.config.mode == TrMode.cache
cache_will_serve_all = (
    self.config.mode == TrMode.fit
    and self.use_cache
    and self.cache_exist
    and not self.config.validation.enabled
)
skip_dit = is_cache_mode
skip_text_encoder = cache_will_serve_all
```

Required config flags to enable the fit-skip path:

```yaml
cache:
  use_cache: true        # see §7.4 cache-first workflow
  cache_dir: <path>      # cache must exist before fit starts
validation:
  enabled: false         # or run validation in a separate pass after fit
```

If you need validation during fit (periodic sample generation), the
text_encoder MUST stay loaded — the ~15 GB cost is unavoidable. On 32 GB
machines this typically forces NF4 text_encoder (§7.2) instead of skipping.

### 7.4 Cache-first workflow

**Trigger condition** (machine-judgable): apply when `cache.use_cache=true`, which the
§5 recommender sets whenever `text_encoder` size ≥ trainable model size. For
qwen-image-edit specifically: text_encoder (Qwen2.5-VL bf16, ~15 GB) dwarfs the
trainable LoRA path on the DiT (~50 MB), so cache mode is recommended on every
AI PC tier. Skip only if the user explicitly disables caching.

For qwen-image-edit, the text_encoder (Qwen2.5-VL) and VAE produce conditioning
signals without themselves being trained. Pre-compute their outputs once and reload
from disk each step — this frees encoder memory entirely during the training loop,
leaving only the ~10 GB DiT (NF4) + LoRA + Adam8bit state on XPU during fit.

Two-phase flow:

```
[cache step]  python -m qflux.main --config <cfg> --cache
              ├─ load text_encoder + VAE onto XPU (DiT skipped — see §7.3)
              ├─ encode all training samples → save embeddings to cache_dir
              └─ encoders freed after pass; build time scales with dataset size

[fit step]    accelerate launch ... -m qflux.main --config <cfg>
              ├─ cache detected → skip loading text_encoder (see §7.3)
              ├─ only DiT (NF4) + LoRA + optimizer resident on XPU
              └─ each step: load pre-computed embeddings from disk
```

Config block:

```yaml
cache:
  devices:
    text_encoder: xpu:0
    vae: xpu:0
  cache_dir: "outputs/<run_name>/cache"
  use_cache: true
```

Reuse semantics: the cache is built once per dataset and reused across training runs
(LoRA rank sweeps, learning-rate sweeps, etc., all hit the same cache dir). Only
rebuild when the dataset itself changes.

Pairs naturally with §7.3 (mode-aware loading), which is what actually skips
the encoder loads during fit.



## 8. Training Execution

By the time the agent reaches §8: env is set up (§2, §4), framework is
adapted (§6), dataset is validated (§3), and `recommend_config.py` has
emitted a YAML config plus a list of selected §7 patterns to apply. §8
runs the actual training.

### 8.1 Apply selected §7 patterns

The §5 recommender printed a list of patterns to stderr. Apply each before
launching:

| Pattern | What "applying" means |
|---|---|
| §7.1 Pre-quantize transformer | **Optional.** Run `templates/quantize_transformer_nf4.py` once; set `model.transformer_path` in the config. Skip to use online NF4 (slower startup, same training memory). |
| §7.2 Pre-quantize text_encoder | **For 32 GB machines**: run `templates\launchers\quantize_text_encoder_xpu.bat` once; set `model.text_encoder_path` in config. Reduces cache-phase RAM by ~8 GB. See §7.2. |
| §7.3 Mode-aware loading | Already implemented in qflux trainer (after §6 adaptation); verify `cache.use_cache: true` in the config and `validation.enabled: false` for fit. |
| §6.6 NF4 QLoRA patches | NF4-A1 (torch_dtype) + NF4-A2 (autocast) baked into qflux after §6 adaptation. Verify with §6.7 import test. |
| §7.4 Cache-first workflow | Run cache phase before fit — see §8.2. |
| §5.1 all other settings | Already in recommender output YAML; no separate action needed. |

### 8.2 Cache phase (one-time per dataset)

```bat
templates\launchers\train_xpu.bat <config.yaml> --cache
```

Runs the text_encoder and VAE forward over every training sample and writes
embeddings to `cache.cache_dir`. Build time scales linearly with sample count
and image resolution; a dataset of ~30-50 samples at default [384, 672]
resolution completes in one pass.

PASS: process exits 0; `cache.cache_dir` contains a per-sample directory
structure (one folder per sample with embedding tensors). Re-running with
the same dataset + config is a no-op (cache is hash-keyed).

> **Corrupt cache (interrupted run)**: If a previous cache run was interrupted,
> it may leave empty or partial files. Re-running will fail with
> `json.JSONDecodeError` or `KeyError` on the first sample. Fix: delete the
> entire `cache.cache_dir` directory and re-run the cache phase from scratch.

### 8.3 Fit phase

```bat
templates\launchers\train_xpu.bat <config.yaml>
```

Launches `accelerate launch --config_file accelerate_config_xpu.yaml -m qflux.main ...`.
Logs go to stdout; checkpoints land in
`<logging.output_dir>/checkpoint-<epoch>-<step>/`.

PASS criteria for a healthy run:
- **Step 1 completes without exception**
- **`torch.xpu.memory_allocated() > 0` during the step** — sample from a
  second CMD:
  ```bat
  python -c "import torch; print(torch.xpu.memory_allocated()/1e9, 'GB')"
  ```
  A near-zero value means compute is happening on CPU (typically §10 (troubleshooting)
  not applied).
- **Loss is finite and shows a downward trend after the warmup period**
  (batch-level fluctuation is normal; the warming period is set by
  `train.warmup_steps` in the config).

### 8.4 Resume from checkpoint

If interrupted, set `resume: <checkpoint-path>` in the config and re-launch.
qflux's checkpoint loader handles state restoration.

### 8.5 Monitoring across longer runs

The training log emits per-step `xpu memory: N.NN GB` lines (decimal GB,
matching Python memory tools; approximately 7% higher than the GiB values
Windows Explorer displays — both describe the same memory). Memory should
be stable across steps; linear growth indicates a tensor or gradient leak.

If step time degrades mid-run, check the Triton kernel cache (§10.4) —
stale kernels after a oneAPI / torch upgrade can manifest as erratic
slowdowns.

## 9. Validation

Validation answers the question: **did the LoRA learn the intended edit?**
The approach has three layers of increasing depth — use as many as needed.

### 9.1 In-training visual monitoring (TensorBoard sampling)

The framework can generate sample outputs during training at regular
intervals and log them to TensorBoard, giving a visual sense of progress
without writing any extra code.
*(Source: `docs/guide/validation_sampling.md`, qwen-image-finetune v3.1.0)*

**Configure in the training YAML:**

```yaml
validation:
  enabled: true
  steps: 100          # generate samples every 100 training steps
  max_samples: 2      # 2-5 is practical; more = slower validation pass
  seed: 42            # fixed seed for consistent comparison across checkpoints
  dataset:
    class_path: "qflux.data.dataset.ImageDataset"
    init_args:
      dataset_path:
        - split: test
          repo_id: <your-dataset>
      selected_control_indexes: [1]
      processor:
        class_path: "qflux.data.preprocess.ImageProcessor"
        init_args:
          process_type: center_crop
          target_size: [384, 672]
          controls_size: [[384, 672], [512, 512]]
```

**Start TensorBoard** (in a separate terminal):
```bash
tensorboard --logdir=outputs/<run_name>/logs
```
Then open the **Images** tab to compare outputs across checkpoints.

> ⚠️ **32 GB AI PC constraint**: `validation.enabled: true` disables the
> §7.3 mode-aware loading optimization, forcing `text_encoder` to stay in
> memory during fit. On 32 GB machines this causes OOM (see §7.3). Options:
> - **Recommended**: keep `validation.enabled: false` during training; run §9.2 post-training instead.
> - Train on a 64 GB+ machine where the extra ~15 GB fits.
> - §7.2 NF4 text_encoder reduces text_encoder from ~15 GB to ~5.5 GB, which may make in-training validation feasible on 32 GB — not tested with `validation.enabled: true`.

### 9.2 Post-training visual comparison (inference_compare.py)

After training completes, run inference twice on a held-out test set — once
with the base model and once with the trained LoRA — then compare the outputs.

**Getting a test split**

If your dataset is a HuggingFace dataset (downloaded locally), the test split
parquet is typically at `<dataset_dir>/data/test-*.parquet`. If your dataset
is a local directory, hold out 3–5 sample pairs before training and use those.
Point `--test-parquet` at the parquet file, or adapt `inference_compare.py`
to load local images directly if you have no parquet.

**Running** (from the qwen-image-finetune project root, with conda env active):

```bat
REM Activate environment first (same as training)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
call conda activate qwen-image-edit-xpu
set QFLUX_DOTENV_LOADED=1
set PYTHONUTF8=1

REM Step 1 — base model (no LoRA)
python templates\inference_compare.py ^
    --config <config.yaml> ^
    --test-parquet <path\to\test.parquet> ^
    --output-dir outputs\compare\<run>\base ^
    --num-samples 5

REM Step 2 — LoRA-applied
python templates\inference_compare.py ^
    --config <config.yaml> ^
    --test-parquet <path\to\test.parquet> ^
    --output-dir outputs\compare\<run>\lora ^
    --num-samples 5 ^
    --lora-weight outputs\<run>\checkpoint-<step>\pytorch_lora_weights.safetensors
```

Each output directory contains `sample_NN_<id>.png` (model output),
`sample_NN_<id>__target.png` (ground-truth if available),
and `manifest.json` with prompts and metadata.

**Quality and speed trade-offs**

Two parameters worth tuning to improve output quality, independently of the
LoRA weights themselves:

- **Model precision**: higher-precision DiT and text encoder consistently
  produce better results. Three configurations, ordered by quality:

  | Config | DiT | Text encoder | Required system RAM |
  |---|---|---|---|
  | A | NF4 on XPU | NF4 on XPU | 32 GB + XPU cap (§2.3) |
  | B *(default)* | NF4 on XPU | bf16 on CPU | 32 GB |
  | C | bf16 on XPU | bf16 on CPU | ≥64 GB |

  `inference_compare.py` uses config B by default. To use config A, set
  `model.text_encoder_path` in your config (§7.2) so both models run on XPU.

- **Inference steps** (`--num-inference-steps`, default 20): more steps improve
  quality at the cost of time. Start with the default and increase only if the
  output quality is insufficient.

> **32 GB AI PC — constraints by config**:
> - **Config B** (default): system RAM approaches 31–32 GB. On some 32 GB
>   machines, bf16 text_encoder (~15 GB) may cause OOM when loading alongside
>   NF4 DiT. If this happens, fall back to Config A by setting
>   `model.text_encoder_path` (§7.2) — this trades some output quality for
>   reliability.
> - **Config A** (both models on XPU): XPU demand is ~20 GB at [384,672]
>   resolution. If you see `UR_RESULT_ERROR_DEVICE_LOST` or a silent crash,
>   the GPU shared memory ceiling (§2.3) may need to be set higher — the
>   default registry value is often insufficient for this configuration.
> - **Higher inference steps** (50+): XPU memory accumulates across the
>   denoising loop; confirm stability at the default 20 steps before increasing.
>
> Other options:
> - `--num-samples 3` for a quicker comparison with fewer samples.
> - Monitor with `python templates\monitor_xpu_memory.py` from a second terminal.

### 9.3 Visual evaluation checklist

When inspecting the base vs. LoRA outputs, apply these three questions for
each sample. They align with established image-editing evaluation dimensions: ⁴

| Dimension | Question to ask |
|---|---|
| **Instruction adherence** | Does the LoRA output follow the edit prompt? (e.g., for "Add the character to the image" — is the character present in the right place?) |
| **Edit quality** | Does the edit look natural and seamless? (no obvious artefacts, blurring, or inconsistency at edit boundaries) |
| **Detail preservation** | Are unedited regions unchanged between base and LoRA outputs? (background, non-target areas should look identical) |

A successful LoRA should pass all three on most samples. Failure patterns:
- **No change**: base and LoRA outputs are identical → training had no effect; check loss trajectory and training steps.
- **Instruction ignored**: character not added, wrong edit → LoRA learned something unrelated; check prompt consistency in dataset.
- **Degraded quality**: artefacts, color shifts, garbled regions → possible overfitting or learning rate too high.

---
⁴ ImgEdit (Ye et al., arXiv:2505.20275, May 2025), https://github.com/PKU-YuanGroup/ImgEdit. Introduces instruction adherence, editing quality, and detail preservation as the three evaluation dimensions for image editing; adopted here as a manual inspection framework. Evaluates image editing models generally, not specifically qwen-image-finetune.

## 10. Troubleshooting

Patterns specific to qwen-image-edit NF4 QLoRA training on Intel AI PC.

### 10.1 `RuntimeError: Expected all tensors to be on the same device`

The framework has a `torch.cuda.current_device()` call that was not replaced
with a device-agnostic wrapper (§6.1). Run the §6.2 scan to locate it and
replace with the appropriate `device_utils.py` helper or an explicit
`.to(self.accelerator.device)` call.

### 10.2 `torch.xpu.memory_allocated()` returns 0 during training

Two distinct causes:
- The framework has a direct `torch.cuda.memory_allocated()` call that was
  not replaced with a `device_utils` wrapper — run the §6.2 scan, locate it,
  and replace with `torch.xpu.memory_allocated()` guarded by
  `if torch.xpu.is_available()`.
- (NF4 + LoRA flow) the trainer moves the model to CPU before LoRA injection
  and never moves it back. Check `src/qflux/trainer/base_trainer.py` for
  `.to("cpu")` calls before `get_peft_model(...)`. If found, add an explicit
  `.to(self.accelerator.device)` after LoRA injection.

### 10.3 Flash Attention 2 not available

`attn_implementation="flash_attention_2"` in `from_pretrained` calls or
in `config.json` will fail — `flash_attn` is a CUDA-only extension. Replace
with `"sdpa"` (see §6.2).

### 10.4 Stale Triton cache after version change

After upgrading oneAPI, `torch+xpu`, or any `triton*` package, Triton may
load kernels compiled against the old version and fail. Typical symptoms:
`RuntimeError` during the first `Adam8bit.step()` or `quantize_4bit`, or a
confusing kernel-binary-format error in the traceback.

**Always clear both caches after any version change before the next training
run:**

```bat
rmdir /s /q "%USERPROFILE%\.triton"
rmdir /s /q "%LOCALAPPDATA%\Temp\torchinductor_%USERNAME%"
```

Both directories are safe to delete — they rebuild automatically on the next
Triton-backed op. If `rmdir` fails because a process holds the directory,
close all Python / training processes first.

### 10.5 `accelerator.device` is `cuda:0` instead of `xpu:0`

**Most common cause**: `accelerate_config_xpu.yaml` (§6.3) is not being
passed to `accelerate launch`, or the config file contains
`distributed_type: MULTI_GPU` instead of `NO`.

**Fix**: confirm the launcher passes `--config_file accelerate_config_xpu.yaml`
and the config has `distributed_type: NO` and `use_cpu: false`.

**Fallback for older accelerate versions** that lack native XPU device
detection (accelerate < 0.27):

```python
# After Accelerator() construction, force the device if detection failed:
if torch.xpu.is_available() and accelerator.device.type != "xpu":
    accelerator.state.device = torch.device("xpu:0")
```

### 10.6 `KeyError: Invalid key: 0` — `load_dataset` returned a `DatasetDict`

`load_dataset(name)` without a `split=` argument returns a `DatasetDict`.
Integer indexing or `len()` on it then fails (only string keys accepted).

Fix:

```python
from datasets import DatasetDict, load_dataset

dataset = load_dataset(dataset_name)
if isinstance(dataset, DatasetDict):
    split = "train" if "train" in dataset else next(iter(dataset))
    dataset = dataset[split]
```

### 10.7 Inference `TypeError: unsupported operand type for //: 'NoneType' and 'int'`

**Trigger**: calling `templates/inference_compare.py` without passing `height`
and `width` to `trainer.predict()`, so the trainer's `prepare_embeddings` path
receives `batch["height"] = None`.

**Fix** (already applied in the bundled `templates/inference_compare.py`):
derive height/width from the first control image before calling `predict()`:

```python
ctrl_w, ctrl_h = s["control_images"][0].size  # PIL.size = (width, height)
out_imgs = trainer.predict(..., height=ctrl_h, width=ctrl_w, ...)
```

If you see this in a custom inference script, apply the same pattern.

### 10.8 Inference `ValueError: offline mode — must specify weight_name`

**Trigger**: loading a LoRA checkpoint when `HF_HUB_OFFLINE=1` is set,
because `diffusers.load_lora_adapter()` calls `_best_guess_weight_name()`
which raises immediately in offline mode even for local file paths.

**Fix** (already applied in `qflux/trainer/base_trainer.py`): if
`pretrained_weight` ends in `.safetensors`, extract the parent directory and
filename, and pass `weight_name` explicitly to `load_lora_adapter()`. If you
are running a version of qflux that does not have this fix, either:

- Upgrade to a version that includes the fix, or
- Pass the checkpoint directory (not the full file path) and ensure the
  filename is `pytorch_lora_weights.safetensors` (PEFT default).

### 10.9 Cache-phase out-of-memory on tighter RAM tiers

If the cache phase (§8.2) hits OOM or the system swaps heavily, the cause
is usually that the bf16 text_encoder (~15 GB) plus dataset plus OS
overhead exceeds the available system RAM ceiling. On 32 GB machines this
is on the edge; 16 GB is over budget.

Mitigations, in order of effort:
1. Confirm §2.3 GPU shared memory ceiling has been raised — gives the
   iGPU more room.
2. Reduce dataset size for the first run (cache fewer samples; verify
   the rest of the pipeline before scaling up).
3. Use §7.2 NF4 text_encoder (validated): run `templates\launchers\quantize_text_encoder_xpu.bat`
   once and set `model.text_encoder_path` in config — reduces cache-phase RAM
   by ~8 GB, eliminating the paging bottleneck.
4. Run cache phase on a CPU-only pass (slower but uses less peak RAM):
   set `cache.devices.text_encoder: cpu` in the YAML.

### 10.10 Triton package errors (wrong package installed)

If you see Triton-related errors (`ImportError`, `RuntimeError` on first
kernel execution, or driver-conflict crashes) that persist after clearing
caches (§10.4), the environment may have the wrong Triton package — for
example `triton` (NVIDIA-targeting) or `triton-windows` installed alongside
`pytorch-triton-xpu` or `triton-xpu`.

**Fix**: uninstall all Triton variants, then reinstall only the correct one
for your `torch+xpu` version (see §2.7 version table):

```bat
pip uninstall triton pytorch-triton pytorch-triton-xpu triton-xpu triton-windows -y
REM Then install the correct package, e.g. for torch 2.11.0+xpu:
pip install triton-xpu==3.7.*
```

After reinstalling, clear the Triton caches (§10.4) before the next run.

### 10.11 Triton kernel compilation fails — Level Zero headers missing

If Triton-backed ops (`quantize_4bit`, `Adam8bit.step()`) fail with a Level
Zero-related error or `RuntimeError: Failed to find C compiler...`, Triton
cannot locate the Level Zero SDK headers it needs to JIT-compile kernels.

`templates/probe_hw.py` reports `"level_zero_sdk": null` when this is the
case.

**Check first** — the GPU driver may have already set `LEVEL_ZERO_V1_SDK_PATH`:

```bat
echo %LEVEL_ZERO_V1_SDK_PATH%
```

If it prints a real directory containing `include\level_zero\ze_api.h`,
you're done — this is not the issue.

**If null**, download the Level Zero SDK manually and set `ZE_PATH`:

1. Download `level-zero-win-sdk-<version>.zip` from
   https://github.com/oneapi-src/level-zero/releases (match the version
   compatible with your GPU driver's Level Zero loader).
2. Extract to a path with no spaces or non-ASCII characters.
3. Set `ZE_PATH` to the extracted directory (the one containing `include/`
   and `lib/`) **before** calling `setvars.bat`:

   ```bat
   set ZE_PATH=C:\path\to\level-zero-sdk
   call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
   ```

4. Add this `set ZE_PATH=...` line to your training launcher so it is set
   on every run.

### 10.12 Machine reboots unexpectedly during training

**Symptom**: the machine restarts without a BSOD or error log during
training — especially under high memory or XPU utilization. Can also
manifest as a hard hang followed by reboot.

**Known cause and fix**: outdated Intel Arc Graphics driver. Install the
latest version from:

https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html

Driver stability under sustained XPU load improved significantly in recent
releases. After installing, reboot as prompted.

**After the reboot — re-verify the shared GPU memory setting**: driver
installation can reset certain registry values. Re-check that
`SystemPartitionCommitLimitPercentage` is still set to your target value
(e.g. 75) by opening Registry Editor and navigating to:

```
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager
```

If it has reverted to the default (~57), re-apply the setting per §2.3 and
reboot again.

## 11. End-to-end Walkthrough

Sequence the agent runs once §2 pre-flight is human-confirmed. Each step
has a machine-judgable PASS signal; if a step fails, the agent should stop
and report rather than proceed — a silently-failed cache phase will surface
as confusing errors during fit, much costlier to debug.

| # | Step | Command | PASS signal |
|---|---|---|---|
| 1 | §3 dataset validate | `python templates/dataset_validate.py <path>` | exit 0; `valid: true` in JSON |
| 2 | §4 env active | `python templates/probe_hw.py` | `xpu_available: true`; `torch_version` ends `+xpu`; `bnb_xpu_available: true` |
| 3 | §5 recommend | `python templates/probe_hw.py \| python templates/recommend_config.py --out config.yaml --dataset-path <path>` | `config.yaml` parses; selected §7 patterns printed to stderr |
| 4 | §6 adaptation applied | run the §6.7 import-test | `[OK] all modified modules imported cleanly` |
| 5 | §7.1 pre-quantize transformer (if selected) | `templates\launchers\quantize_xpu.bat <pipeline> <nf4-out>` | Output dir has `config.json` with `quantization_config` field; ~10 GB total |
| 5.5 | §7.2 pre-quantize text_encoder (**32 GB machines only**) | `templates\launchers\quantize_text_encoder_xpu.bat <pipeline> <nf4-te-out>` | Output dir has `config.json` with `quantization_config` field; ~5.5 GB total |
| 6 | §7 patterns applied to YAML | inspect `config.yaml` | All keys from §5.1 tier table for the user's RAM tier present; `text_encoder_path` set if 32 GB machine |
| 7 | §8.2 cache phase | `templates\launchers\train_xpu.bat <config> --cache` | exit 0; `cache.cache_dir` contains per-sample subdirs |
| 8 | §8.3 fit step 1 | `templates\launchers\train_xpu.bat <config>` (first step may take longer as Triton JIT compiles) | Step completes; `xpu.memory_allocated() > 0`; loss finite |
| 9 | §8.3 fit to completion | continue from step 1 | Final loss meaningfully below initial; checkpoints written |
| 10 | §9 validate LoRA | Run `templates/inference_compare.py` twice (base + lora per §9.2); inspect outputs with §9.3 visual checklist | LoRA outputs visibly differ from base outputs; apply §9.3 checklist to judge direction and quality of the change |

For a small dataset (~30 samples), use **smooth loss** rather than per-step
raw loss to judge convergence — raw loss is noisy on small datasets due to
repeated batches. A sustained downward trend in smooth loss is the key signal,
regardless of the absolute values.

If resuming training from a checkpoint, note that the cosine LR schedule
restarts from its initial value, which can cause smooth loss to temporarily
rise before continuing to decrease. This is expected behavior, not a sign of
instability.

If the run fails at step 8 with near-zero XPU memory, check §10.2 — the
trainer may be running on CPU (likely a §6 adaptation issue). Check before
suspecting anything else.
