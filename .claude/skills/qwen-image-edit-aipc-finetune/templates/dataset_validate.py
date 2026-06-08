"""Validate a dataset for qwen-image-edit fine-tuning.

Detects the user's dataset format (local dir / HuggingFace parquet / CSV)
and runs format-appropriate checks BEFORE the user commits to a training
run. Outputs a JSON verdict.

Tested against the schema of `qflux.data.dataset.ImageDataset` and the
TsienDragon/character-composition dataset.

Usage:
    python templates/dataset_validate.py <dataset-path>

Examples:
    python templates/dataset_validate.py <path/to/local-dataset>
    python templates/dataset_validate.py <path/to/dataset.csv>
    python templates/dataset_validate.py <path/to/hf-parquet-dir>

Output (stdout): JSON verdict.
Exit code: 0 if valid, 1 otherwise.

Network policy: this script DOES NOT fetch remote HuggingFace repos. For a
remote `owner/repo` argument, it only sanity-checks the format string and
emits a warning recommending verification via `max_train_steps=1`.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

IMAGE_DIR_NAMES = ["training_images", "images", "target_images", "target", "targets"]
CONTROL_DIR_NAMES = ["control_images", "control", "condition_images", "controls"]

CONTROL_EXTRA_PATTERN = re.compile(r"_control_(\d+)\.(?:png|jpe?g|webp|bmp)$", re.IGNORECASE)
MASK_PATTERN = re.compile(r"_mask\.png$", re.IGNORECASE)


def detect_format(dataset_path: str) -> str:
    p = Path(dataset_path)
    if str(dataset_path).lower().endswith(".csv"):
        return "csv"
    if p.is_dir():
        if (p / "data").exists() and any((p / "data").glob("*.parquet")):
            return "huggingface"
        return "local"
    if "/" in str(dataset_path) and not p.exists():
        return "huggingface"
    return "unknown"


def _size_assessment(count: int) -> tuple[list, list]:
    """Return (issues, warnings) for the sample count."""
    issues, warnings = [], []
    if count < 5:
        issues.append(
            f"Sample count {count} below the hard floor of 5; pipeline cannot meaningfully train."
        )
    elif count < 10:
        warnings.append(
            f"Sample count {count} is in smoke-test territory (5-9); "
            "expect rapid overfitting within ~100 steps."
        )
    elif count < 30:
        warnings.append(
            f"Sample count {count} below recommended baseline of 30; "
            "expect mild quality variance for single-concept LoRA."
        )
    return issues, warnings


def validate_local(path: Path) -> dict:
    issues, warnings = [], []

    images_dir = next((path / n for n in IMAGE_DIR_NAMES if (path / n).exists()), None)
    if images_dir is None:
        return {
            "valid": False, "format": "local", "sample_count": 0,
            "issues": [f"No images directory found. Expected one of {IMAGE_DIR_NAMES} under {path}"],
            "warnings": [],
        }

    control_dir = next((path / n for n in CONTROL_DIR_NAMES if (path / n).exists()), None)
    if control_dir is None:
        return {
            "valid": False, "format": "local", "sample_count": 0,
            "issues": [f"No control directory found. Expected one of {CONTROL_DIR_NAMES} under {path}"],
            "warnings": [],
        }

    targets = []
    for f in images_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
            continue
        if MASK_PATTERN.search(f.name) or CONTROL_EXTRA_PATTERN.search(f.name):
            continue
        targets.append(f)

    if not targets:
        issues.append(f"No target images found in {images_dir}")
        return {"valid": False, "format": "local", "sample_count": 0,
                "issues": issues, "warnings": warnings}

    paired = 0
    no_control, no_prompt = [], []
    for t in targets:
        stem = t.stem
        has_control = any(
            (control_dir / f"{stem}{ext}").exists() for ext in IMG_EXTS
        )
        if not has_control:
            no_control.append(stem)
            continue
        has_prompt = (images_dir / f"{stem}.txt").exists() or (control_dir / f"{stem}.txt").exists()
        if not has_prompt:
            no_prompt.append(stem)
            continue
        paired += 1

    if no_control:
        sample = no_control[:5]
        issues.append(
            f"{len(no_control)} target images have no matching control: "
            f"{sample}{' ...' if len(no_control) > 5 else ''}"
        )
    if no_prompt:
        sample = no_prompt[:5]
        issues.append(
            f"{len(no_prompt)} samples have no prompt .txt file: "
            f"{sample}{' ...' if len(no_prompt) > 5 else ''}"
        )

    size_issues, size_warnings = _size_assessment(paired)
    issues.extend(size_issues)
    warnings.extend(size_warnings)

    return {
        "valid": len(issues) == 0,
        "format": "local",
        "sample_count": paired,
        "issues": issues,
        "warnings": warnings,
    }


def validate_csv(path: Path) -> dict:
    issues, warnings = [], []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as e:
        return {"valid": False, "format": "csv", "sample_count": 0,
                "issues": [f"Failed to read CSV: {e}"], "warnings": []}

    required = ["path_target", "prompt"]
    missing = [c for c in required if c not in cols]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    control_cols = [c for c in cols if "path_control" in c]
    if not control_cols:
        issues.append("No path_control* columns found")

    size_issues, size_warnings = _size_assessment(len(rows))
    issues.extend(size_issues)
    warnings.extend(size_warnings)

    return {
        "valid": len(issues) == 0,
        "format": "csv",
        "sample_count": len(rows),
        "issues": issues,
        "warnings": warnings,
    }


def validate_huggingface(repo_or_path: str) -> dict:
    issues, warnings = [], []
    sample_count = -1

    p = Path(repo_or_path)
    if p.is_dir():
        parquets = list((p / "data").glob("*.parquet"))
        if not parquets:
            return {"valid": False, "format": "huggingface", "sample_count": 0,
                    "issues": [f"Local dir {p} has no data/*.parquet files"],
                    "warnings": []}
        try:
            import pandas as pd
            df = pd.read_parquet(parquets[0])
            cols = df.columns.tolist()
            required = ["target_image", "control_images", "prompt"]
            missing = [c for c in required if c not in cols]
            if missing:
                issues.append(
                    f"Missing required columns in parquet: {missing} (need {required})"
                )
            sample_count = len(df)
            size_issues, size_warnings = _size_assessment(sample_count)
            issues.extend(size_issues)
            warnings.extend(size_warnings)
        except ImportError:
            warnings.append("pandas not available; could not inspect parquet schema")
            sample_count = len(parquets)
    else:
        if "/" not in repo_or_path:
            issues.append(
                f"Not a valid HF repo id (need 'owner/name' format): {repo_or_path}"
            )
        else:
            warnings.append(
                f"Remote HF repo '{repo_or_path}' — schema validation requires "
                "network (skill's local-only policy disallows). Verify by "
                "running training once with max_train_steps=1."
            )

    return {
        "valid": len(issues) == 0,
        "format": "huggingface",
        "sample_count": sample_count,
        "issues": issues,
        "warnings": warnings,
    }


def main():
    p = argparse.ArgumentParser(
        description="Validate a dataset for qwen-image-edit fine-tuning."
    )
    p.add_argument("dataset_path",
                   help="Local dir, HuggingFace repo id, or .csv file path")
    args = p.parse_args()

    fmt = detect_format(args.dataset_path)
    if fmt == "local":
        result = validate_local(Path(args.dataset_path))
    elif fmt == "huggingface":
        result = validate_huggingface(args.dataset_path)
    elif fmt == "csv":
        result = validate_csv(Path(args.dataset_path))
    else:
        result = {
            "valid": False, "format": "unknown", "sample_count": 0,
            "issues": [f"Could not detect format for {args.dataset_path}"],
            "warnings": [],
        }

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
