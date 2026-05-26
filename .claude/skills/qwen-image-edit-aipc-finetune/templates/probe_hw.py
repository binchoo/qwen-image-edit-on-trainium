"""Probe the local Intel AI PC hardware + key software components.

Emits a JSON dict to stdout describing what's on this machine. Consumed by
templates/recommend_config.py (T2) to pick a training config that fits.

The output schema is the contract between probe_hw.py and recommend_config.py.
Keep field names + types stable; add new fields rather than renaming.

Schema:
    {
      "xpu_model":         str | null,   // e.g. "Intel(R) Arc(TM) 140V GPU"; null if no XPU
      "xpu_available":     bool,         // torch.xpu.is_available()
      "vram_gb":           float,        // get_device_properties(0).total_memory / 1e9
      "ram_gb":            float,        // psutil.virtual_memory().total / 1e9
      "ram_available_gb":  float,        // psutil.virtual_memory().available / 1e9
      "oneapi_version":    str | null,   // probed from `icpx --version`, null if not on PATH
      "conda_env":         str | null,   // CONDA_DEFAULT_ENV, null if not in conda
      "bnb_version":       str | null,   // bitsandbytes.__version__, null if not installed
      "bnb_xpu_available": bool,         // bnb has XPU backend module
      "torch_version":     str | null,
      "platform":          str,          // "windows" / "linux" / "darwin"
      "level_zero_sdk":    str | null    // LEVEL_ZERO_V1_SDK_PATH or ZE_PATH if set, else null
    }

Usage:
    python templates/probe_hw.py > probe.json
    python templates/recommend_config.py --probe probe.json --out config.yaml

Network policy: this script does NOT touch the network. All probes are local.

(See bottom of file for a sample output captured at skill-authoring time.)
"""
import argparse
import json
import os
import platform
import re
import subprocess


def _probe_xpu():
    info = {
        "xpu_model": None,
        "xpu_available": False,
        "vram_gb": 0.0,
        "torch_version": None,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            info["xpu_available"] = True
            props = torch.xpu.get_device_properties(0)
            info["xpu_model"] = props.name
            info["vram_gb"] = round(props.total_memory / 1e9, 2)
    except ImportError:
        pass
    return info


def _probe_ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "ram_gb": round(vm.total / 1e9, 2),
            "ram_available_gb": round(vm.available / 1e9, 2),
        }
    except ImportError:
        try:
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            return {"ram_gb": round(ram / 1e9, 2), "ram_available_gb": -1.0}
        except (ValueError, AttributeError):
            return {"ram_gb": -1.0, "ram_available_gb": -1.0}


def _probe_oneapi():
    """Return the Intel oneAPI **toolkit** version (e.g. '2025.3').

    Strategy:
    1. Parse the InstalledDir path from `icpx --version` output, which
       contains the two-part toolkit version: .../compiler/2025.3/...
       This is more reliable than the three-part compiler build string
       (e.g. 2025.3.3) that may differ slightly from the toolkit version.
    2. Fall back to parsing the Compiler line and keeping major.minor only.

    Requires `setvars.bat` to have been called (Windows) or oneAPI on PATH.
    Returns None if icpx is not found.
    """
    try:
        which = "where" if platform.system() == "Windows" else "which"
        r = subprocess.run(
            [which, "icpx"], capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return None
        r2 = subprocess.run(
            ["icpx", "--version"], capture_output=True, text=True, timeout=10
        )
        if r2.returncode != 0:
            return None
        output = r2.stdout + r2.stderr
        # Primary: extract toolkit version from InstalledDir path
        # e.g. "InstalledDir: C:\...\oneAPI\compiler\2025.3\bin\compiler"
        m = re.search(r"[Ii]nstalled[Dd]ir.*[/\\]compiler[/\\](\d+\.\d+)[/\\]", output)
        if m:
            return m.group(1)
        # Fallback: parse "Compiler X.X.X" and keep major.minor only
        m = re.search(r"Compiler (\d+\.\d+)", output)
        return m.group(1) if m else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _probe_conda():
    return os.environ.get("CONDA_DEFAULT_ENV")


def _probe_level_zero():
    """Return the Level Zero SDK path if available, else None.

    Triton XPU requires Level Zero headers to JIT-compile kernels.  They are
    normally provided automatically by the GPU driver and discoverable via
    LEVEL_ZERO_V1_SDK_PATH.  ZE_PATH is the manual fallback when the driver
    did not set LEVEL_ZERO_V1_SDK_PATH (download from
    https://github.com/oneapi-src/level-zero/releases and set ZE_PATH).

    Returns the path of whichever variable is set (LEVEL_ZERO_V1_SDK_PATH
    preferred), or None if neither is set.  A null value here means Triton
    kernel compilation will likely fail — see §10.11 in SKILL.md.
    """
    for var in ("LEVEL_ZERO_V1_SDK_PATH", "ZE_PATH"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _probe_bnb():
    info = {"bnb_version": None, "bnb_xpu_available": False}
    try:
        import bitsandbytes as bnb
        info["bnb_version"] = bnb.__version__
        try:
            import bitsandbytes.backends.xpu  # noqa: F401
            info["bnb_xpu_available"] = True
        except ImportError:
            pass
    except ImportError:
        pass
    return info


def main():
    p = argparse.ArgumentParser(
        description=(
            "Probe Intel AI PC hardware + key software components. "
            "Output is JSON consumed by recommend_config.py."
        )
    )
    p.parse_args()

    result = {
        **_probe_xpu(),
        **_probe_ram(),
        "oneapi_version": _probe_oneapi(),
        "conda_env": _probe_conda(),
        **_probe_bnb(),
        "platform": platform.system().lower(),
        "level_zero_sdk": _probe_level_zero(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


# Output notes:
# - vram_gb is the XPU-visible portion of unified memory; the system
#   reserves the rest for OS/CPU. On 32 GB machines, psutil may report
#   slightly more than the nominal 32 GB due to memory controller addressing.
# - oneapi_version is null when probing from a shell where setvars.bat has
#   not been called; recommend_config.py treats null as "not yet activated".
# - bnb_xpu_available=true confirms the bitsandbytes XPU backend is
#   compiled in (NF4 dequantize, gemv_4bit, etc.; see SKILL.md §6 / §7.4).
# - level_zero_sdk: path set by the GPU driver or ZE_PATH env var.
#   null here means Triton kernel compilation will likely fail — see §10.11.
