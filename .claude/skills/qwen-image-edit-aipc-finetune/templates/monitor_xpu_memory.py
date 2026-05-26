"""Poll XPU memory allocation at a fixed interval.

Run from a second terminal while training or inference is happening to
confirm the XPU is actually being used and to track memory headroom.

Usage:
    python templates/monitor_xpu_memory.py
    python templates/monitor_xpu_memory.py --interval 1.0
    python templates/monitor_xpu_memory.py --count 10   # stop after 10 readings

Output (one line per interval):
    [21:05:03] XPU alloc: 15.77 GB  reserved: 17.05 GB  (21.80 GB capacity)
"""
import argparse
import time
from datetime import datetime


def main():
    p = argparse.ArgumentParser(
        description="Poll XPU memory allocation at a fixed interval."
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Sampling interval in seconds (default: 2.0)",
    )
    p.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of readings before stopping; 0 = unlimited (default: 0)",
    )
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("[monitor] torch is not installed in this environment.")
        return

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("[monitor] XPU not available — nothing to monitor.")
        return

    props = torch.xpu.get_device_properties(0)
    capacity_gb = props.total_memory / 1e9

    i = 0
    try:
        while args.count == 0 or i < args.count:
            ts = datetime.now().strftime("%H:%M:%S")
            alloc = torch.xpu.memory_allocated(0) / 1e9
            reserved = torch.xpu.memory_reserved(0) / 1e9
            print(
                f"[{ts}] XPU alloc: {alloc:.2f} GB  "
                f"reserved: {reserved:.2f} GB  "
                f"({capacity_gb:.2f} GB capacity)"
            )
            time.sleep(args.interval)
            i += 1
    except KeyboardInterrupt:
        print("\n[monitor] stopped.")


if __name__ == "__main__":
    main()
