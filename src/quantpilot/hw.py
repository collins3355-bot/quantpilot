"""Detect the hardware we're benchmarking on, so reports are self-describing."""

from __future__ import annotations

import platform
import subprocess


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def describe() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "chip": None,
        "memory_gb": None,
    }
    if platform.system() == "Darwin":
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        memsize = _sysctl("hw.memsize")
        if memsize and memsize.isdigit():
            info["memory_gb"] = round(int(memsize) / 1024**3)
    return info
