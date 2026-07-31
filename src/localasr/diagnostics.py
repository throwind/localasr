from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import platform
import resource
import subprocess


def app_support_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "localasr"
    if platform.system() == "Windows":
        root = os.environ.get("APPDATA")
        if root:
            return Path(root) / "localasr"
    return Path.home() / ".local" / "share" / "localasr"


def log_dir() -> Path:
    path = app_support_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def memory_log_path() -> Path:
    return log_dir() / f"memory-{datetime.now():%Y%m%d}.log"


def current_rss_mb() -> float | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raw_value = result.stdout.strip().splitlines()[0].strip()
            if raw_value:
                return int(raw_value) / 1024
    except Exception:
        return None
    return None


def peak_rss_mb() -> float | None:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def memory_summary() -> str:
    current = current_rss_mb()
    peak = peak_rss_mb()
    parts: list[str] = []
    if current is not None:
        parts.append(f"RSS {current:.0f} MB")
    if peak is not None:
        parts.append(f"峰值 {peak:.0f} MB")
    return "，".join(parts) if parts else "内存未知"


def append_memory_log(stage: str, message: str = "") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = f" | {message}" if message else ""
    line = f"{timestamp} | pid={os.getpid()} | {stage} | {memory_summary()}{suffix}\n"
    try:
        memory_log_path().open("a", encoding="utf-8").write(line)
    except Exception:
        pass
