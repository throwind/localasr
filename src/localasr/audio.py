from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys

from .diagnostics import append_memory_log


class AudioToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioChunk:
    index: int
    path: Path
    start_seconds: float
    end_seconds: float
    keep_start_seconds: float
    keep_end_seconds: float


@dataclass(frozen=True)
class SilenceInterval:
    start_seconds: float
    end_seconds: float


def ensure_audio_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if resolve_audio_tool(name) is None]
    if missing:
        joined = "、".join(missing)
        raise AudioToolError(f"缺少音频工具：{joined}。请安装 ffmpeg，或把 ffmpeg/ffprobe 放到应用的 bin 目录。")


def resolve_audio_tool(name: str) -> str | None:
    executable = f"{name}.exe" if os.name == "nt" else name
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "bin" / executable)
    candidates.append(Path(sys.executable).resolve().parent / "bin" / executable)
    candidates.append(Path(__file__).resolve().parents[2] / "bin" / executable)
    for candidate in candidates:
        if candidate.exists() and audio_tool_is_usable(candidate):
            return str(candidate)
    system_tool = shutil.which(executable)
    if system_tool and audio_tool_is_usable(Path(system_tool)):
        return system_tool
    return None


def audio_tool_is_usable(path: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(path), "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        append_memory_log("audio_tool_unusable", f"{path} | {exc}")
        return False
    if completed.returncode == 0:
        return True
    message = (completed.stderr or completed.stdout or "").strip().splitlines()
    detail = message[0] if message else f"exit={completed.returncode}"
    append_memory_log("audio_tool_unusable", f"{path} | {detail}")
    return False


def probe_duration(path: Path) -> float:
    ffprobe = resolve_audio_tool("ffprobe")
    if ffprobe is None:
        raise AudioToolError("缺少音频工具：ffprobe。")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        raise AudioToolError(f"无法读取音频时长：{path}")
    return float(duration)


def transcode_chunk(
    source: Path,
    target: Path,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    channels: int,
) -> None:
    ffmpeg = resolve_audio_tool("ffmpeg")
    if ffmpeg is None:
        raise AudioToolError("缺少音频工具：ffmpeg。")
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    subprocess.run(cmd, check=True)


def detect_silences(source: Path, silence_threshold_db: int, silence_min_duration: float) -> list[SilenceInterval]:
    ffmpeg = resolve_audio_tool("ffmpeg")
    if ffmpeg is None:
        raise AudioToolError("缺少音频工具：ffmpeg。")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-vn",
        "-af",
        f"silencedetect=noise={silence_threshold_db}dB:d={silence_min_duration:.3f}",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    intervals: list[SilenceInterval] = []
    pending_start: float | None = None
    for line in completed.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and pending_start is not None:
            end = float(end_match.group(1))
            if end > pending_start:
                intervals.append(SilenceInterval(pending_start, end))
            pending_start = None
    return intervals


def plan_chunk_ranges(
    duration: float,
    chunk_seconds: int,
    boundary_search_seconds: int,
    overlap_seconds: int,
    silence_intervals: list[SilenceInterval],
) -> list[tuple[float, float, float, float]]:
    if duration <= 0:
        return [(0.0, 0.1, 0.0, 0.1)]
    if chunk_seconds <= 0:
        raise AudioToolError("chunk_seconds 必须大于 0")

    cut_points = [0.0]
    target = float(chunk_seconds)
    min_gap = max(10.0, float(overlap_seconds * 2 + 1))
    while target < duration:
        cut = choose_cut_point(
            target_seconds=target,
            duration=duration,
            boundary_search_seconds=boundary_search_seconds,
            silence_intervals=silence_intervals,
        )
        if cut - cut_points[-1] >= min_gap and duration - cut >= min_gap:
            cut_points.append(cut)
        target += float(chunk_seconds)
    cut_points.append(duration)

    ranges: list[tuple[float, float, float, float]] = []
    for index in range(len(cut_points) - 1):
        keep_start = cut_points[index]
        keep_end = cut_points[index + 1]
        actual_start = keep_start if index == 0 else max(0.0, keep_start - overlap_seconds)
        actual_end = keep_end if index == len(cut_points) - 2 else min(duration, keep_end + overlap_seconds)
        ranges.append((actual_start, actual_end, keep_start, keep_end))
    return ranges


def choose_cut_point(
    target_seconds: float,
    duration: float,
    boundary_search_seconds: int,
    silence_intervals: list[SilenceInterval],
) -> float:
    if boundary_search_seconds <= 0:
        return max(0.0, min(duration, target_seconds))
    window_start = max(0.0, target_seconds - boundary_search_seconds)
    window_end = min(duration, target_seconds + boundary_search_seconds)
    best_cut: float | None = None
    best_distance: float | None = None
    for interval in silence_intervals:
        start = max(window_start, interval.start_seconds)
        end = min(window_end, interval.end_seconds)
        if end <= start:
            continue
        candidate = (start + end) / 2
        distance = abs(candidate - target_seconds)
        if best_distance is None or distance < best_distance:
            best_cut = candidate
            best_distance = distance
    return best_cut if best_cut is not None else max(0.0, min(duration, target_seconds))


def split_audio(
    source: Path,
    workdir: Path,
    chunk_seconds: int,
    sample_rate: int,
    channels: int,
    boundary_search_seconds: int = 30,
    overlap_seconds: int = 5,
    silence_threshold_db: int = -35,
    silence_min_duration: float = 0.5,
) -> list[AudioChunk]:
    duration = probe_duration(source)
    silence_intervals = (
        detect_silences(source, silence_threshold_db, silence_min_duration) if boundary_search_seconds > 0 else []
    )
    ranges = plan_chunk_ranges(
        duration=duration,
        chunk_seconds=chunk_seconds,
        boundary_search_seconds=boundary_search_seconds,
        overlap_seconds=overlap_seconds,
        silence_intervals=silence_intervals,
    )
    chunks: list[AudioChunk] = []
    for index, (start, end, keep_start, keep_end) in enumerate(ranges):
        chunk_path = workdir / f"chunk_{index + 1:05d}.wav"
        if not chunk_path.exists() or chunk_path.stat().st_size <= 0:
            transcode_chunk(
                source=source,
                target=chunk_path,
                start_seconds=start,
                duration_seconds=max(0.1, end - start),
                sample_rate=sample_rate,
                channels=channels,
            )
        chunks.append(
            AudioChunk(
                index=index,
                path=chunk_path,
                start_seconds=start,
                end_seconds=end,
                keep_start_seconds=keep_start,
                keep_end_seconds=keep_end,
            )
        )
    return chunks


def find_audio_files(root: Path, recursive: bool, extensions: tuple[str, ...]) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    normalized = {ext.lower() for ext in extensions}
    return sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in normalized)
