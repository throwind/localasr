from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import tomllib
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    input_dir: Path = Path("./audios")
    output_dir: Path = Path("./transcripts")
    audio_files: tuple[Path, ...] = ()
    recursive: bool = True
    formats: tuple[str, ...] = ("txt", "json", "srt")
    overwrite: bool = False
    keep_workdir: bool = False
    chunk_seconds: int = 600
    boundary_search_seconds: int = 30
    overlap_seconds: int = 5
    silence_threshold_db: int = -35
    silence_min_duration: float = 0.5
    sample_rate: int = 16000
    channels: int = 1
    engine: str = field(default_factory=lambda: os.environ.get("LOCALASR_ENGINE", "funasr"))
    model: str = "iic/SenseVoiceSmall"
    model_dir: Path | None = None
    language: str = "auto"
    device: str = "cpu"
    batch_size_s: int = 60
    merge_vad: bool = True
    merge_length_s: int = 15
    max_single_segment_ms: int = 30000
    use_itn: bool = True
    trust_remote_code: bool = True
    speaker_diarization: bool = False
    audio_extensions: tuple[str, ...] = field(
        default=(
            ".aac",
            ".aiff",
            ".alac",
            ".flac",
            ".m4a",
            ".mka",
            ".mkv",
            ".mov",
            ".mp3",
            ".mp4",
            ".ogg",
            ".opus",
            ".wav",
            ".webm",
            ".wma",
        )
    )

    @classmethod
    def from_file(cls, path: Path) -> "AppConfig":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        flat: dict[str, Any] = {}

        input_section = data.get("input", {})
        output_section = data.get("output", {})
        audio_section = data.get("audio", {})
        sensevoice_section = data.get("sensevoice", {})

        if "dir" in input_section:
            flat["input_dir"] = Path(input_section["dir"])
        if "recursive" in input_section:
            flat["recursive"] = bool(input_section["recursive"])
        if "dir" in output_section:
            flat["output_dir"] = Path(output_section["dir"])
        if "formats" in output_section:
            flat["formats"] = tuple(str(item).lower() for item in output_section["formats"])
        if "overwrite" in output_section:
            flat["overwrite"] = bool(output_section["overwrite"])
        if "keep_workdir" in output_section:
            flat["keep_workdir"] = bool(output_section["keep_workdir"])

        mapping = {
            "chunk_seconds": "chunk_seconds",
            "boundary_search_seconds": "boundary_search_seconds",
            "overlap_seconds": "overlap_seconds",
            "silence_threshold_db": "silence_threshold_db",
            "sample_rate": "sample_rate",
            "channels": "channels",
        }
        for source, target in mapping.items():
            if source in audio_section:
                flat[target] = int(audio_section[source])
        if "silence_min_duration" in audio_section:
            flat["silence_min_duration"] = float(audio_section["silence_min_duration"])

        sensevoice_mapping = {
            "engine": "engine",
            "model": "model",
            "model_dir": "model_dir",
            "language": "language",
            "device": "device",
            "batch_size_s": "batch_size_s",
            "merge_vad": "merge_vad",
            "merge_length_s": "merge_length_s",
            "max_single_segment_ms": "max_single_segment_ms",
            "use_itn": "use_itn",
            "trust_remote_code": "trust_remote_code",
            "speaker_diarization": "speaker_diarization",
        }
        for source, target in sensevoice_mapping.items():
            if source in sensevoice_section:
                value = sensevoice_section[source]
                if isinstance(getattr(cls(), target), bool):
                    flat[target] = bool(value)
                elif isinstance(getattr(cls(), target), int):
                    flat[target] = int(value)
                elif target == "model_dir":
                    flat[target] = Path(str(value))
                else:
                    flat[target] = str(value)

        return cls(**flat)

    def with_overrides(self, **kwargs: Any) -> "AppConfig":
        clean = {key: value for key, value in kwargs.items() if value is not None}
        if "formats" in clean and isinstance(clean["formats"], str):
            clean["formats"] = tuple(part.strip().lower() for part in clean["formats"].split(",") if part.strip())
        if "input_dir" in clean:
            clean["input_dir"] = Path(clean["input_dir"])
        if "output_dir" in clean:
            clean["output_dir"] = Path(clean["output_dir"])
        if "audio_files" in clean:
            clean["audio_files"] = tuple(Path(path) for path in clean["audio_files"])
        if "model_dir" in clean and clean["model_dir"]:
            clean["model_dir"] = Path(clean["model_dir"])
        return replace(self, **clean)

    def validate(self) -> None:
        if not self.input_dir.exists():
            raise ValueError(f"输入目录不存在：{self.input_dir}")
        if not self.input_dir.is_dir():
            raise ValueError(f"输入路径不是目录：{self.input_dir}")
        for audio_file in self.audio_files:
            if not audio_file.exists():
                raise ValueError(f"音频文件不存在：{audio_file}")
            if not audio_file.is_file():
                raise ValueError(f"音频路径不是文件：{audio_file}")
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds 必须大于 0")
        if self.boundary_search_seconds < 0:
            raise ValueError("boundary_search_seconds 不能小于 0")
        if self.overlap_seconds < 0:
            raise ValueError("overlap_seconds 不能小于 0")
        if self.overlap_seconds * 2 >= self.chunk_seconds:
            raise ValueError("overlap_seconds 不能超过 chunk_seconds 的一半")
        if self.silence_min_duration <= 0:
            raise ValueError("silence_min_duration 必须大于 0")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate 必须大于 0")
        if self.channels <= 0:
            raise ValueError("channels 必须大于 0")
        valid_engines = {"funasr", "sherpa-onnx"}
        if self.engine not in valid_engines:
            raise ValueError(f"不支持的转写引擎：{self.engine}")
        valid_formats = {"txt", "json", "srt"}
        if not self.formats:
            raise ValueError("至少需要指定一种输出格式")
        unknown_formats = set(self.formats) - valid_formats
        if unknown_formats:
            raise ValueError(f"不支持的输出格式：{', '.join(sorted(unknown_formats))}")
