from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import os
import re
from collections.abc import Iterable
from typing import Any

from .resources import install_bundled_onnx_models
from .transcriber import ChunkTranscript, Segment


DEFAULT_MAX_DIARIZATION_WINDOW_SECONDS = 30


@dataclass(frozen=True)
class SherpaOnnxModelPaths:
    root: Path
    sensevoice_model: Path
    tokens: Path
    segmentation_model: Path
    embedding_model: Path
    vad_model: Path


class SherpaOnnxTranscriber:
    def __init__(
        self,
        *,
        model_dir: Path | None,
        language: str,
        use_itn: bool,
        speaker_diarization: bool,
    ) -> None:
        self.language = language
        self.use_itn = use_itn
        self.speaker_diarization = speaker_diarization
        self.paths = resolve_sherpa_onnx_paths(model_dir)

        configure_onnx_runtime_threads()

        import sherpa_onnx

        self._sherpa_onnx = sherpa_onnx
        self.recognizer = create_sense_voice_recognizer(
            sherpa_onnx=sherpa_onnx,
            model=self.paths.sensevoice_model,
            tokens=self.paths.tokens,
            language=language,
            use_itn=use_itn,
        )
        self._diarizer = None

    def transcribe_chunk(
        self,
        *,
        chunk_path: Path,
        chunk_index: int,
        start_seconds: float,
        end_seconds: float,
    ) -> ChunkTranscript:
        audio, sample_rate = read_audio(chunk_path)
        if self.speaker_diarization:
            segments = self._transcribe_with_speakers(audio, sample_rate, start_seconds)
            text = "\n".join(segment.text for segment in segments if segment.text).strip()
        else:
            text = self._recognize(audio, sample_rate)
            segments = [
                Segment(
                    text=text,
                    start_ms=int(start_seconds * 1000),
                    end_ms=int(end_seconds * 1000),
                )
            ] if text else []
        return ChunkTranscript(
            index=chunk_index,
            source=chunk_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            text=text,
            segments=segments,
            raw=None,
        )

    def disable_speaker_diarization(self) -> None:
        self.speaker_diarization = False

    def preload(self) -> None:
        if self.speaker_diarization:
            self._get_diarizer()

    def close(self) -> None:
        self._diarizer = None
        self.recognizer = None
        gc.collect()

    def _recognize(self, audio: Any, sample_rate: int) -> str:
        stream = self.recognizer.create_stream()
        try:
            stream.accept_waveform(sample_rate, audio)
            self.recognizer.decode_stream(stream)
            return clean_sense_voice_text(extract_result_text(stream.result))
        finally:
            del stream

    def _transcribe_with_speakers(self, audio: Any, sample_rate: int, chunk_start_seconds: float) -> list[Segment]:
        diarizer = self._get_diarizer()
        if sample_rate != diarizer.sample_rate:
            raise RuntimeError(f"sherpa-onnx 说话人识别需要 {diarizer.sample_rate}Hz 音频，当前为 {sample_rate}Hz")
        segments: list[Segment] = []
        max_window_seconds = max_diarization_window_seconds()
        for window_start_seconds, window_audio in iter_diarization_windows(audio, sample_rate, max_window_seconds):
            try:
                result = diarizer.process(window_audio).sort_by_start_time()
                window_segments: list[Segment] = []
                for item in result:
                    start = max(0.0, _safe_float(item.start, 0.0))
                    end = min(len(window_audio) / sample_rate, _safe_float(item.end, 0.0))
                    if end - start < 0.1:
                        continue
                    start_sample = int(start * sample_rate)
                    end_sample = int(end * sample_rate)
                    segment_audio = window_audio[start_sample:end_sample].copy()
                    text = self._recognize(segment_audio, sample_rate)
                    del segment_audio
                    if not text:
                        continue
                    absolute_start = chunk_start_seconds + window_start_seconds + start
                    absolute_end = chunk_start_seconds + window_start_seconds + end
                    window_segments.append(
                        Segment(
                            text=text,
                            start_ms=int(absolute_start * 1000),
                            end_ms=int(absolute_end * 1000),
                            speaker=int(item.speaker),
                        )
                    )
                segments.extend(window_segments)
            finally:
                del window_audio
                gc.collect()
        segments.sort(key=lambda item: item.start_ms if item.start_ms is not None else 0)
        if segments:
            return segments
        fallback_text = self._recognize(audio, sample_rate)
        return [
            Segment(
                text=fallback_text,
                start_ms=int(chunk_start_seconds * 1000),
                end_ms=int((chunk_start_seconds + len(audio) / sample_rate) * 1000),
            )
        ] if fallback_text else []

    def _get_diarizer(self):
        if self._diarizer is not None:
            return self._diarizer
        sherpa_onnx = self._sherpa_onnx
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.paths.segmentation_model),
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.paths.embedding_model),
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=-1,
                threshold=0.5,
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError("sherpa-onnx 说话人识别模型配置无效")
        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        return self._diarizer


def configure_onnx_runtime_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def max_diarization_window_seconds() -> int:
    raw_value = os.environ.get("LOCALASR_MAX_DIARIZATION_SECONDS", "")
    if not raw_value:
        return DEFAULT_MAX_DIARIZATION_WINDOW_SECONDS
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_DIARIZATION_WINDOW_SECONDS
    return max(10, min(value, 600))


def iter_diarization_windows(audio: Any, sample_rate: int, max_window_seconds: int) -> Iterable[tuple[float, Any]]:
    total_samples = len(audio)
    if total_samples <= 0:
        return
    window_samples = max(sample_rate, int(max_window_seconds * sample_rate))
    for start_sample in range(0, total_samples, window_samples):
        end_sample = min(total_samples, start_sample + window_samples)
        yield start_sample / sample_rate, audio[start_sample:end_sample].copy()


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_sherpa_onnx_paths(model_dir: Path | None) -> SherpaOnnxModelPaths:
    root = install_bundled_onnx_models(model_dir)
    paths = SherpaOnnxModelPaths(
        root=root,
        sensevoice_model=root / "sensevoice" / "model.int8.onnx",
        tokens=root / "sensevoice" / "tokens.txt",
        segmentation_model=root / "speaker" / "pyannote-segmentation.int8.onnx",
        embedding_model=root / "speaker" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        vad_model=root / "vad" / "silero_vad.onnx",
    )
    missing = [path for path in paths.__dict__.values() if isinstance(path, Path) and path != root and not path.exists()]
    if missing:
        raise FileNotFoundError("ONNX 模型文件缺失：" + "、".join(str(path) for path in missing))
    return paths


def create_sense_voice_recognizer(*, sherpa_onnx: Any, model: Path, tokens: Path, language: str, use_itn: bool):
    kwargs = {
        "model": str(model),
        "tokens": str(tokens),
        "use_itn": use_itn,
        "debug": False,
    }
    if language and language != "auto":
        kwargs["language"] = language
    try:
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(**kwargs)
    except TypeError:
        kwargs.pop("language", None)
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(**kwargs)


def read_audio(path: Path):
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio[:, 0], int(sample_rate)


def extract_result_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    if isinstance(result, dict):
        return str(result.get("text", ""))
    match = re.search(r'"text"\s*:\s*"([^"]*)"', str(result))
    return match.group(1) if match else str(result)


def clean_sense_voice_text(text: str) -> str:
    return re.sub(r"<\|[^|]+\|>", "", text).strip()
