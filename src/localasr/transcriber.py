from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import gc
from typing import Any

from .resources import configure_model_cache, resolve_model


@dataclass(frozen=True)
class Segment:
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    speaker: str | int | None = None


@dataclass(frozen=True)
class ChunkTranscript:
    index: int
    source: Path
    start_seconds: float
    end_seconds: float
    text: str
    segments: list[Segment] = field(default_factory=list)
    warning: str | None = None
    raw: Any | None = None


class SenseVoiceTranscriber:
    def __init__(
        self,
        *,
        model_name: str,
        model_dir: Path | None,
        language: str,
        device: str,
        batch_size_s: int,
        merge_vad: bool,
        merge_length_s: int,
        max_single_segment_ms: int,
        use_itn: bool,
        trust_remote_code: bool,
        speaker_diarization: bool,
    ) -> None:
        self.language = language
        self.batch_size_s = batch_size_s
        self.merge_vad = merge_vad
        self.merge_length_s = merge_length_s
        self.use_itn = use_itn
        self.speaker_diarization = speaker_diarization
        self._loaded_with_speaker = speaker_diarization
        self._fallback_model = None
        model_dirs = [model_dir] if model_dir else []
        configure_model_cache(model_dir)

        from funasr import AutoModel

        kwargs: dict[str, Any] = {
            "model": resolve_model(model_name, model_dirs),
            "trust_remote_code": trust_remote_code,
            "vad_model": resolve_model("fsmn-vad", model_dirs),
            "vad_kwargs": {"max_single_segment_time": max_single_segment_ms},
            "device": device,
            "disable_update": True,
        }
        self._base_model_kwargs = dict(kwargs)
        if speaker_diarization:
            kwargs.update(
                {
                    "spk_model": resolve_model("cam++", model_dirs),
                    "punc_model": resolve_model("ct-punc", model_dirs),
                    "spk_mode": "vad_segment",
                }
            )
        self.model = AutoModel(**kwargs)

        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self.postprocess = rich_transcription_postprocess

    def transcribe_chunk(
        self,
        *,
        chunk_path: Path,
        chunk_index: int,
        start_seconds: float,
        end_seconds: float,
    ) -> ChunkTranscript:
        warning = None
        try:
            result = self._generate(self._active_model(), chunk_path)
        except Exception as exc:
            if not self.speaker_diarization:
                raise
            warning = f"说话人识别失败，已降级为普通转写：{exc}"
            self.disable_speaker_diarization()
            result = self._generate(self._get_fallback_model(), chunk_path)
        item = result[0] if result else {}
        if warning:
            item = {**item, "speaker_diarization_error": warning}
        text = self.postprocess(str(item.get("text", ""))).strip()
        segments = self._extract_segments(item, chunk_offset_ms=int(start_seconds * 1000))
        if segments:
            text = "\n".join(segment.text for segment in segments if segment.text).strip()
        return ChunkTranscript(
            index=chunk_index,
            source=chunk_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            text=text,
            segments=segments,
            warning=warning,
            raw=None,
        )

    def _generate(self, model: Any, chunk_path: Path) -> list[dict[str, Any]]:
        return model.generate(
            input=str(chunk_path),
            cache={},
            language=self.language,
            use_itn=self.use_itn,
            batch_size_s=self.batch_size_s,
            merge_vad=self.merge_vad,
            merge_length_s=self.merge_length_s,
        )

    def _active_model(self) -> Any:
        if self.speaker_diarization or not self._loaded_with_speaker:
            return self.model
        return self._get_fallback_model()

    def disable_speaker_diarization(self) -> None:
        self.speaker_diarization = False

    def close(self) -> None:
        self._fallback_model = None
        self.model = None
        gc.collect()

    def _get_fallback_model(self) -> Any:
        if self._fallback_model is None:
            from funasr import AutoModel

            self._fallback_model = AutoModel(**self._base_model_kwargs)
        return self._fallback_model

    def _extract_segments(self, item: dict[str, Any], chunk_offset_ms: int) -> list[Segment]:
        sentence_info = item.get("sentence_info") or []
        segments: list[Segment] = []
        for sentence in sentence_info:
            raw_text = str(sentence.get("text") or sentence.get("sentence") or "")
            text = self.postprocess(raw_text).strip()
            if not text or not _has_transcribable_text(text):
                continue
            start = _as_int(sentence.get("start"))
            end = _as_int(sentence.get("end"))
            segments.append(
                Segment(
                    text=text,
                    start_ms=start + chunk_offset_ms if start is not None else None,
                    end_ms=end + chunk_offset_ms if end is not None else None,
                    speaker=sentence.get("spk"),
                )
            )
        return segments


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_transcribable_text(text: str) -> bool:
    return any(char.isalnum() for char in text)
