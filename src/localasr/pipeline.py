from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import time
from typing import Iterable

from .audio import ensure_audio_tools, find_audio_files, split_audio
from .config import AppConfig
from .diagnostics import append_memory_log, memory_summary
from .onnx_transcriber import SherpaOnnxTranscriber
from .resources import DEFAULT_MODEL, DEFAULT_ONNX_MODEL
from .transcriber import ChunkTranscript, Segment, SenseVoiceTranscriber


@dataclass(frozen=True)
class FileResult:
    source: Path
    txt_path: Path | None
    json_path: Path | None
    srt_path: Path | None
    skipped: bool


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    progress: float
    current_file: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]
Transcriber = SenseVoiceTranscriber | SherpaOnnxTranscriber
TranscriberFactory = Callable[[float, str | None], Transcriber]


def run_batch(config: AppConfig, progress_callback: ProgressCallback | None = None) -> list[FileResult]:
    append_memory_log("run_batch_start", f"engine={config.engine} speaker={config.speaker_diarization}")
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(progress_callback, "scan", "正在扫描音频目录", 0.02)

    files = configured_audio_files(config)
    if not files:
        emit_progress(progress_callback, "done", "没有找到可处理的音频文件", 1.0)
        return []

    results: list[FileResult] = []
    pending_files: list[Path] = []
    for audio_path in files:
        txt_path, json_path, srt_path = planned_output_paths(audio_path, config)
        expected = [path for path in (txt_path, json_path, srt_path) if path is not None]
        if expected and all(path.exists() for path in expected) and not config.overwrite:
            results.append(FileResult(audio_path, txt_path, json_path, srt_path, skipped=True))
            emit_progress(progress_callback, "skip", f"跳过已有结果：{audio_path.name}", 0.05, str(audio_path))
        else:
            pending_files.append(audio_path)

    if not pending_files:
        emit_progress(progress_callback, "done", "所有音频都已有输出结果", 1.0)
        return results

    emit_progress(progress_callback, "audio_tools", "正在检查 ffmpeg/ffprobe", 0.08)
    ensure_audio_tools()

    transcriber: Transcriber | None = None
    preload_executor: ThreadPoolExecutor | None = None
    preload_future: Future[Transcriber] | None = None

    if config.engine == "sherpa-onnx":
        emit_progress(progress_callback, "load_model", "正在后台预加载 ONNX 模型", 0.10)
        append_memory_log("model_preload_start", "engine=sherpa-onnx")
        preload_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="localasr-model-preload")
        preload_future = preload_executor.submit(preload_transcriber, config)

    def get_transcriber(progress: float, current_file: str | None) -> Transcriber:
        nonlocal transcriber
        if transcriber is None:
            if preload_future is not None:
                if not preload_future.done():
                    emit_progress(
                        progress_callback,
                        "load_model",
                        f"正在等待 {engine_label(config.engine)} 模型预加载完成",
                        progress,
                        current_file,
                    )
                transcriber = preload_future.result()
                append_memory_log("model_preload_done", f"engine={config.engine}")
            else:
                emit_progress(
                    progress_callback,
                    "load_model",
                    f"正在加载 {engine_label(config.engine)} 模型",
                    progress,
                    current_file,
                )
                transcriber = create_transcriber(config)
        return transcriber

    try:
        total = len(pending_files)
        for file_index, audio_path in enumerate(pending_files, start=1):
            base_progress = 0.12 + ((file_index - 1) / total) * 0.86
            emit_progress(
                progress_callback,
                "file_start",
                f"开始处理 {audio_path.name} ({file_index}/{total})",
                base_progress,
                str(audio_path),
            )
            append_memory_log("file_start", audio_path.name)
            results.append(
                process_one(
                    audio_path,
                    config,
                    get_transcriber,
                    progress_callback=progress_callback,
                    progress_start=base_progress,
                    progress_end=0.12 + (file_index / total) * 0.86,
                )
            )
            append_memory_log("file_done", f"{audio_path.name} | {memory_summary()}")
        emit_progress(progress_callback, "memory", f"转写完成，{memory_summary()}", 0.99)
        emit_progress(progress_callback, "done", "转写任务完成", 1.0)
        return results
    finally:
        if preload_executor is not None:
            preload_executor.shutdown(wait=True, cancel_futures=True)
        if transcriber is None and preload_future is not None and preload_future.done() and not preload_future.cancelled():
            try:
                transcriber = preload_future.result()
            except Exception:
                pass
        if transcriber is not None:
            close = getattr(transcriber, "close", None)
            if callable(close):
                close()
        transcriber = None
        append_memory_log("run_batch_cleanup", "模型对象已释放")


def create_transcriber(config: AppConfig) -> Transcriber:
    if config.engine == "sherpa-onnx":
        return SherpaOnnxTranscriber(
            model_dir=config.model_dir,
            language=config.language,
            use_itn=config.use_itn,
            speaker_diarization=config.speaker_diarization,
        )
    return SenseVoiceTranscriber(
        model_name=config.model,
        model_dir=config.model_dir,
        language=config.language,
        device=config.device,
        batch_size_s=config.batch_size_s,
        merge_vad=config.merge_vad,
        merge_length_s=config.merge_length_s,
        max_single_segment_ms=config.max_single_segment_ms,
        use_itn=config.use_itn,
        trust_remote_code=config.trust_remote_code,
        speaker_diarization=config.speaker_diarization,
    )


def preload_transcriber(config: AppConfig) -> Transcriber:
    transcriber = create_transcriber(config)
    try:
        preload = getattr(transcriber, "preload", None)
        if callable(preload):
            preload()
        return transcriber
    except Exception:
        close = getattr(transcriber, "close", None)
        if callable(close):
            close()
        raise


def engine_label(engine: str) -> str:
    if engine == "sherpa-onnx":
        return "ONNX"
    return "SenseVoice"


def process_one(
    audio_path: Path,
    config: AppConfig,
    get_transcriber: TranscriberFactory,
    progress_callback: ProgressCallback | None = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
) -> FileResult:
    txt_path, json_path, srt_path = planned_output_paths(audio_path, config)

    expected = [path for path in (txt_path, json_path, srt_path) if path is not None]
    if expected and all(path.exists() for path in expected) and not config.overwrite:
        emit_progress(progress_callback, "skip", f"跳过已有结果：{audio_path.name}", progress_end, str(audio_path))
        return FileResult(audio_path, txt_path, json_path, srt_path, skipped=True)

    started_at = time.perf_counter()
    append_memory_log("process_one_start", audio_path.name)
    workdir = cache_workdir(audio_path, config)
    if config.overwrite and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    emit_progress(
        progress_callback,
        "split",
        f"正在转码并切分：{audio_path.name}",
        progress_start + (progress_end - progress_start) * 0.15,
        str(audio_path),
    )
    chunks = split_audio(
        source=audio_path,
        workdir=workdir,
        chunk_seconds=config.chunk_seconds,
        sample_rate=config.sample_rate,
        channels=config.channels,
        boundary_search_seconds=config.boundary_search_seconds,
        overlap_seconds=config.overlap_seconds,
        silence_threshold_db=config.silence_threshold_db,
        silence_min_duration=config.silence_min_duration,
    )
    transcripts: list[ChunkTranscript] = []
    speaker_disabled_by_cache = False
    for chunk_index, chunk in enumerate(chunks, start=1):
        cache_path = chunk_transcript_path(workdir, chunk.index)
        if cache_path.exists() and not config.overwrite:
            transcript = load_chunk_transcript(cache_path)
            transcripts.append(transcript)
            if transcript.warning:
                speaker_disabled_by_cache = True
                emit_progress(
                    progress_callback,
                    "speaker_warning",
                    f"{audio_path.name} 片段 {chunk_index}：{transcript.warning}",
                    interpolate_progress(progress_start, progress_end, 0.15, 0.9, chunk_index, len(chunks)),
                    str(audio_path),
                )
            emit_progress(
                progress_callback,
                "cache",
                f"复用缓存 {audio_path.name}：片段 {chunk_index}/{len(chunks)}",
                interpolate_progress(progress_start, progress_end, 0.15, 0.9, chunk_index, len(chunks)),
                str(audio_path),
            )
            write_outputs(
                source=audio_path,
                transcripts=transcripts,
                txt_path=txt_path,
                json_path=json_path,
                srt_path=srt_path,
                config=config,
            )
            continue

        emit_progress(
            progress_callback,
            "transcribe",
            f"正在转写 {audio_path.name}：片段 {chunk_index}/{len(chunks)}",
            interpolate_progress(progress_start, progress_end, 0.15, 0.9, chunk_index - 1, len(chunks)),
            str(audio_path),
        )
        transcriber = get_transcriber(
            interpolate_progress(progress_start, progress_end, 0.15, 0.9, chunk_index - 1, len(chunks)),
            str(audio_path),
        )
        if speaker_disabled_by_cache:
            transcriber.disable_speaker_diarization()
        transcript = transcriber.transcribe_chunk(
            chunk_path=chunk.path,
            chunk_index=chunk.index,
            start_seconds=chunk.start_seconds,
            end_seconds=chunk.end_seconds,
        )
        transcript = trim_transcript_to_window(
            transcript,
            keep_start_seconds=chunk.keep_start_seconds,
            keep_end_seconds=chunk.keep_end_seconds,
        )
        save_chunk_transcript(cache_path, transcript)
        transcripts.append(transcript)
        write_outputs(
            source=audio_path,
            transcripts=transcripts,
            txt_path=txt_path,
            json_path=json_path,
            srt_path=srt_path,
            config=config,
        )
        if transcript.warning:
            emit_progress(
                progress_callback,
                "speaker_warning",
                f"{audio_path.name} 片段 {chunk_index}：{transcript.warning}",
                interpolate_progress(progress_start, progress_end, 0.15, 0.9, chunk_index, len(chunks)),
                str(audio_path),
            )
    emit_progress(
        progress_callback,
        "write",
        f"正在写入输出：{audio_path.name}",
        progress_start + (progress_end - progress_start) * 0.95,
        str(audio_path),
    )
    write_outputs(
        source=audio_path,
        transcripts=transcripts,
        txt_path=txt_path,
        json_path=json_path,
        srt_path=srt_path,
        config=config,
    )
    elapsed = time.perf_counter() - started_at
    emit_progress(
        progress_callback,
        "file_done",
        f"完成：{audio_path.name}，转写耗时 {format_elapsed(elapsed)}",
        progress_end,
        str(audio_path),
    )
    append_memory_log("process_one_done", f"{audio_path.name} elapsed={format_elapsed(elapsed)}")
    return FileResult(audio_path, txt_path, json_path, srt_path, skipped=False)


def output_stem(audio_path: Path, input_dir: Path, output_dir: Path) -> Path:
    try:
        relative = audio_path.relative_to(input_dir)
    except ValueError:
        relative = Path(audio_path.name)
    return output_dir / relative.with_suffix("")


def planned_output_paths(audio_path: Path, config: AppConfig) -> tuple[Path | None, Path | None, Path | None]:
    output_base = output_stem(audio_path, config.input_dir, config.output_dir)
    txt_path = output_base.with_suffix(".txt") if "txt" in config.formats else None
    json_path = output_base.with_suffix(".json") if "json" in config.formats else None
    srt_path = output_base.with_suffix(".srt") if "srt" in config.formats else None
    return txt_path, json_path, srt_path


def configured_audio_files(config: AppConfig) -> list[Path]:
    if config.audio_files:
        return sorted(Path(path) for path in config.audio_files)
    return find_audio_files(config.input_dir, config.recursive, config.audio_extensions)


def safe_workdir_name(path: Path) -> str:
    try:
        stat = path.stat()
        fingerprint = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        fingerprint = str(path.resolve())
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
    stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem).strip("_")
    return f"{stem[:80] or 'audio'}_{digest}"


def cache_workdir(path: Path, config: AppConfig | None = None) -> Path:
    name = safe_workdir_name(path)
    if config is None:
        return Path(tempfile.gettempdir()) / "localasr-cache" / name
    return Path(tempfile.gettempdir()) / "localasr-cache" / f"{name}_{slice_config_hash(config)}"


def slice_config_hash(config: AppConfig) -> str:
    payload = {
        "chunk_seconds": config.chunk_seconds,
        "boundary_search_seconds": config.boundary_search_seconds,
        "overlap_seconds": config.overlap_seconds,
        "silence_threshold_db": config.silence_threshold_db,
        "silence_min_duration": config.silence_min_duration,
        "sample_rate": config.sample_rate,
        "channels": config.channels,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def chunk_transcript_path(workdir: Path, index: int) -> Path:
    return workdir / f"chunk_{index + 1:05d}.json"


def save_chunk_transcript(path: Path, transcript: ChunkTranscript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": transcript.index,
        "source": str(transcript.source),
        "start_seconds": transcript.start_seconds,
        "end_seconds": transcript.end_seconds,
        "text": transcript.text,
        "warning": transcript.warning,
        "segments": [asdict(segment) for segment in transcript.segments],
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_chunk_transcript(path: Path) -> ChunkTranscript:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(
            text=str(item.get("text", "")),
            start_ms=item.get("start_ms"),
            end_ms=item.get("end_ms"),
            speaker=item.get("speaker"),
        )
        for item in payload.get("segments", [])
    ]
    return ChunkTranscript(
        index=int(payload["index"]),
        source=Path(str(payload["source"])),
        start_seconds=float(payload["start_seconds"]),
        end_seconds=float(payload["end_seconds"]),
        text=str(payload.get("text", "")),
        segments=segments,
        warning=payload.get("warning"),
    )


def trim_transcript_to_window(
    transcript: ChunkTranscript,
    *,
    keep_start_seconds: float,
    keep_end_seconds: float,
) -> ChunkTranscript:
    if not transcript.segments:
        return ChunkTranscript(
            index=transcript.index,
            source=transcript.source,
            start_seconds=keep_start_seconds,
            end_seconds=keep_end_seconds,
            text=transcript.text,
            segments=[],
            warning=transcript.warning,
            raw=transcript.raw,
        )

    keep_start_ms = int(keep_start_seconds * 1000)
    keep_end_ms = int(keep_end_seconds * 1000)
    kept_segments: list[Segment] = []
    for segment in transcript.segments:
        if segment.start_ms is None or segment.end_ms is None:
            kept_segments.append(segment)
            continue
        center_ms = (segment.start_ms + segment.end_ms) / 2
        if keep_start_ms <= center_ms < keep_end_ms:
            kept_segments.append(segment)
    text = "\n".join(segment.text for segment in kept_segments if segment.text).strip()
    return ChunkTranscript(
        index=transcript.index,
        source=transcript.source,
        start_seconds=keep_start_seconds,
        end_seconds=keep_end_seconds,
        text=text,
        segments=kept_segments,
        warning=transcript.warning,
        raw=transcript.raw,
    )


def write_outputs(
    *,
    source: Path,
    transcripts: list[ChunkTranscript],
    txt_path: Path | None,
    json_path: Path | None,
    srt_path: Path | None,
    config: AppConfig,
) -> None:
    if txt_path is not None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(render_txt(transcripts), encoding="utf-8")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(render_json(source, transcripts, config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if srt_path is not None:
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        srt_path.write_text(render_srt(transcripts), encoding="utf-8")


def render_txt(transcripts: Iterable[ChunkTranscript]) -> str:
    parts: list[str] = []
    current_speaker: str | int | None = None
    current_texts: list[str] = []

    def flush_current() -> None:
        nonlocal current_speaker, current_texts
        if not current_texts:
            return
        text = render_segment_text(current_speaker, join_segment_texts(current_texts))
        if text:
            parts.append(text)
        current_speaker = None
        current_texts = []

    for item in transcripts:
        if item.segments:
            for segment in item.segments:
                clean_text = segment.text.strip()
                if not clean_text or not has_transcribable_text(clean_text):
                    continue
                if current_texts and segment.speaker != current_speaker:
                    flush_current()
                current_speaker = segment.speaker
                current_texts.append(clean_text)
        elif item.text.strip():
            flush_current()
            parts.append(item.text.strip())
    flush_current()
    return "\n\n".join(parts).strip() + ("\n" if parts else "")


def render_json(source: Path, transcripts: list[ChunkTranscript], config: AppConfig) -> dict[str, object]:
    return {
        "source": str(source),
        "created_at": int(time.time()),
        "model": display_model(config),
        "engine": config.engine,
        "language": config.language,
        "chunks": [
            {
                "index": item.index,
                "start_seconds": item.start_seconds,
                "end_seconds": item.end_seconds,
                "text": item.text,
                "warning": item.warning,
                "segments": [asdict(segment) for segment in item.segments],
            }
            for item in transcripts
        ],
    }


def display_model(config: AppConfig) -> str:
    if config.engine == "sherpa-onnx" and config.model == DEFAULT_MODEL:
        return DEFAULT_ONNX_MODEL
    return config.model


def render_srt(transcripts: Iterable[ChunkTranscript]) -> str:
    blocks: list[str] = []
    index = 1
    for chunk in transcripts:
        segments = chunk.segments or [
            _chunk_as_segment(chunk),
        ]
        for segment in segments:
            start_ms = segment.start_ms if segment.start_ms is not None else int(chunk.start_seconds * 1000)
            end_ms = segment.end_ms if segment.end_ms is not None else int(chunk.end_seconds * 1000)
            text = render_segment_text(segment.speaker, segment.text)
            if not text:
                continue
            blocks.append(
                "\n".join(
                    [
                        str(index),
                        f"{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}",
                        text,
                    ]
                )
            )
            index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_segment_text(speaker: str | int | None, text: str) -> str:
    clean_text = text.strip()
    if not clean_text or not has_transcribable_text(clean_text):
        return ""
    if speaker is None:
        return clean_text
    return f"说话人 {speaker}: {clean_text}"


def join_segment_texts(texts: Iterable[str]) -> str:
    merged = ""
    for raw_text in texts:
        text = raw_text.strip()
        if not text:
            continue
        if merged and _needs_space_between(merged[-1], text[0]):
            merged += " "
        merged += text
    return merged


def _needs_space_between(left: str, right: str) -> bool:
    return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()


def has_transcribable_text(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _chunk_as_segment(chunk: ChunkTranscript):
    from .transcriber import Segment

    return Segment(
        text=chunk.text,
        start_ms=int(chunk.start_seconds * 1000),
        end_ms=int(chunk.end_seconds * 1000),
    )


def format_srt_time(total_ms: int) -> str:
    hours, rest = divmod(max(0, total_ms), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rest = divmod(total_seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分{secs:02d}秒"
    if minutes:
        return f"{minutes}分{secs:02d}秒"
    return f"{secs}秒"


def emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    progress: float,
    current_file: str | None = None,
) -> None:
    if callback is None:
        return
    callback(ProgressEvent(stage=stage, message=message, progress=max(0.0, min(1.0, progress)), current_file=current_file))


def interpolate_progress(
    start: float,
    end: float,
    phase_start: float,
    phase_end: float,
    index: int,
    total: int,
) -> float:
    if total <= 0:
        return start
    phase = phase_start + (phase_end - phase_start) * (index / total)
    return start + (end - start) * phase
