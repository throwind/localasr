from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import multiprocessing
from pathlib import Path
import queue
from threading import Lock
import time
import uuid

from .config import AppConfig
from .diagnostics import append_memory_log, memory_summary
from .pipeline import FileResult, ProgressEvent, run_batch


@dataclass(frozen=True)
class TranscriptionRequest:
    input_dir: Path
    output_dir: Path
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
    language: str = "auto"
    engine: str = "funasr"
    device: str = "cpu"
    model: str = "iic/SenseVoiceSmall"
    model_dir: Path | None = None
    speaker_diarization: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "TranscriptionRequest":
        formats = data.get("formats", ("txt", "json", "srt"))
        if isinstance(formats, str):
            normalized_formats = tuple(part.strip().lower() for part in formats.split(",") if part.strip())
        else:
            normalized_formats = tuple(str(item).lower() for item in formats)
        audio_files = data.get("audio_files", ())
        if audio_files is None:
            normalized_audio_files = ()
        elif isinstance(audio_files, str):
            normalized_audio_files = tuple(Path(part.strip()) for part in audio_files.split(",") if part.strip())
        else:
            normalized_audio_files = tuple(Path(str(item)) for item in audio_files)
        return cls(
            input_dir=Path(str(data["input_dir"])),
            output_dir=Path(str(data["output_dir"])),
            audio_files=normalized_audio_files,
            recursive=bool(data.get("recursive", True)),
            formats=normalized_formats,
            overwrite=bool(data.get("overwrite", False)),
            keep_workdir=bool(data.get("keep_workdir", False)),
            chunk_seconds=int(data.get("chunk_seconds", 600)),
            boundary_search_seconds=int(data.get("boundary_search_seconds", 30)),
            overlap_seconds=int(data.get("overlap_seconds", 5)),
            silence_threshold_db=int(data.get("silence_threshold_db", -35)),
            silence_min_duration=float(data.get("silence_min_duration", 0.5)),
            language=str(data.get("language", "auto")),
            engine=str(data.get("engine", "funasr")),
            device=str(data.get("device", "cpu")),
            model=str(data.get("model", "iic/SenseVoiceSmall")),
            model_dir=Path(str(data["model_dir"])) if data.get("model_dir") else None,
            speaker_diarization=bool(data.get("speaker_diarization", False)),
        )

    def to_config(self) -> AppConfig:
        return AppConfig().with_overrides(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            audio_files=self.audio_files,
            recursive=self.recursive,
            formats=self.formats,
            overwrite=self.overwrite,
            keep_workdir=self.keep_workdir,
            chunk_seconds=self.chunk_seconds,
            boundary_search_seconds=self.boundary_search_seconds,
            overlap_seconds=self.overlap_seconds,
            silence_threshold_db=self.silence_threshold_db,
            silence_min_duration=self.silence_min_duration,
            language=self.language,
            engine=self.engine,
            device=self.device,
            model=self.model,
            model_dir=self.model_dir,
            speaker_diarization=self.speaker_diarization,
        )


@dataclass
class JobRecord:
    id: str
    request: TranscriptionRequest
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "等待开始"
    current_file: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    results: list[FileResult] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "current_file": self.current_file,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "request": {
                **asdict(self.request),
                "input_dir": str(self.request.input_dir),
                "output_dir": str(self.request.output_dir),
                "audio_files": [str(path) for path in self.request.audio_files],
                "model_dir": str(self.request.model_dir) if self.request.model_dir else None,
                "formats": list(self.request.formats),
            },
            "results": [serialize_file_result(result) for result in self.results],
            "logs": list(self.logs),
        }


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="localasr")
        self._jobs: dict[str, JobRecord] = {}
        self._monitors: dict[str, Future[None]] = {}
        self._processes: dict[str, multiprocessing.Process] = {}
        self._queues: dict[str, multiprocessing.Queue] = {}
        self._lock = Lock()
        self._context = multiprocessing.get_context("spawn")

    def submit(self, request: TranscriptionRequest) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(id=job_id, request=request)
        progress_queue = self._context.Queue()
        process = self._context.Process(
            target=run_job_worker,
            args=(job_id, request, progress_queue),
            name=f"localasr-worker-{job_id[:8]}",
        )
        with self._lock:
            self._jobs[job_id] = record
            self._queues[job_id] = progress_queue
            self._processes[job_id] = process
        process.start()
        future = self._executor.submit(self._monitor_job, job_id)
        with self._lock:
            self._monitors[job_id] = future
        return self.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return sorted((clone_job(job) for job in self._jobs.values()), key=lambda item: item.created_at, reverse=True)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return clone_job(self._jobs[job_id])

    def _monitor_job(self, job_id: str) -> None:
        self._update(job_id, status="running", stage="starting", started_at=time.time(), message="任务启动")
        append_memory_log("job_monitor_start", job_id)
        with self._lock:
            progress_queue = self._queues[job_id]
            process = self._processes[job_id]

        try:
            while True:
                try:
                    kind, payload = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    if not process.is_alive():
                        exitcode = process.exitcode
                        if exitcode not in (0, None):
                            message = f"worker 进程异常退出，exitcode={exitcode}"
                            append_memory_log("job_worker_crashed", f"{job_id} | {message}")
                            self._update(
                                job_id,
                                status="failed",
                                stage="failed",
                                finished_at=time.time(),
                                error=message,
                                message=f"失败：{message}",
                            )
                        break
                    continue

                if kind == "progress":
                    self._handle_progress(job_id, payload)
                elif kind == "completed":
                    process.join(timeout=5)
                    append_memory_log("job_completed", f"{job_id} | {memory_summary()}")
                    self._update(
                        job_id,
                        status="completed",
                        stage="done",
                        finished_at=time.time(),
                        progress=1.0,
                        message="任务完成",
                        results=payload,
                    )
                    break
                elif kind == "failed":
                    process.join(timeout=5)
                    error = str(payload)
                    append_memory_log("job_failed", f"{job_id} | {error}")
                    self._update(
                        job_id,
                        status="failed",
                        stage="failed",
                        finished_at=time.time(),
                        error=error,
                        message=f"失败：{error}",
                    )
                    break
        finally:
            if process.is_alive():
                process.join(timeout=1)
            self._cleanup_process_handles(job_id)

    def _handle_progress(self, job_id: str, event: ProgressEvent) -> None:
        self._update(
            job_id,
            progress=event.progress,
            stage=event.stage,
            message=event.message,
            current_file=event.current_file,
            log=f"[{event.stage}] {event.message}",
        )

    def _update(self, job_id: str, **changes: object) -> None:
        log = changes.pop("log", None)
        with self._lock:
            record = self._jobs[job_id]
            for key, value in changes.items():
                setattr(record, key, value)
            if log is not None:
                record.logs.append(str(log))

    def forget(self, job_id: str) -> None:
        process: multiprocessing.Process | None = None
        progress_queue: multiprocessing.Queue | None = None
        with self._lock:
            self._jobs.pop(job_id, None)
            future = self._monitors.pop(job_id, None)
            process = self._processes.pop(job_id, None)
            progress_queue = self._queues.pop(job_id, None)
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=3)
        if progress_queue is not None:
            close_queue(progress_queue)
        if future is not None and not future.done():
            future.cancel()
        append_memory_log("job_forget", job_id)

    def _cleanup_process_handles(self, job_id: str) -> None:
        with self._lock:
            progress_queue = self._queues.pop(job_id, None)
            self._processes.pop(job_id, None)
        if progress_queue is not None:
            close_queue(progress_queue)


def serialize_file_result(result: FileResult) -> dict[str, object]:
    return {
        "source": str(result.source),
        "txt_path": str(result.txt_path) if result.txt_path else None,
        "json_path": str(result.json_path) if result.json_path else None,
        "srt_path": str(result.srt_path) if result.srt_path else None,
        "skipped": result.skipped,
    }


def clone_job(job: JobRecord) -> JobRecord:
    return JobRecord(
        id=job.id,
        request=job.request,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        current_file=job.current_file,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        results=list(job.results),
        logs=list(job.logs),
    )


def run_job_worker(job_id: str, request: TranscriptionRequest, progress_queue: multiprocessing.Queue) -> None:
    append_memory_log("job_worker_start", job_id)

    def send_progress(event: ProgressEvent) -> None:
        progress_queue.put(("progress", event))

    try:
        results = run_batch(request.to_config(), progress_callback=send_progress)
    except Exception as exc:
        append_memory_log("job_worker_failed", f"{job_id} | {exc}")
        progress_queue.put(("failed", str(exc)))
        return
    append_memory_log("job_worker_completed", f"{job_id} | {memory_summary()}")
    progress_queue.put(("completed", results))


def close_queue(progress_queue: multiprocessing.Queue) -> None:
    try:
        progress_queue.close()
    except Exception:
        pass
    try:
        progress_queue.join_thread()
    except Exception:
        pass
