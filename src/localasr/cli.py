from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import AppConfig
from .pipeline import ProgressEvent, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localasr",
        description="批量读取目录中的音频文件，自动转码切片，并用 SenseVoice/FunASR 转写成本地文本。",
    )
    parser.add_argument("--config", type=Path, help="TOML 配置文件路径")
    parser.add_argument("--input-dir", type=Path, help="音频目录")
    parser.add_argument("--output-dir", type=Path, help="输出目录")
    parser.add_argument("--language", help='识别语言：auto、zh、en、yue、ja、ko、nospeech')
    parser.add_argument("--engine", choices=("funasr", "sherpa-onnx"), help="转写引擎，默认 funasr")
    parser.add_argument("--device", help="运行设备，例如 cpu、cuda:0")
    parser.add_argument("--model", help="模型名或本地模型路径，默认 iic/SenseVoiceSmall")
    parser.add_argument("--model-dir", type=Path, help="模型缓存/查找目录，默认系统应用数据目录下的 localasr/models")
    parser.add_argument("--chunk-seconds", type=int, help="外层缓存片段目标秒数，默认 600")
    parser.add_argument("--boundary-search-seconds", type=int, help="在目标切点前后搜索静音边界的秒数，默认 30")
    parser.add_argument("--overlap-seconds", type=int, help="相邻片段保留的上下文重叠秒数，默认 5")
    parser.add_argument("--silence-threshold-db", type=int, help="静音检测阈值，默认 -35")
    parser.add_argument("--silence-min-duration", type=float, help="判定静音所需的最短持续秒数，默认 0.5")
    parser.add_argument("--formats", help="输出格式，逗号分隔：txt,json,srt")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, help="是否递归扫描目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已经存在的输出文件")
    parser.add_argument("--keep-workdir", action="store_true", help="保留中间 wav 切片，便于排查")
    parser.add_argument("--speaker-diarization", action="store_true", help="启用说话人分离，需要对应 FunASR 版本支持")
    return parser


def load_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_file(args.config) if args.config else AppConfig()
    return config.with_overrides(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        language=args.language,
        engine=args.engine,
        device=args.device,
        model=args.model,
        model_dir=args.model_dir,
        chunk_seconds=args.chunk_seconds,
        boundary_search_seconds=args.boundary_search_seconds,
        overlap_seconds=args.overlap_seconds,
        silence_threshold_db=args.silence_threshold_db,
        silence_min_duration=args.silence_min_duration,
        formats=args.formats,
        recursive=args.recursive,
        overwrite=True if args.overwrite else None,
        keep_workdir=True if args.keep_workdir else None,
        speaker_diarization=True if args.speaker_diarization else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args)
        results = run_batch(config, progress_callback=print_progress)
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if not results:
        print("没有找到可处理的音频文件。")
        return 0

    for result in results:
        status = "跳过" if result.skipped else "完成"
        outputs = ", ".join(str(path) for path in (result.txt_path, result.json_path, result.srt_path) if path)
        print(f"{status}: {result.source} -> {outputs}")
    return 0


def print_progress(event: ProgressEvent) -> None:
    percent = round(event.progress * 100)
    print(f"[{percent:3d}%] {event.message}")
