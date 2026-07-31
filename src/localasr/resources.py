from __future__ import annotations

from pathlib import Path
import os
import shutil
import sys


DEFAULT_MODEL = "iic/SenseVoiceSmall"
DEFAULT_ONNX_MODEL = "sherpa-onnx/SenseVoiceSmall-2025-09-09"
DEFAULT_SUPPORT_MODELS = (
    "iic/SenseVoiceSmall",
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "iic/speech_campplus_sv_zh-cn_16k-common",
    "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
)

MODEL_ALIASES = {
    DEFAULT_MODEL: Path("iic") / "SenseVoiceSmall",
    "fsmn-vad": Path("iic") / "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch": Path("iic") / "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "cam++": Path("iic") / "speech_campplus_sv_zh-cn_16k-common",
    "iic/speech_campplus_sv_zh-cn_16k-common": Path("iic") / "speech_campplus_sv_zh-cn_16k-common",
    "ct-punc": Path("iic") / "punc_ct-transformer_cn-en-common-vocab471067-large",
    "iic/punc_ct-transformer_cn-en-common-vocab471067-large": Path("iic")
    / "punc_ct-transformer_cn-en-common-vocab471067-large",
}


def default_model_cache_dir() -> Path:
    override = os.environ.get("LOCALASR_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return app_data_dir() / "models"


def app_data_dir() -> Path:
    override = os.environ.get("LOCALASR_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "localasr"
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "localasr"
        return Path.home() / "AppData" / "Roaming" / "localasr"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "localasr"
    return Path.home() / ".local" / "share" / "localasr"


def legacy_model_cache_dirs() -> tuple[Path, ...]:
    return (Path.home() / "localasr-models",)


def resource_roots() -> list[Path]:
    roots: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        roots.append(Path(sys._MEIPASS))
    roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parents[2])
    return roots


def bundled_onnx_models_dir() -> Path | None:
    for root in resource_roots():
        for candidate in (
            root / "onnx-runtime-models",
            root / "models" / "onnx-runtime-models",
            root / "packaging" / "onnx-runtime-models",
        ):
            if (candidate / "sensevoice" / "model.int8.onnx").exists():
                return candidate
    return None


def onnx_model_cache_dir(model_dir: Path | None = None) -> Path:
    return (model_dir or default_model_cache_dir()) / "onnx" / "sherpa-sensevoice-2025-09-09"


def install_bundled_onnx_models(model_dir: Path | None = None) -> Path:
    target = onnx_model_cache_dir(model_dir)
    if onnx_models_ready(target):
        return target
    source = bundled_onnx_models_dir()
    if source is None:
        raise FileNotFoundError("找不到内置 ONNX 模型资源，请确认离线版 DMG 已包含 onnx-runtime-models。")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp")
    if tmp_target.exists():
        shutil.rmtree(tmp_target)
    shutil.copytree(source, tmp_target)
    if target.exists():
        shutil.rmtree(target)
    tmp_target.replace(target)
    return target


def onnx_models_ready(root: Path) -> bool:
    required = (
        root / "sensevoice" / "model.int8.onnx",
        root / "sensevoice" / "tokens.txt",
        root / "speaker" / "pyannote-segmentation.int8.onnx",
        root / "speaker" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        root / "vad" / "silero_vad.onnx",
    )
    return all(path.exists() for path in required)


def resolve_model(model: str, model_dirs: list[Path] | tuple[Path, ...] | None = None) -> str:
    roots = list(model_dirs or [])
    roots += [
        default_model_cache_dir(),
        *legacy_model_cache_dirs(),
        Path.home() / ".cache" / "modelscope",
    ]
    for root in resource_roots():
        roots.extend([root, root / "packaging"])

    found = find_model_in_roots(model, roots)
    return str(found) if found is not None else model


def model_relative_path(model: str) -> Path | None:
    relative = MODEL_ALIASES.get(model)
    if relative is not None:
        return relative
    path = Path(model)
    if not path.is_absolute() and len(path.parts) == 2:
        return path
    return None


def find_model_in_roots(model: str, roots: list[Path] | tuple[Path, ...]) -> Path | None:
    path = Path(model)
    if path.exists():
        return path

    relative = model_relative_path(model)
    if relative is None:
        return None

    for root in roots:
        candidates = [
            root / "models" / relative,
            root / "hub" / "models" / relative,
            root / relative,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def list_cached_models(model_dir: Path | None) -> list[str]:
    root = model_dir or default_model_cache_dir()
    names = {DEFAULT_MODEL}
    if onnx_models_ready(onnx_model_cache_dir(model_dir)) or bundled_onnx_models_dir() is not None:
        names.add(DEFAULT_ONNX_MODEL)
    for base in (root / "models", root / "hub" / "models", root):
        if not base.exists():
            continue
        for org_dir in base.iterdir():
            if not org_dir.is_dir() or org_dir.name.startswith("."):
                continue
            for model_path in org_dir.iterdir():
                if model_path.is_dir() and not model_path.name.startswith("."):
                    names.add(f"{org_dir.name}/{model_path.name}")
    return sorted(names, key=lambda name: (name != DEFAULT_MODEL, name.lower()))


def model_is_cached(model: str, model_dir: Path | None) -> bool:
    if model == DEFAULT_ONNX_MODEL:
        return onnx_models_ready(onnx_model_cache_dir(model_dir)) or bundled_onnx_models_dir() is not None
    root = model_dir or default_model_cache_dir()
    roots = [root] if model_dir else [root, *legacy_model_cache_dirs()]
    return find_model_in_roots(model, roots) is not None


def configure_model_cache(model_dir: Path | None) -> None:
    target = model_dir or default_model_cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(target)


def download_default_models(model_dir: Path | None, progress_callback=None) -> list[Path]:
    target = model_dir or default_model_cache_dir()
    configure_model_cache(target)

    from modelscope import snapshot_download

    paths: list[Path] = []
    total = len(DEFAULT_SUPPORT_MODELS)
    for index, model_id in enumerate(DEFAULT_SUPPORT_MODELS, start=1):
        if progress_callback is not None:
            progress_callback(model_id, index, total)
        path = snapshot_download(model_id=model_id, cache_dir=str(target), local_files_only=False)
        paths.append(Path(path))
    return paths


def resolve_asset(relative_path: str) -> Path | None:
    relative = Path(relative_path)
    for root in resource_roots():
        candidates = [
            root / "localasr" / "assets" / relative,
            root / "src" / "localasr" / "assets" / relative,
            root / "assets" / relative,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None
