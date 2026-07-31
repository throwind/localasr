# 使用方式：
#   pyinstaller packaging/localasr-gui.spec
#
# 可选：把 ffmpeg/ffprobe 放到以下目录后会被自动打入应用：
#   packaging/bin/darwin/ffmpeg
#   packaging/bin/darwin/ffprobe
#   packaging/bin/windows/ffmpeg.exe
#   packaging/bin/windows/ffprobe.exe

from pathlib import Path
import os
import platform
import subprocess

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path.cwd()
SYSTEM = platform.system().lower()
BIN_DIR = ROOT / "packaging" / "bin" / ("windows" if SYSTEM == "windows" else "darwin" if SYSTEM == "darwin" else "linux")
MODELS_DIR = ROOT / "packaging" / "models"
ASSETS_DIR = ROOT / "src" / "localasr" / "assets"
ICON_ICNS = ROOT / "packaging" / "assets" / "localasr.icns"


def collect_macho_rpath_dylibs(seed_binaries):
    if SYSTEM != "darwin":
        return []

    search_dirs = [
        Path("/opt/homebrew/lib"),
        Path("/usr/local/lib"),
        Path("/opt/homebrew/opt/ffmpeg/lib"),
        Path("/usr/local/opt/ffmpeg/lib"),
    ]
    collected = {}
    pending = [Path(item) for item in seed_binaries]
    seen = set()

    while pending:
        binary = pending.pop()
        key = str(binary.resolve()) if binary.exists() else str(binary)
        if key in seen or not binary.exists():
            continue
        seen.add(key)
        try:
            completed = subprocess.run(
                ["otool", "-L", str(binary)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue

        for line in completed.stdout.splitlines()[1:]:
            dep = line.strip().split(" ", 1)[0]
            if not dep or dep.startswith(("/System/Library/", "/usr/lib/")):
                continue
            dep_path = None
            if dep.startswith("@rpath/"):
                name = dep.removeprefix("@rpath/")
                candidates = [binary.parent / name, *[folder / name for folder in search_dirs]]
                dep_path = next((candidate for candidate in candidates if candidate.exists()), None)
            elif dep.startswith("@loader_path/"):
                dep_path = (binary.parent / dep.removeprefix("@loader_path/")).resolve()
            elif dep.startswith("@executable_path/"):
                dep_path = (BIN_DIR / dep.removeprefix("@executable_path/")).resolve()
            else:
                dep_path = Path(dep)

            if dep_path and dep_path.exists():
                resolved = dep_path.resolve()
                if resolved.name not in collected:
                    collected[resolved.name] = resolved
                    pending.append(resolved)

    return [(str(path), ".") for path in collected.values()]


binaries = []
ffmpeg_seed_binaries = []
for name in ("ffmpeg.exe", "ffprobe.exe") if SYSTEM == "windows" else ("ffmpeg", "ffprobe"):
    candidate = BIN_DIR / name
    if candidate.exists():
        binaries.append((str(candidate), "bin"))
        ffmpeg_seed_binaries.append(candidate)
binaries += collect_macho_rpath_dylibs(ffmpeg_seed_binaries)

datas = [(str(ROOT / "configs" / "default.toml"), "configs")]
if MODELS_DIR.exists() and os.environ.get("LOCALASR_BUNDLE_MODELS") == "1":
    datas.append((str(MODELS_DIR), "models"))
if ASSETS_DIR.exists():
    datas.append((str(ASSETS_DIR), "localasr/assets"))
hiddenimports = []

hiddenimports += ["PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"]

datas += collect_data_files(
    "funasr",
    includes=["version.txt"],
)
datas += copy_metadata("funasr")
datas += copy_metadata("modelscope")
datas += copy_metadata("torch")
datas += copy_metadata("torchaudio")
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("torchaudio")

hiddenimports += [
    "funasr",
    "funasr.auto.auto_frontend",
    "funasr.auto.auto_model",
    "funasr.auto.auto_tokenizer",
    "funasr.download.download_model_from_hub",
    "funasr.download.file",
    "funasr.download.name_maps_from_hub",
    "funasr.frontends.abs_frontend",
    "funasr.frontends.default",
    "funasr.frontends.fused",
    "funasr.frontends.utils.feature_transform",
    "funasr.frontends.utils.frontend",
    "funasr.frontends.utils.log_mel",
    "funasr.frontends.utils.stft",
    "funasr.frontends.wav_frontend",
    "funasr.frontends.whisper_frontend",
    "funasr.frontends.windowing",
    "funasr.register",
    "funasr.tokenizer.build_tokenizer",
    "funasr.tokenizer.sentencepiece_tokenizer",
    "funasr.tokenizer.token_id_converter",
    "funasr.tokenizer.whisper_tokenizer",
    "funasr.train_utils.load_pretrained_model",
    "funasr.train_utils.set_all_random_seed",
    "funasr.utils.export_utils",
    "funasr.utils.load_utils",
    "funasr.utils.misc",
    "funasr.utils.postprocess_utils",
    "funasr.utils.timestamp_tools",
    "funasr.utils.vad_utils",
    "modelscope",
    "modelscope.hub.snapshot_download",
    "modelscope.utils.constant",
    "torch",
    "torchaudio",
    "torchaudio.lib",
]

# 这些模型包通过 FunASR 注册表发现，不能只靠静态 import 推断。
for package in (
    "funasr.models.campplus",
    "funasr.models.ct_transformer",
    "funasr.models.fsmn_vad_streaming",
    "funasr.models.sense_voice",
    "funasr.models.sanm",
    "funasr.models.transformer",
):
    hiddenimports += collect_submodules(package)


a = Analysis(
    [str(ROOT / "packaging" / "gui_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "packaging" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "fastapi",
        "jupyter",
        "matplotlib",
        "notebook",
        "pydantic",
        "pytest",
        "starlette",
        "tkinter",
        "uvicorn",
        "uvloop",
        "watchfiles",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="localasr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(ICON_ICNS) if ICON_ICNS.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="localasr",
)
