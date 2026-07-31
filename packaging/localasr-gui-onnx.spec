# 使用方式：
#   pyinstaller --clean packaging/localasr-gui-onnx.spec
#
# 这个 spec 构建 ONNX 离线版，不打入 FunASR/PyTorch。

from pathlib import Path
import platform
import site
import subprocess

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata


ROOT = Path.cwd()
SYSTEM = platform.system().lower()
BIN_DIR = ROOT / "packaging" / "bin" / ("windows" if SYSTEM == "windows" else "darwin" if SYSTEM == "darwin" else "linux")
ONNX_MODELS_DIR = ROOT / "packaging" / "onnx-runtime-models"
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

            # Homebrew 升级后，仓库内 ffmpeg 可能仍记录旧 Cellar 版本路径。
            # 同一主版本 ABI 的动态库可从当前 opt 路径按文件名重新定位。
            if (dep_path is None or not dep_path.exists()) and dep.startswith("/"):
                dep_name = Path(dep).name
                dep_path = next(
                    (folder / dep_name for folder in search_dirs if (folder / dep_name).exists()),
                    None,
                )

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
if ASSETS_DIR.exists():
    datas.append((str(ASSETS_DIR), "localasr/assets"))
if ONNX_MODELS_DIR.exists():
    datas.append((str(ONNX_MODELS_DIR), "onnx-runtime-models"))

datas += copy_metadata("sherpa-onnx")
datas += copy_metadata("sherpa-onnx-core")
datas += copy_metadata("soundfile")
binaries += collect_dynamic_libs("sherpa_onnx")

for site_packages in site.getsitepackages():
    soundfile_data = Path(site_packages) / "_soundfile_data"
    if soundfile_data.exists():
        datas.append((str(soundfile_data), "_soundfile_data"))
        break

hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "_soundfile",
    "_soundfile_data",
    "cffi",
    "localasr.onnx_transcriber",
    "numpy",
    "sherpa_onnx",
    "soundfile",
]

a = Analysis(
    [str(ROOT / "packaging" / "gui_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "packaging" / "hooks")],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "runtime_onnx.py")],
    excludes=[
        "IPython",
        "fastapi",
        "funasr",
        "jupyter",
        "librosa",
        "llvmlite",
        "matplotlib",
        "modelscope",
        "notebook",
        "numba",
        "pandas",
        "pydantic",
        "pytest",
        "scipy",
        "sentencepiece",
        "sklearn",
        "starlette",
        "tensorboard",
        "tkinter",
        "torch",
        "torchaudio",
        "transformers",
        "umap",
        "uvicorn",
        "uvloop",
        "watchfiles",
        "yaml",
        "charset_normalizer",
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
    name="localasr-onnx",
)
