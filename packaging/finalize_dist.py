from __future__ import annotations

from pathlib import Path
import os
import plistlib
import platform
import shutil
import stat
import sysconfig
import site
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DIST_NAME = os.environ.get("LOCALASR_DIST_NAME", "localasr")
BUILD_NAME = os.environ.get("LOCALASR_BUILD_NAME", "localasr-gui")
APP_DISPLAY_NAME = os.environ.get("LOCALASR_APP_DISPLAY_NAME", "localasr")
DIST_DIR = ROOT / "dist" / DIST_NAME
APP_DIR = DIST_DIR / "localasr.app"
SYSTEM = platform.system().lower()
EXECUTABLE = DIST_DIR / ("localasr.exe" if SYSTEM == "windows" else "localasr")
ICON_SOURCE = ROOT / "packaging" / "assets" / "localasr.icns"
MODELS_SOURCE = ROOT / "packaging" / "models"
BASE_LIBRARY_SOURCE = ROOT / "build" / BUILD_NAME / "base_library.zip"
BUILD_EXECUTABLE = ROOT / "build" / BUILD_NAME / ("localasr.exe" if SYSTEM == "windows" else "localasr")


def main() -> int:
    if not EXECUTABLE.exists():
        if BUILD_EXECUTABLE.exists():
            DIST_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BUILD_EXECUTABLE, EXECUTABLE)
            EXECUTABLE.chmod(EXECUTABLE.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            raise SystemExit(f"找不到可执行文件：{EXECUTABLE}")
    internal_dir = DIST_DIR / "_internal"
    internal_dir.mkdir(parents=True, exist_ok=True)
    if BASE_LIBRARY_SOURCE.exists():
        shutil.copy2(BASE_LIBRARY_SOURCE, internal_dir / "base_library.zip")
    copy_python_shared_library(internal_dir)
    copy_python_lib_dynload(internal_dir)
    copy_setuptools_vendor_data(internal_dir)
    copy_package_data_file(internal_dir, "funasr", "version.txt")
    copy_pyside_top_level_runtime(internal_dir)
    copy_python_package_runtime(internal_dir, "shiboken6")
    prune_packaged_runtime(internal_dir)
    remove_broken_symlinks(internal_dir)
    ensure_ffmpeg_library_aliases(internal_dir)
    sync_models(internal_dir)

    if SYSTEM == "darwin":
        create_macos_app()

    remove_ds_store(DIST_DIR)
    (DIST_DIR / "使用说明.txt").write_text(
        f"{APP_DISPLAY_NAME} 使用说明\n"
        "\n"
        "1. 双击 localasr.app 启动桌面界面。\n"
        "2. 默认音频目录为用户目录；手工选择目录后，下次会自动记住。\n"
        "3. 模型默认放到系统应用数据目录：macOS 为 ~/Library/Application Support/localasr/models。\n"
        "4. 开启“说话人识别”后，TXT/SRT/JSON 会按句输出说话人编号。\n"
        "5. Agent/自动化后续优先通过本地 CLI 或 MCP stdio 接入。\n",
        encoding="utf-8",
    )
    return 0


def create_macos_app() -> None:
    macos_dir = APP_DIR / "Contents" / "MacOS"
    resources_dir = APP_DIR / "Contents" / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    launcher = macos_dir / "localasr"
    launcher.write_text(
        "#!/bin/sh\n"
        'APP_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"\n'
        'exec "$APP_DIR/localasr" "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    version = read_version()

    plist = {
        "CFBundleName": APP_DISPLAY_NAME,
        "CFBundleDisplayName": APP_DISPLAY_NAME,
        "CFBundleIdentifier": "cn.localasr.desktop",
        "CFBundleVersion": version,
        "CFBundleShortVersionString": version,
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": "localasr",
        "LSMinimumSystemVersion": "11.0",
    }
    if ICON_SOURCE.exists():
        shutil.copy2(ICON_SOURCE, resources_dir / "localasr.icns")
        plist["CFBundleIconFile"] = "localasr"

    with (APP_DIR / "Contents" / "Info.plist").open("wb") as file:
        plistlib.dump(plist, file)


def read_version() -> str:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return "0.1.0"
    with pyproject.open("rb") as file:
        data = tomllib.load(file)
    return str(data.get("project", {}).get("version", "0.1.0"))


def copy_python_lib_dynload(internal_dir: Path) -> None:
    source_text = sysconfig.get_config_var("DESTSHARED")
    if not source_text:
        return
    source = Path(source_text)
    if not source.exists():
        return
    target = internal_dir / f"python{sysconfig.get_python_version()}" / "lib-dynload"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)


def copy_python_shared_library(internal_dir: Path) -> None:
    framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
    library_name = sysconfig.get_config_var("LDLIBRARY")
    if not framework_prefix or not library_name:
        return
    source = Path(framework_prefix) / library_name
    if source.exists():
        target = internal_dir / "Python"
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(source, target)


def copy_setuptools_vendor_data(internal_dir: Path) -> None:
    for site_packages in site.getsitepackages():
        source = Path(site_packages) / "setuptools" / "_vendor" / "jaraco" / "text"
        if source.exists():
            target = internal_dir / "setuptools" / "_vendor" / "jaraco" / "text"
            shutil.copytree(source, target, dirs_exist_ok=True)
            return


def sync_models(internal_dir: Path) -> None:
    target = internal_dir / "models"
    if os.environ.get("LOCALASR_BUNDLE_MODELS") == "1" and MODELS_SOURCE.exists():
        shutil.copytree(MODELS_SOURCE, target, dirs_exist_ok=True)
    elif target.exists():
        shutil.rmtree(target)


def copy_package_data_file(internal_dir: Path, package_name: str, relative_file: str) -> None:
    for site_packages in site.getsitepackages():
        source = Path(site_packages) / package_name / relative_file
        if source.exists():
            target = internal_dir / package_name / relative_file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return


def copy_pyside_top_level_runtime(internal_dir: Path) -> None:
    keep_names = {
        "__init__.py",
        "_config.py",
        "QtCore.abi3.so",
        "QtGui.abi3.so",
        "QtWidgets.abi3.so",
    }
    for site_packages in site.getsitepackages():
        source = Path(site_packages) / "PySide6"
        target = internal_dir / "PySide6"
        if not source.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file() and (item.name in keep_names or item.name.startswith("libpyside6")):
                shutil.copy2(item, target / item.name)
        return


def copy_python_package_runtime(internal_dir: Path, package_name: str) -> None:
    for site_packages in site.getsitepackages():
        source = Path(site_packages) / package_name
        if source.exists():
            target = internal_dir / package_name
            if target.exists() or target.is_symlink():
                shutil.rmtree(target)
            shutil.copytree(source, target, symlinks=True)
            return


def prune_packaged_runtime(internal_dir: Path) -> None:
    prune_pyside_top_level_modules(internal_dir / "PySide6")
    prune_pyside_qt_runtime(internal_dir / "PySide6" / "Qt")
    for relative in (
        "torch/include",
        "torch/testing",
        "torch/fx/passes/tests",
        "torch/_numpy/testing",
        "shiboken6/include",
    ):
        remove_path(internal_dir / relative)
    for pattern in (
        "fastapi-*.dist-info",
        "starlette-*.dist-info",
        "uvicorn-*.dist-info",
    ):
        for path in internal_dir.glob(pattern):
            remove_path(path)
    for pattern in ("**/__pycache__",):
        for path in internal_dir.glob(pattern):
            remove_path(path)


def prune_pyside_top_level_modules(pyside_dir: Path) -> None:
    if not pyside_dir.exists():
        return
    keep = {
        "QtCore.abi3.so",
        "QtGui.abi3.so",
        "QtWidgets.abi3.so",
        "__init__.py",
        "_config.py",
    }
    for item in pyside_dir.iterdir():
        if item.is_file() and (
            item.name.startswith("Qt") and item.suffix == ".so" and item.name not in keep
            or item.suffix == ".pyi"
        ):
            remove_path(item)


def prune_pyside_qt_runtime(qt_dir: Path) -> None:
    if not qt_dir.exists():
        return
    for relative in (
        "translations",
        "lib/QtPdf.framework",
        "lib/QtQml.framework",
        "lib/QtQmlMeta.framework",
        "lib/QtQmlModels.framework",
        "lib/QtQmlWorkerScript.framework",
        "lib/QtQuick.framework",
        "lib/QtVirtualKeyboard.framework",
        "lib/QtVirtualKeyboardQml.framework",
        "plugins/networkinformation",
        "plugins/platforminputcontexts",
        "plugins/tls",
    ):
        remove_path(qt_dir / relative)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def remove_ds_store(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob(".DS_Store"):
        remove_path(path)


def remove_broken_symlinks(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_symlink() and not path.exists():
            path.unlink()


def ensure_ffmpeg_library_aliases(internal_dir: Path) -> None:
    """补齐 PyInstaller 偶尔漏建的 FFmpeg 主版本动态库链接。"""
    aliases = {
        "libavcodec.62.dylib": "libavcodec.62.*.dylib",
        "libavdevice.62.dylib": "libavdevice.62.*.dylib",
        "libavfilter.11.dylib": "libavfilter.11.*.dylib",
        "libavformat.62.dylib": "libavformat.62.*.dylib",
        "libavutil.60.dylib": "libavutil.60.*.dylib",
        "libswresample.6.dylib": "libswresample.6.*.dylib",
        "libswscale.9.dylib": "libswscale.9.*.dylib",
    }
    for alias_name, pattern in aliases.items():
        alias = internal_dir / alias_name
        if alias.exists():
            continue
        candidates = sorted(path for path in internal_dir.glob(pattern) if path.is_file())
        if candidates:
            alias.symlink_to(candidates[-1].name)


if __name__ == "__main__":
    raise SystemExit(main())
