from __future__ import annotations

from pathlib import Path
import os
import platform
import plistlib
import shutil
import stat
import subprocess
import sys
import time
import tomllib

from ds_store import DSStore


ROOT = Path(__file__).resolve().parents[1]
DIST_NAME = os.environ.get("LOCALASR_DIST_NAME", "localasr")
APP_DISPLAY_NAME = os.environ.get("LOCALASR_APP_DISPLAY_NAME", "音频转文本")
DMG_BASENAME = os.environ.get("LOCALASR_DMG_BASENAME", APP_DISPLAY_NAME)
OFFLINE_MODELS = os.environ.get("LOCALASR_DMG_OFFLINE_MODELS") == "1"
DIST_DIR = ROOT / "dist" / DIST_NAME
RELEASE_DIR = ROOT / "dist" / "release"
STAGING_DIR = ROOT / "dist" / f"dmg-staging-{DIST_NAME}"
APP_NAME = f"{APP_DISPLAY_NAME}.app"
EXECUTABLE_NAME = "localasr"
ICON_SOURCE = ROOT / "packaging" / "assets" / "localasr.icns"
BACKGROUND_NAME = "install-background-v5.png"


def main() -> int:
    if platform.system().lower() != "darwin":
        raise SystemExit("DMG 只能在 macOS 上构建。")
    executable = DIST_DIR / EXECUTABLE_NAME
    internal = DIST_DIR / "_internal"
    if not executable.exists() or not internal.exists():
        raise SystemExit("请先运行 PyInstaller 并执行 packaging/finalize_dist.py。")

    version = read_version()
    arch = platform.machine() or "unknown"
    volume_name = dmg_volume_name(version)
    app_dir = RELEASE_DIR / APP_NAME
    dmg_path = RELEASE_DIR / f"{DMG_BASENAME}-{version}-macOS-{arch}.dmg"

    remove_path(app_dir)
    remove_path(STAGING_DIR)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    build_app_bundle(app_dir, executable, internal, version)
    build_dmg_staging(app_dir)
    remove_path(dmg_path)
    create_dmg(dmg_path, volume_name)
    print(f"DMG 构建完成：{dmg_path}")
    return 0


def read_version() -> str:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return "0.1.0"
    with pyproject.open("rb") as file:
        data = tomllib.load(file)
    return str(data.get("project", {}).get("version", "0.1.0"))


def dmg_volume_name(version: str) -> str:
    return os.environ.get("LOCALASR_DMG_VOLUME_NAME", f"安装{APP_DISPLAY_NAME}-{version}")


def build_app_bundle(app_dir: Path, executable: Path, internal: Path, version: str) -> None:
    macos_dir = app_dir / "Contents" / "MacOS"
    frameworks_dir = app_dir / "Contents" / "Frameworks"
    resources_dir = app_dir / "Contents" / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    frameworks_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    copy_file(executable, macos_dir / EXECUTABLE_NAME)
    app_executable = macos_dir / EXECUTABLE_NAME
    app_executable.chmod(app_executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    copy_internal_to_frameworks(internal, frameworks_dir)

    plist = {
        "CFBundleName": APP_DISPLAY_NAME,
        "CFBundleDisplayName": APP_DISPLAY_NAME,
        "CFBundleIdentifier": "cn.localasr.desktop",
        "CFBundleVersion": version,
        "CFBundleShortVersionString": version,
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": EXECUTABLE_NAME,
        "LSMinimumSystemVersion": "11.0",
    }
    if ICON_SOURCE.exists():
        copy_file(ICON_SOURCE, resources_dir / "localasr.icns")
        plist["CFBundleIconFile"] = "localasr"

    with (app_dir / "Contents" / "Info.plist").open("wb") as file:
        plistlib.dump(plist, file)


def build_dmg_staging(app_dir: Path) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    copytree_linked(app_dir, STAGING_DIR / APP_NAME)
    (STAGING_DIR / "Applications").symlink_to("/Applications")
    background_dir = STAGING_DIR / ".background"
    background_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "packaging" / "make_dmg_background.py"), str(background_dir / BACKGROUND_NAME)],
        check=True,
    )
    subprocess.run(["chflags", "hidden", str(background_dir)], check=False)


def copy_internal_to_frameworks(source: Path, target: Path) -> None:
    for item in source.iterdir():
        if item.name == ".DS_Store":
            continue
        destination = target / item.name
        if item.is_dir() and not item.is_symlink():
            copytree_linked(item, destination)
        else:
            copy_file(item, destination)


def create_dmg(dmg_path: Path, volume_name: str) -> None:
    temp_dmg = RELEASE_DIR / f"{dmg_path.stem}-rw.dmg"
    remove_path(temp_dmg)
    subprocess.run(
        [
            "hdiutil",
            "create",
            "-volname",
            volume_name,
            "-srcfolder",
            str(STAGING_DIR),
            "-ov",
            "-format",
            "UDRW",
            str(temp_dmg),
        ],
        check=True,
    )
    attached = False
    mount_point: Path | None = None
    try:
        attached_info = subprocess.run(
            ["hdiutil", "attach", str(temp_dmg), "-readwrite", "-noverify", "-noautoopen", "-plist"],
            check=True,
            capture_output=True,
        )
        mount_point = mounted_volume_path(attached_info.stdout)
        attached = True
        prepare_ds_store(mount_point)
        hide_support_items(mount_point)
        configure_finder_layout(mount_point, volume_name)
        cleanup_support_items(mount_point)
        subprocess.run(["sync"], check=True)
    finally:
        if attached and mount_point is not None:
            subprocess.run(["hdiutil", "detach", str(mount_point)], check=True)
    subprocess.run(["hdiutil", "convert", str(temp_dmg), "-format", "UDZO", "-o", str(dmg_path), "-ov"], check=True)
    subprocess.run(["hdiutil", "verify", str(dmg_path)], check=True)
    remove_path(temp_dmg)


def mounted_volume_path(plist_bytes: bytes) -> Path:
    data = plistlib.loads(plist_bytes)
    for entity in data.get("system-entities", []):
        mount_point = entity.get("mount-point")
        if mount_point:
            return Path(str(mount_point))
    raise SystemExit("挂载 DMG 失败：未找到 mount-point。")


def prepare_ds_store(mount_point: Path) -> None:
    ds_store = mount_point / ".DS_Store"
    with DSStore.open(str(ds_store), "w+"):
        pass


def hide_support_items(mount_point: Path) -> None:
    for name in (".background", ".fseventsd", ".DS_Store"):
        path = mount_point / name
        if path.exists():
            subprocess.run(["chflags", "hidden", str(path)], check=False)


def cleanup_support_items(mount_point: Path) -> None:
    fseventsd = mount_point / ".fseventsd"
    if fseventsd.exists():
        shutil.rmtree(fseventsd, ignore_errors=True)
    hide_support_items(mount_point)


def configure_finder_layout(mount_point: Path, volume_name: str) -> None:
    background_path = mount_point / ".background" / BACKGROUND_NAME
    script = f"""
set bgPic to POSIX file "{background_path}" as alias
tell application "Finder"
    tell disk "{volume_name}"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to {{120, 120, 840, 550}}
        set viewOptions to icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 136
        set background picture of viewOptions to bgPic
        set position of item "{APP_NAME}" of container window to {{160, 190}}
        set position of item "Applications" of container window to {{405, 190}}
        if exists item ".background" of container window then set position of item ".background" of container window to {{900, 390}}
        if exists item ".fseventsd" of container window then set position of item ".fseventsd" of container window to {{900, 320}}
        close
        open
        update without registering applications
        delay 1
    end tell
end tell
"""
    subprocess.run(["osascript", "-e", script], check=True)
    ds_store = mount_point / ".DS_Store"
    for _ in range(20):
        if ds_store.exists():
            return
        time.sleep(0.2)
    raise SystemExit("Finder 布局设置失败：未生成 .DS_Store。")


def copytree_linked(source: Path, target: Path) -> None:
    shutil.copytree(source, target, symlinks=True, copy_function=copy_file)


def copy_file(source: Path | str, target: Path | str) -> str:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)
    return str(target_path)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


if __name__ == "__main__":
    raise SystemExit(main())
