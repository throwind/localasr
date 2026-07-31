from __future__ import annotations

from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "packaging" / "onnx-models"
TARGET_DIR = ROOT / "packaging" / "onnx-runtime-models"

ASR_ARCHIVE = SOURCE_DIR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2"
SEGMENTATION_ARCHIVE = SOURCE_DIR / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_MODEL = SOURCE_DIR / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
VAD_MODEL = SOURCE_DIR / "silero_vad.onnx"


def main() -> int:
    validate_sources()
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    (TARGET_DIR / "sensevoice").mkdir(parents=True)
    (TARGET_DIR / "speaker").mkdir(parents=True)
    (TARGET_DIR / "vad").mkdir(parents=True)

    extract_member(ASR_ARCHIVE, "model.int8.onnx", TARGET_DIR / "sensevoice" / "model.int8.onnx")
    extract_member(ASR_ARCHIVE, "tokens.txt", TARGET_DIR / "sensevoice" / "tokens.txt")
    extract_member(ASR_ARCHIVE, "README.md", TARGET_DIR / "sensevoice" / "README.md")

    extract_member(SEGMENTATION_ARCHIVE, "model.int8.onnx", TARGET_DIR / "speaker" / "pyannote-segmentation.int8.onnx")
    extract_member(SEGMENTATION_ARCHIVE, "LICENSE", TARGET_DIR / "speaker" / "pyannote-segmentation.LICENSE")
    shutil.copy2(EMBEDDING_MODEL, TARGET_DIR / "speaker" / EMBEDDING_MODEL.name)
    shutil.copy2(VAD_MODEL, TARGET_DIR / "vad" / VAD_MODEL.name)

    print(f"ONNX 模型资源已准备：{TARGET_DIR}")
    return 0


def validate_sources() -> None:
    missing = [path for path in (ASR_ARCHIVE, SEGMENTATION_ARCHIVE, EMBEDDING_MODEL, VAD_MODEL) if not path.exists()]
    if missing:
        lines = "\n".join(str(path) for path in missing)
        raise SystemExit(f"缺少 ONNX 模型文件：\n{lines}")


def extract_member(archive: Path, member_name: str, target: Path) -> None:
    with tarfile.open(archive, "r:bz2") as tar:
        member = next((item for item in tar.getmembers() if Path(item.name).name == member_name), None)
        if member is None:
            raise SystemExit(f"{archive} 中找不到 {member_name}")
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise SystemExit(f"{archive} 中 {member_name} 不是普通文件")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            shutil.copyfileobj(file_obj, output)


if __name__ == "__main__":
    raise SystemExit(main())
