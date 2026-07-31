# 音频转文本

基于 PySide6、SenseVoice 和 sherpa-onnx 的本地音频转写工具。选择一个音频文件，点击开始转写，即可在原目录得到纯文本结果。

## 下载

[下载 macOS arm64 离线版 0.1.9](https://github.com/throwind/localasr/releases/download/v0.1.9/localasr-onnx-offline-0.1.9-macOS-arm64.dmg)

- 支持 macOS 11 及以上的 Apple Silicon Mac。
- 安装包未签名。首次打开如被 macOS 阻止，请在 Finder 中右键应用并选择“打开”。
- 模型、FFmpeg 和 Python 运行时均已包含，无网络环境也可以使用。

## 功能

- 单文件操作流程：选择音频、开始转写、打开生成文本。
- 支持 AAC、AIFF、ALAC、FLAC、M4A、MKA、MKV、MOV、MP3、MP4、OGG、Opus、WAV、WebM 和 WMA。
- 使用 SenseVoice ONNX int8 模型进行中、英、粤、日、韩语音识别。
- 使用 Silero VAD、Pyannote segmentation 和 3D-Speaker 进行多人说话人区分。
- 长音频按静音边界切分，并保留片段缓存，支持中断后继续。
- 模型加载与音频预处理并行执行，任务结束后释放推理进程和内存。
- 转写结果默认保存为源音频同名 `.txt` 文件。
- 提供本地 CLI，便于脚本和 Agent 调用，不需要启动 HTTP 服务。

## 使用源码运行

需要 Python 3.11 或 3.12，以及可用的 `ffmpeg`：

```bash
brew install ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[gui,onnx,packaging]'
```

启动 GUI：

```bash
LOCALASR_ENGINE=sherpa-onnx localasr-gui
```

批量 CLI 示例：

```bash
localasr \
  --engine sherpa-onnx \
  --input-dir ./audios \
  --output-dir ./transcripts \
  --formats txt \
  --speaker-diarization
```

GUI 面向单文件转写；CLI 仍支持目录批量处理和 TXT、JSON、SRT 输出。

## 模型目录

离线版首次启动时，会把随包模型复制到：

```text
~/Library/Application Support/localasr/models/onnx/sherpa-sensevoice-2025-09-09
```

升级应用不会覆盖已经存在的模型。可以通过 `LOCALASR_MODEL_DIR` 指定其他模型目录，通过 `LOCALASR_DATA_DIR` 覆盖整个应用数据目录。

## 开发验证

```bash
.venv/bin/python -m compileall -q src tests packaging
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
QT_QPA_PLATFORM=offscreen LOCALASR_ENGINE=sherpa-onnx PYTHONPATH=src \
  .venv/bin/localasr-gui --smoke-test
```

## macOS 离线版打包

完整构建流程见 [打包瘦身方案](docs/packaging-slim-plan.md)。主要命令：

```bash
.venv/bin/pyinstaller --noconfirm --clean packaging/localasr-gui-onnx.spec

LOCALASR_DIST_NAME=localasr-onnx \
LOCALASR_BUILD_NAME=localasr-gui-onnx \
LOCALASR_APP_DISPLAY_NAME=音频转文本-ONNX \
.venv/bin/python packaging/finalize_dist.py

LOCALASR_DIST_NAME=localasr-onnx \
LOCALASR_APP_DISPLAY_NAME=音频转文本-ONNX \
LOCALASR_DMG_BASENAME=音频转文本-ONNX-离线版 \
LOCALASR_DMG_OFFLINE_MODELS=1 \
.venv/bin/python packaging/build_dmg.py
```

## 项目结构

```text
src/localasr/                     核心转写、任务服务、GUI 和 CLI
packaging/onnx-runtime-models/    Git LFS 管理的离线模型
packaging/                        PyInstaller 与 DMG 构建脚本
tests/                            单元测试和测试音频
docs/                             开发、界面和发布经验
```

GitHub 源码与安装包发布流程见 [GitHub Release 指南](docs/github-release.md)。

## 致谢

- [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [PySide6](https://doc.qt.io/qtforpython-6/)
