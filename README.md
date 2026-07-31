# localasr

`localasr` 是一个基于 SenseVoice/FunASR 的本地音频转写软件。它面向两类使用方式：

- 普通用户：打开桌面 GUI，选择音频目录，点击开始转写；转写结果默认保存在音频同目录。
- Agent/自动化工具：调用本地 CLI 执行批量转写，读取输出路径和本地文件。

核心转写逻辑和界面/协议入口是解耦的：`pipeline` 负责音频扫描、转码切片、模型推理和输出保存；`service` 负责异步任务和状态管理；GUI 和 CLI 都只是服务层的适配器。后续如果要增加 MCP stdio server，只需要再加一个协议适配器，不需要改 ASR 核心。

## 架构

```text
PySide6 GUI ─┐
CLI         ├─ JobManager ─ AppConfig ─ pipeline ─ ffmpeg/ffprobe ─ SenseVoice/FunASR
```

已实现入口：

- `localasr-gui`：桌面 GUI。
- `localasr`：命令行批量转写。

## 功能

- 指定输入目录，递归或非递归读取常见音频/视频封装文件。
- 自动调用 `ffprobe` 获取时长，调用 `ffmpeg` 解码成 16 kHz 单声道 wav 并切片。
- 使用 `iic/SenseVoiceSmall` 转写，默认语言自动识别。
- 输出 `.txt`、`.json`、`.srt`。
- 可选说话人识别：使用 SenseVoice + VAD + CAM++ + 标点模型，按句输出 `说话人 0/1/...`。
- 已存在结果默认跳过，适合中断后重跑。
- GUI 可查看进度和日志。
- 打包后优先使用随包携带的 `bin/ffmpeg`、`bin/ffprobe`；找不到时再使用系统 PATH。

## 开发运行

建议使用 Python 3.11 或 3.12。机器学习依赖通常不会第一时间支持最新 Python。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[app]'
```

说话人识别需要 FunASR 源码版；`pyproject.toml` 已固定到当前验证过的官方提交。

本机开发时还需要安装 `ffmpeg`：

```bash
brew install ffmpeg
```

启动桌面界面：

```bash
localasr-gui
```

命令行转写：

```bash
localasr --input-dir ./audios --output-dir ./transcripts --language auto --device cpu
```

启用说话人识别：

```bash
localasr --input-dir ./audios --output-dir ./transcripts --speaker-diarization
```

说明：SenseVoice 会输出 `sentence_info[*].spk` 这样的说话人编号。它能区分“这是不同的人在说话”，但不知道谁是面试官、谁是被面试者；实际面试音频里可以按首轮发言或 JSON 中的说话人编号，把 `说话人 0/1` 人工重命名为“面试官/被面试者”。如果上游某个片段的说话人分离因时间戳异常失败，localasr 会自动降级为普通转写，并在日志和 JSON 的 `warning` 字段记录原因。

## Agent 集成建议

当前推荐用 CLI 对接 Agent：

1. Agent 调用 `localasr --input-dir ... --output-dir ... --formats txt,json`。
2. 进程退出码为 `0` 表示任务完成。
3. 完成后读取输出目录里的 `.txt`、`.json`、`.srt` 文件。

后续要支持 MCP 时，建议新增 `localasr-mcp`：

- `transcribe_directory`：提交目录转写任务。
- `transcribe_file`：提交单个文件转写任务。
- `read_transcript`：读取已生成文本。

这些 MCP tools 可以直接调用 `JobManager` 或 CLI，不经过 GUI，也不需要常驻 HTTP 服务。

## 打包桌面应用

推荐使用 PyInstaller 的 `onedir` 模式。相比单文件 exe，`onedir` 对 `torch`、`funasr`、`PySide6` 和 `ffmpeg` 这类重依赖更稳。

当前 macOS arm64 实测瘦包约 `843MB`，不包含模型文件。打包脚本使用项目内 PyInstaller hook，避免全量递归收集 Torch/FunASR/ModelScope 的无关模块。

默认构建的是瘦包：不内置模型，也不内置 HTTP API 服务依赖。模型在首次运行时下载到系统应用数据目录：

- macOS：`~/Library/Application Support/localasr/models`
- Windows：`%APPDATA%\\localasr\\models`
- Linux：`~/.local/share/localasr/models`

如果需要临时覆盖模型目录，可以设置环境变量 `LOCALASR_MODEL_DIR`；如果需要覆盖整个应用数据目录，可以设置 `LOCALASR_DATA_DIR`。

目录约定：

```text
packaging/
  bin/
    darwin/
      ffmpeg
      ffprobe
    windows/
      ffmpeg.exe
      ffprobe.exe
```

macOS 构建：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[app]'
pyinstaller packaging/localasr-gui.spec
python packaging/finalize_dist.py
python packaging/build_dmg.py
```

Windows 构建需要在 Windows 上执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[app]"
pyinstaller packaging/localasr-gui.spec
python packaging\finalize_dist.py
```

构建产物在 `dist/localasr/`。这个目录可以整体压缩分发。

macOS DMG 产物在 `dist/release/`，形如：

```text
dist/release/音频转文本-0.1.1-macOS-arm64.dmg
```

DMG 内包含一个自包含的 `音频转文本.app` 和 `Applications` 快捷方式，用户可以按常规 macOS 软件方式拖入 Applications。当前 macOS arm64 DMG 实测约 `358MB`。

可选环境变量：

- `LOCALASR_BUNDLE_MODELS=1`：把 `packaging/models` 里的模型打进包，只建议内部离线分发使用。

## 模型分发策略

有两种方式：

- 轻量包：应用包含程序、GUI、Python 依赖和 `ffmpeg`；首次运行时下载模型到应用数据目录。
- 完整包：提前下载模型，把模型目录随应用一起分发；构建时设置 `LOCALASR_BUNDLE_MODELS=1`。

当前 PyTorch/FunASR 路线下，默认下载的模型包括：

- `iic/SenseVoiceSmall`：主 ASR 模型。
- `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`：VAD 分段模型。
- `iic/speech_campplus_sv_zh-cn_16k-common`：说话人识别模型。
- `iic/punc_ct-transformer_cn-en-common-vocab471067-large`：FunASR 说话人链路要求的标点模型。

公开分发建议先用轻量包；内部离线环境再考虑完整包。

## 配置文件

可以继续使用 `configs/default.toml`：

```bash
localasr --config configs/default.toml
```

关键配置：

- `input.dir`：音频目录。
- `output.dir`：转写结果目录。
- `audio.chunk_seconds`：外层切片长度，默认 600 秒。
- `sensevoice.language`：`auto`、`zh`、`en`、`yue`、`ja`、`ko`、`nospeech`。
- `sensevoice.device`：`cpu` 或 `cuda:0`。

## 开发验证

不加载模型的基础测试：

```bash
python -m compileall src tests
PYTHONPATH=src python -m unittest discover -s tests
```

真实转写需要可用的 `ffmpeg/ffprobe`，首次运行会下载模型。
