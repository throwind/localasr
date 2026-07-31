# 打包瘦身方案

目标：公开分发包开箱即用，同时尽量小。大型模型和可延迟初始化的资源尽量不进入安装包；确实需要离线使用时，优先走 ONNX 小模型路线。

## 发布前硬性检查

- 每次生成新的可分发包前，必须先增加版本号。至少更新 `pyproject.toml` 的 `project.version`，再重新打包。
- 如果只是本机临时验证，可以不改版本号；只要要给别人复制、发 DMG、发 release，就先升版本。
- 打包完成后必须跑产物自己的 smoke test，不只跑源码环境。
- 生成 DMG 后必须执行 `hdiutil verify`，确认镜像有效。

## 当前体积结论

当前 PyTorch/FunASR 路线下，即使不打包模型，`dist/localasr` 仍约 843MB，`_internal` 约 764MB。主要体积来自运行时依赖：

- `torch`：约 309MB。
- `PySide6`：约 96MB。
- `llvmlite`：约 110MB。
- `transformers`、`scipy`、`jieba`、`sklearn` 等：几十 MB 级。

模型如果打进包会更大。当前本机模型体积约：

- `iic/SenseVoiceSmall`：约 900MB。
- `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`：约 4MB。
- `iic/speech_campplus_sv_zh-cn_16k-common`：约 28MB。
- `iic/punc_ct-transformer_cn-en-common-vocab471067-large`：约 1.1GB。

所以公开分发必须默认走“瘦包 + 首次下载模型”。

## 应用数据目录

默认模型目录改为系统应用数据目录：

- macOS：`~/Library/Application Support/localasr/models`
- Windows：`%APPDATA%\localasr\models`
- Linux：`~/.local/share/localasr/models`

环境变量：

- `LOCALASR_DATA_DIR`：覆盖整个 app 数据目录。
- `LOCALASR_MODEL_DIR`：只覆盖模型目录。

兼容旧目录：程序查找模型时仍会回退到 `~/localasr-models`，避免老用户已有模型完全失效。

## 当前 PyTorch 瘦包策略

默认构建：

```bash
pyinstaller packaging/localasr-gui.spec
python packaging/finalize_dist.py
```

默认行为：

- 不打包 `packaging/models`。
- 包含 GUI、核心转写逻辑、FunASR/PyTorch 运行时、PySide6、ffmpeg/ffprobe。
- PyInstaller 使用项目内定制 hook，避免默认 Torch hook 和 `collect_all` 递归收完整个 Torch/FunASR/ModelScope 世界。
- 首次启动检查默认模型；如果缺失，提示用户进入模型设置下载。
- HTTP API 入口已移除，不再引入 FastAPI/Uvicorn；Agent/自动化优先走 CLI，后续再做 MCP stdio server。

可选环境变量：

```bash
LOCALASR_BUNDLE_MODELS=1 pyinstaller packaging/localasr-gui.spec
```

- `LOCALASR_BUNDLE_MODELS=1`：仅内部离线分发使用，会把 `packaging/models` 打进包。

Agent/自动化优先通过本地 CLI 或后续 MCP stdio server 接入，避免为 GUI 包引入 FastAPI/Uvicorn 等常驻服务依赖。

## macOS DMG

在 `dist/localasr` 构建并收尾后，可以生成 DMG：

```bash
python packaging/build_dmg.py
```

产物：

```text
dist/release/音频转文本-0.1.1-macOS-arm64.dmg
```

DMG 内部不是旧的 `onedir` 目录，而是重组后的自包含 `音频转文本.app`：

- `Contents/MacOS/localasr`：PyInstaller 可执行文件。
- `Contents/Frameworks/`：原 `_internal` 运行时内容。
- `Contents/Resources/localasr.icns`：应用图标。

这样用户可以把 `音频转文本.app` 直接拖到 Applications。当前 macOS arm64 实测：

- 自包含 app：约 843MB。
- 压缩 DMG：约 358MB。

注意：当前 DMG 未签名、未 notarize。外部分发时，macOS 可能提示无法确认开发者，需要右键打开；正式发布应补 Developer ID 签名和 notarization。

## ONNX 离线版打包

当前已验证 ONNX/sherpa-onnx 路线，可以在不依赖网络下载模型的情况下开箱即用。模型资源由 `packaging/prepare_onnx_models.py` 整理到：

```text
packaging/onnx-runtime-models/
```

运行时首次复制到：

```text
~/Library/Application Support/localasr/models/onnx/sherpa-sensevoice-2025-09-09
```

构建命令：

```bash
.venv/bin/python -m compileall -q src tests packaging
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
QT_QPA_PLATFORM=offscreen LOCALASR_ENGINE=sherpa-onnx PYTHONPATH=src .venv/bin/python -m localasr.gui --smoke-test

.venv/bin/pyinstaller --noconfirm --clean packaging/localasr-gui-onnx.spec
LOCALASR_DIST_NAME=localasr-onnx \
LOCALASR_BUILD_NAME=localasr-gui-onnx \
LOCALASR_APP_DISPLAY_NAME=音频转文本-ONNX \
.venv/bin/python packaging/finalize_dist.py

QT_QPA_PLATFORM=offscreen dist/localasr-onnx/localasr --smoke-test

LOCALASR_DIST_NAME=localasr-onnx \
LOCALASR_APP_DISPLAY_NAME=音频转文本-ONNX \
LOCALASR_DMG_BASENAME=音频转文本-ONNX-离线版 \
LOCALASR_DMG_OFFLINE_MODELS=1 \
.venv/bin/python packaging/build_dmg.py

QT_QPA_PLATFORM=offscreen 'dist/release/音频转文本-ONNX.app/Contents/MacOS/localasr' --smoke-test
```

当前 macOS arm64 实测：

- `dist/localasr-onnx`：约 484MB。
- `dist/release/音频转文本-ONNX.app`：约 484MB。
- `dist/release/音频转文本-ONNX-离线版-0.1.1-macOS-arm64.dmg`：约 277MB。

ONNX 版注意事项：

- 使用独立 spec：`packaging/localasr-gui-onnx.spec`，不要复用 PyTorch spec。
- runtime hook 设置 `LOCALASR_ENGINE=sherpa-onnx`，让默认 GUI 直接走 ONNX。
- `sherpa_onnx/lib` 和 `_soundfile_data` 要显式收集。
- 可以修剪 PySide6 中当前 QWidget app 不用的 QML/Quick/PDF/翻译资源；修剪后必须清理断开的 symlink，否则 DMG 构建会因为找不到 Qt framework 目标而失败。

DMG 安装界面参考 `localdict` 的经验：

- 默认卷名使用 `安装{APP_DISPLAY_NAME}`，避开 Finder 对旧同名卷的布局/背景缓存。
- staging 中保留 `.background/install-background-v2.png`，并隐藏 `.background` 和 `.DS_Store`。
- 先创建可写 DMG，挂载后通过 Finder 写入图标大小、背景图和图标位置，再转换为只读压缩 DMG。
- 构建后挂载检查时，普通可见项应只有 app 和 `Applications` 链接。

## 模型用途

当前默认下载的模型：

- `iic/SenseVoiceSmall`：主 ASR。
- `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`：VAD 分段。
- `iic/speech_campplus_sv_zh-cn_16k-common`：说话人识别。
- `iic/punc_ct-transformer_cn-en-common-vocab471067-large`：FunASR 说话人链路要求的标点模型。

实测去掉 `punc_model` 后，说话人识别仍能输出 speaker，但 FunASR 会打印 `Missing punc_model, which is required by spk_model.`。在未替换底层链路前，不建议直接删掉默认下载中的 punc 模型，否则首次体验会出现错误日志。

## 下一阶段：ONNX 路线

如果目标是把分发包从数百 MB 进一步压缩，关键不是继续挤 PyInstaller，而是替换推理后端：

- 官方 ModelScope 已有 `iic/SenseVoiceSmall-onnx`。
- FunASR 文档包含 ONNX 导出和 ONNX 测试入口。
- ONNX Runtime / sherpa-onnx 路线有机会移除 `torch`、`torchaudio`、部分 `transformers`/`modelscope` 运行时依赖。

建议下一阶段单独开分支验证：

1. 增加 `engine` 配置：`funasr-pytorch` / `sensevoice-onnx`。
2. 新增 ONNX 引擎适配器，先只支持普通 ASR + VAD。
3. 保留 PyTorch 引擎作为功能完整路径，尤其是说话人识别。
4. 对比包体、速度、准确率、说话人功能缺口。
5. ONNX 路线稳定后，再调整默认引擎和默认下载模型。

短期不要把 ONNX 和当前 GUI/打包改动混在一起，否则问题定位会很难。
