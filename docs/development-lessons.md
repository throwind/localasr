# 开发经验沉淀

这份文档记录 `localasr` 截止目前的产品、工程、打包经验。后续继续开发桌面 app、转写引擎、分发包时，优先按这些约定执行，减少重复试错。

## 产品与交互

- 工具型 app 第一屏直接放核心工作流，不做落地页。当前主流程是：打开音频目录、勾选音频、开始转写、看进度和结果。
- 主界面保持紧凑。低频设置放到设置页，设置页按 tab 和分组组织，避免一长串表单。
- 用户路径要符合直觉：音频目录默认用户目录，手动选择后记住；转写文本默认和音频在同一目录。
- 表格必须随窗口缩放，不能溢出分组框。文件结果用子行挂在音频下面，方便直接判断是否已转写。
- 中文环境下按钮文案统一中文，例如“确认”“取消”“下载默认模型”。
- GUI 只负责操作和展示，业务逻辑放在 `pipeline` / `service`，避免以后 CLI、MCP、Agent 接入时重复实现。

## 转写链路

- 长音频处理要分阶段落盘：先转码切片，再逐片转写，再持续写出 TXT/JSON/SRT。这样中断后能复用缓存，也能尽早看到部分结果。
- 外层固定切片不能硬切语音，应通过静音边界和重叠保护减少截断风险。
- TXT 面向阅读，连续相同说话人要合并成自然段；JSON/SRT 面向结构化和时间轴，可以保留更细粒度片段。
- 纯标点、空片段不要写入最终 TXT/SRT。
- 说话人识别失败后应尽快降级普通转写，尤其是长音频。不要让后续每个切片重复慢失败。
- 进度日志尽量在 `pipeline` 里产生，而不是只在 GUI 里拼。这样 GUI、CLI、未来 MCP 都能共享一致的状态，例如每个音频完成时显示转写耗时。

## 模型策略

- PyTorch/FunASR 路线功能完整，但运行时依赖大；即使不含模型，app 也会比较重。
- ONNX/sherpa-onnx 路线能显著缩小包体，并支持离线内置模型，适合内部少量分发和无网络使用。
- ONNX 说话人识别比普通 ASR 明显更吃内存。实测 92 秒测试音频：普通 ASR 峰值约 1.3GB，开启说话人识别约 4-5GB。长音频必须限制 diarization 内部窗口，避免整段 600 秒直接进入底层说话人分离。
- 模型默认缓存目录使用系统应用数据目录：
  - macOS：`~/Library/Application Support/localasr/models`
  - Windows：`%APPDATA%\localasr\models`
  - Linux：`~/.local/share/localasr/models`
- 离线 ONNX 版的内置模型放在 app 包内，首次运行复制到应用数据目录，后续 app 更新不应重复动用户缓存里的模型。
- 设置页里模型相关动作要靠前，首次使用应能很快找到下载或安装默认模型的入口。

## 打包与发布

- 每次生成新的可分发包前，必须先增加版本号。至少更新 `pyproject.toml` 的 `project.version`，再重新打包。除非用户明确说“这次先算了”。
- PyInstaller 走 `onedir`，比单文件模式更适合 PySide6、机器学习依赖和 ffmpeg。
- macOS 分发优先做 DMG：把自包含 `.app`、`Applications` 快捷方式和使用说明放进去。
- 打包后必须跑：
  - 源码编译：`.venv/bin/python -m compileall -q src tests packaging`
  - 单测：`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`
  - dist 可执行 smoke test：`QT_QPA_PLATFORM=offscreen dist/localasr-onnx/localasr --smoke-test`
  - release app smoke test：`QT_QPA_PLATFORM=offscreen 'dist/release/音频转文本-ONNX.app/Contents/MacOS/localasr' --smoke-test`
  - DMG 校验：`hdiutil verify ...`
- ONNX 版打包时使用独立 spec：`packaging/localasr-gui-onnx.spec`，通过 runtime hook 设置 `LOCALASR_ENGINE=sherpa-onnx`，避免把 FunASR/PyTorch 引进包。
- ONNX 版 runtime hook 还应设置 `LOCALASR_MAX_DIARIZATION_SECONDS=30` 和常见计算库线程数为 `1`，避免长音频说话人识别造成内存和线程过度膨胀。
- `sherpa_onnx/lib` 和 `_soundfile_data` 必须显式收集，否则冻结环境容易缺动态库。
- PySide6 可以修剪当前 QWidget app 不用的 QML/Quick/PDF/翻译资源，但修剪后必须清理断开的 symlink，否则 DMG 复制 app bundle 时会失败。
- 压缩 DMG 体积不能等同于未压缩 app 体积。当前 ONNX 离线版实测大约：未压缩 app `484M`，DMG `277M`。

## 依赖与边界

- 本地 GUI 不需要常驻 HTTP API。Agent 集成优先通过 CLI，后续如果做 MCP，建议新增 stdio server，而不是把 FastAPI/Uvicorn 重新打进桌面包。
- 本机容器工具是 podman，不是 docker。但当前桌面分发方案不走本地容器。
- 依赖下载慢时可以临时用国内 PyPI 镜像加速本机打包环境安装，但不要把镜像源写死到产品代码。

## 文档习惯

- UI 细节沉淀到 `docs/app-ui-guidelines.md`。
- 包体、模型、发布策略沉淀到 `docs/packaging-slim-plan.md`。
- 跨模块经验和发布检查沉淀到本文档。
- 重要结论同步写入 `../memory/YYYY-MM-DD.md` 或 `../MEMORY.md`，避免后续会话丢失上下文。
