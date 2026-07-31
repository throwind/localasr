# ASR 模型演进与评测计划

更新日期：2026-07-31。

## 目标

在保持本地离线、桌面端可分发和内存可控的前提下，提高中文会议、访谈和面试录音的转写准确率。模型不再只有一个固定选择，而是逐步形成“中文优先、多语种、高精度、方言专项”四类配置。

模型是否进入产品必须以真实业务音频的同机评测为准，不能直接比较不同项目各自公布的榜单。

## 第一批候选

| 候选 | 定位 | 模型体积 | 优点 | 主要风险 | 优先级 |
| --- | --- | ---: | --- | --- | --- |
| SenseVoice int8 | 当前多语种基线 | 约 226 MB | 快、成熟、已支持说话人链路 | 中文专有名词和复杂会议准确率仍有提升空间 | 基线 |
| Zipformer 中文 int8（2025-06-30） | 中文快速档 | 约 160 MB | 中文专项、比当前模型更小、sherpa-onnx 原生支持 | 输出格式、标点和长音频效果需要实测 | A |
| Paraformer 中文 int8 | 中文对照档 | 约 227 MB | 中文成熟方案、sherpa-onnx 原生支持 | 通用模型较旧；川渝版只适合对应方言 | A |
| Fun-ASR-Nano int8（2025-12-30） | 中文高精度档 | 约 948 MB | 覆盖多种中文方言和口音，sherpa-onnx 已支持 | 冷启动、内存和 CPU 速度明显高于小模型 | B |
| Qwen3-ASR 0.6B int8（2026-03-25） | 多语种高精度实验档 | 约 1.9 GB | 52 种语言和方言，具备较强上下文能力 | 模型较大，不适合作为默认下载 | C |
| FireRedASR2S | 中文精度参考 | 待核实 | 普通话、方言、中英混说能力强，包含 VAD/LID/标点 | 当前运行时与包体不适合直接并入桌面版 | 观察 |

资料来源：

- [sherpa-onnx 中文 Zipformer](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)
- [sherpa-onnx Paraformer](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html)
- [sherpa-onnx Fun-ASR-Nano](https://k2-fsa.github.io/sherpa/onnx/funasr-nano/pretrained.html)
- [sherpa-onnx Qwen3-ASR](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/export.html)
- [FunASR 当前模型与速度说明](https://github.com/modelscope/FunASR/blob/main/README_zh.md)
- [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)

## 中文基准集

从获得授权的真实音频中截取并人工校对 30 至 60 分钟，至少覆盖：

1. 普通话双人面试。
2. 三人以上会议，包含打断和重叠说话。
3. 手机或会议室远场录音。
4. 中英混说、数字、日期和英文缩写。
5. 人名、产品名、部门名等专有名词。
6. 至少一种常见方言或地方口音。

基准音频和人工标注默认不进入公开仓库，只保存评测脚本和脱敏统计结果。

## 统一指标

- 中文字符错误率 CER：同时计算去标点和保留标点两种结果。
- 专有名词准确率：单独统计人名、产品名、缩写和数字。
- 标点可读性：统计断句错误，并进行少量人工评分。
- 说话人结果：记录分离是否成功、误合并和误拆分，不把 ASR 与 diarization 混成一个指标。
- 速度：冷启动时间、模型加载时间、转写 RTF 和总耗时。
- 资源：峰值 RSS、任务结束后的残留内存、模型磁盘体积。
- 稳定性：长音频、静音、噪声、损坏片段和中断续跑。

## 产品策略

- 默认 DMG 继续只内置一个小模型；其他模型按需下载到 Application Support，应用升级不重复下载。
- 第一阶段增加“中文优先”实验模型，不立即替换 SenseVoice。
- 设置页最终只暴露易懂的策略：`自动`、`中文优先`、`多语种`、`高精度`；具体模型版本放在高级信息中。
- 已明确语言时直接选择对应模型；未知语言先使用轻量语言识别或保持 SenseVoice 自动判断。
- 方言模型只在用户明确选择方言时启用，避免自动误判导致精度下降。
- 模型缓存键必须包含模型 ID、版本、量化类型和转写参数，切换模型后不能复用旧 ASR 结果。

## 实施顺序

1. 建立可重复运行的评测命令和结果 JSON。
2. 固化当前 SenseVoice 基线数据。
3. 新增模型描述与适配器接口，先接入中文 Zipformer int8。
4. 对比 Paraformer；只保留真实业务音频中表现更好的中文小模型。
5. 评估 Fun-ASR-Nano int8 作为可选“高精度”下载项。
6. 持续关注 Qwen3-ASR、FireRedASR 和 sherpa-onnx 新模型，但不因发布新模型就直接进入默认包。

## 进入默认模型的门槛

- 中文 CER 相比当前基线有稳定、可复现的改善。
- 面试与会议两个核心场景均不能明显退化。
- macOS Apple Silicon 上长音频处理速度仍快于实时。
- 峰值内存和任务结束释放满足现有稳定性要求。
- 许可证允许项目当前的分发方式。
- 模型来源、版本、SHA-256 和下载地址可长期追踪。
