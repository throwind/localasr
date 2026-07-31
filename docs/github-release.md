# GitHub 源码与 Release 发布

## 存储策略

- Git 仓库保存源码、配置、测试、打包脚本和文档。
- `packaging/onnx-runtime-models/` 中的 ONNX 模型通过 Git LFS 保存，保证离线包可复现。
- `packaging/onnx-models/` 是原始下载归档，不提交，避免模型重复占用空间。
- `dist/`、`build/`、虚拟环境和转写结果不提交。
- DMG 不进入 Git 历史，作为 GitHub Release asset 上传。

GitHub 普通 Git 对单文件限制为 100 MiB，内置 ONNX 模型必须使用 Git LFS。参考：[GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 首次创建仓库

```bash
brew install git-lfs gh

cd /Users/zhuhaoqi/project/localasr
git init -b main
git lfs install
git add .
git lfs ls-files
git commit -m "feat: 发布音频转文本 0.1.9"

git remote add origin https://github.com/throwind/localasr.git
git push -u origin main
```

执行 `git add .` 后必须确认 `git lfs ls-files` 能看到内置 ONNX 模型，再提交和推送。

## 发布 0.1.9 DMG

先为对应源码提交创建并推送标签：

```bash
git tag -a v0.1.9 -m "音频转文本 0.1.9"
git push origin v0.1.9
```

再创建 Release 并上传 DMG：

```bash
cp \
  "dist/release/音频转文本-ONNX-离线版-0.1.9-macOS-arm64.dmg" \
  "dist/release/localasr-onnx-offline-0.1.9-macOS-arm64.dmg"

gh release create v0.1.9 \
  "dist/release/localasr-onnx-offline-0.1.9-macOS-arm64.dmg#macOS arm64 离线安装包" \
  --verify-tag \
  --title "音频转文本 0.1.9" \
  --notes "修正转写阶段提示，增加 ONNX 模型并行预加载，并修复 FFmpeg 动态库打包。"
```

GitHub Release 单个 asset 上限为 2 GiB，当前 DMG 可以直接上传。参考：[GitHub Release 文档](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)和 [`gh release create`](https://cli.github.com/manual/gh_release_create)。

## 后续版本

1. 更新 `pyproject.toml` 和 `src/localasr/__init__.py` 的版本号。
2. 完成测试、应用烟测、真实音频工具校验和 DMG 挂载截图验收。
3. 提交源码并推送 `main`。
4. 创建并推送同版本标签，例如 `v0.2.0`。
5. 使用 `gh release create` 上传对应 DMG。
