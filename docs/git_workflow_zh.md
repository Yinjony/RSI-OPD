# 用 Git 同步 minWM 代码

Git 管理 Python/CUDA 源码、YAML 配方、Shell 脚本、测试、文档和文档图片。
当前 Guider2 回放实验的迁移脚本也纳入版本管理，路径保持为
`outputs/autodl_guider2_replay_handoff_20261104/`，以兼容脚本现有的项目根目录解析。

模型权重、LoRA adapter、training_state、视频、数据集、预编码缓存、训练日志、
环境压缩包及旧归档由 `.gitignore` 排除。本地文件仍然保留。
`Wan21/model/`、`HY15/trainer/models/` 等模型实现代码不在排除范围内。
`.env.example` 等配置样例允许提交，真实 `.env` 和凭据目录不提交。

## 日常保存修改

```bash
cd /root/data/ltl/timeOPD/minWM
git status --short
git add .
git diff --cached --stat
git commit -m "Describe the change"
```

提交前检查暂存区，避免用 `git add -f` 将模型或私有配置强行纳入版本管理。
`.gitignore` 按路径和文件类型匹配，不能自动按文件大小过滤任意新文件。

## 绑定自己的远程仓库

当前先建立本地仓库，不自动创建或发布 GitHub/GitLab 项目。
先创建合适的远程仓库，再把下面引号内的占位内容替换为真实地址：

```bash
git remote add origin "<远程Git仓库地址>"
git push -u origin main
```

在另一台服务器首次获取代码：

```bash
git clone "<远程Git仓库地址>" /root/autodl-tmp/ltl/minWM-code
```

该示例使用新的 `minWM-code` 目录，不覆盖已有的 AutoDL 实验目录。
已有目录含运行数据或本地修改时，先检查再同步；不要直接覆盖正在使用的代码。

模型、数据和 Python 环境仍需通过已准备的迁移脚本或 rsync 单独传输。
克隆代码不会自动下载被忽略的模型和数据，也不会自动替换源服务器绝对路径；
跨服务器 SFT 使用迁移脚本生成的目标配置。

## 查询某个文件为何被忽略

```bash
git check-ignore -v --no-index logs/example/model.pt
git check-ignore -v --no-index outputs/example/video.mp4
```
