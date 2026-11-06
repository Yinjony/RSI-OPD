# Guider SFT → Generator BiOPD：两轮实验

日常只改 `configs/` 顶层的四份 YAML，使用两个公共入口。`_base/` 是内部共用采样配置。

当前四份配置默认使用 **20 步去噪**：每块 20 次模型预测、19 次中间重新加噪，最后一次直接输出 x0。`num_frame_per_block: 4` 仍表示每块四个 latent 帧。

| 环节 | 20 步的含义 |
| --- | --- |
| Generator1/2 训练与训练中验证 | 每个视频块完整走 20 步；训练在每个状态计算并反传 BiOPD loss |
| Guider2 anchor | 在 20 步日程上抽取状态，默认每批抽 1 个状态；按所抽状态求均值 |
| Guider1/2 普通 SFT 及 MSE 验证 | 保留从 1000 个噪声时刻随机采样的 velocity MSE，不展开整条 20 步链 |
| Guider/Generator 视频推理及 RSI 数据生成 | 使用 20 步预测/加噪链 |

`denoising_step_list` 是各 recipe 的采样日程；Guider 的 `anchor_denoising_step_list` 默认引用它。Generator 的 `step_weights` 必须与预测次数一致，当前为 20 个 1.0。保留 `warp_denoising_step: true`、`timestep_shift: 5.0` 和底层 1000 项 scheduler 表。

公共推理入口从 recipe 解析日程，实际传给推理脚本；新运行将其保存到输出目录的 `inference_config.yaml`，dry-run 仅使用临时文件。顶层 shift 与模型 wrapper 的 shift 同步；不一致的 anchor/推理日程会报错。旧四步独立推理模板保留在 `_base/inference_4step.yaml`；原训练的 `--resume` 继续读取其保存的旧配置。

这次改动不重新生成训练数据。现有 JSON/缓存仍对应旧四步视频；要建立完整 20 步自举数据链，需要另行按 20 步生成视频、重新标注并预编码。20 步实现已经过 CPU 回归，真实画质、显存、吞吐仍须实测。

| 模型 | 配置 | 训练初始化及数据 | 默认开关 |
| --- | --- | --- | --- |
| Guider1 | [guider1.yaml](configs/guider1.yaml) | 原始 DMD + 新 LoRA；`guider_sft_iter1.json` | `anchor: false`、`relora: false` |
| Guider2 | [guider2.yaml](configs/guider2.yaml) | 原始 DMD + Guider1 1425 的原有 A/B；`guider_sft_iter2.json` | `anchor: true`、`relora: false` |
| Generator1 | [generator1.yaml](configs/generator1.yaml) | 原始 DMD + 新 LoRA；冻结 Guider1；`selector_iter1.json` | `relora: false` |
| Generator2 | [generator2.yaml](configs/generator2.yaml) | 原始 DMD + Generator1 475 的原有 A/B；冻结 Guider2；`selector_iter2.json` | `relora: false` |

四份配置都保留 `lora_rank: 64`、`lora_alpha: 64`。既有检查点、原始 DMD、文本/VAE/tokenizer 资产、训练数据和缓存都使用已经核验的原路径。配置中的 `inference.checkpoint` 指向已训练完成的模型；重新训练后，改该项或用 `--checkpoint /新目录/model.pt` 指向新模型。后续阶段的 `initial_adapter` / `guider_ckpt` 也需随实验选择更新；入口不会自动改用“最新”模型。

## 两个入口

从 minWM 目录运行（从其他目录运行也支持，脚本会自行定位工程）：

```bash
# 每条训练命令单独启动一个模型，不要同时在同一组卡运行这些命令。
bash Wan21/scripts/train.sh guider1 --gpus 2,3
bash Wan21/scripts/train.sh guider2 --gpus 2,3
bash Wan21/scripts/train.sh generator1 --gpus 2,3
bash Wan21/scripts/train.sh generator2 --gpus 2,3

# 每条推理命令默认输出 10 个独立视频，使用指定的一张卡。
bash Wan21/scripts/infer.sh guider1 --gpus 2
bash Wan21/scripts/infer.sh guider2 --gpus 2
bash Wan21/scripts/infer.sh generator1 --gpus 2
bash Wan21/scripts/infer.sh generator2 --gpus 2
```

[train.sh](scripts/train.sh) 和 [infer.sh](scripts/infer.sh) 都委托给 [run_experiment.py](run_experiment.py)。默认解释器沿用 `/root/miniconda3/envs/timeopd/bin/python`，可用 `PYTHON_BIN=/路径/python` 覆盖。新训练输出 `logs/Wan21/Action2V/模型名_20step_时间戳/`，新推理输出 `outputs/模型名_first10_时间戳/`；可用 `--output-dir` 指定精确新路径。新训练拒绝复用已有输出目录。

Generator1/2 的推理沿用 RSI 4000 条 prompt/action 对齐文件，取索引 0–9，第一条是机器人，seed 为 0–9，20 步采样，不使用 guider CFG。视频位于输出目录的 `iteration_000/videos/`。Guider1/2 默认都用 **iter2 的同一批留出验证样本**、同一动作和 seed 规则，CFG=2.0；seed 是 `base_seed + 原始 JSON 索引`。这项通过 `inference.validation_config` 控制，只影响推理选样，不改变训练数据。四种模型默认均为 77 帧、832×480、16fps。

## 新 Generator1 step 475：四卡推理 RSI 后 2000 条

本次模型为 `logs/Wan21/Action2V/generator1_20step_20261020_233626_989045/step_0000475/model.pt`，已包含合并后的 LoRA。使用指定的 mcy `RSI_BiOPD_4000_prompts.txt` 和 `RSI_BiOPD_4000_trajectories.txt`，取第 2001–4000 行（零基索引 `[2000,4000)`）。

普通 `infer.sh` 目前固定从第 0 条开始、只使用所选列表的第一张 GPU，因此此次使用单独准备的四进程启动脚本：GPU 0/1/2/3 各处理 500 条，分片为 `[2000,2500)`、`[2500,3000)`、`[3000,3500)`、`[3500,4000)`。

```bash
cd /root/data/ltl/timeOPD/minWM
RUN=outputs/generator1_20step_20261020_233626_989045_step0000475_last2000

# CPU 检查全部 2000 对数据，不启动 CUDA。
bash "$RUN/run.sh" --dry-run

# 正式启动：在 tmux 中运行四个独立单卡进程。
tmux new-session -d -s gen1_s475_last2000 "bash '$PWD/$RUN/run.sh'"

# 查看各卡日志；Ctrl-C 只退出日志查看。
tail -F "$RUN"/worker_logs/gpu{0,1,2,3}.log
```

视频保存到 `$RUN/iteration_001/videos/002000.mp4` 至 `003999.mp4`，逐条元数据位于 `iteration_001/metadata/`。每片的 `iteration_001/shards/*/summary.json` 应显示 `complete: true`、`completed_total: 500`、`failed_total: 0`。

`$RUN/inference_config.yaml` 已按该 checkpoint 的保存配置准备：20 步、shift=5、77 帧、832×480、16fps，不使用 guider CFG；seed 为原始零基索引（2000–3999）。保持 checkpoint、配置和分片参数不变，重复同一命令可跳过已完成且签名匹配的视频。不要同时启动两份；若 tmux 会话已存在，用 `tmux attach -t gen1_s475_last2000` 查看。

## Anchor、ReLoRA 与 GPU

SFT 中 `anchor: true` 解析成 `sft_loss_type: supervised_anchor`，`false` 解析成 `supervised`。`anchor_eta` 控制权重，当前为 0.1；权重为 0 时不建立 reference。reference 固定为本阶段开始时的模型；Guider2 对应原始 DMD + 初始 Guider1 adapter。reference 不更新，梯度只更新当前 LoRA。BiOPD 使用自己的冻结 guider 目标，没有 SFT anchor 开关。

`relora: false`：如果 `initial_adapter` 有值，就加载并继续训练已有 A/B；为空则第一次初始化 LoRA。旧 base 一直冻结。此前没有 `relora` 布尔参数，“不 ReLoRA”就是通过 `initial_adapter` 加载旧 A/B，在优化器创建之前恢复它们；没有把旧 adapter 合并成新 base，也没有重置 A/B。

`relora: true`：**这里只表示本阶段开始时的一次合并与重置，不是训练过程中周期性 ReLoRA**。必须同时指定 `initial_adapter`；公共训练入口会先在 CPU 校验其原始 base，合并到本次输出目录的 `initialization/relora_base.pt`，再去掉 `initial_adapter`、初始化新的 A/B 和优化器。新 adapter 的 `base_identity` 记录这个新 base，避免后续合并时漏加/重复加。来源文件不修改，初始化来源记录在 `.merge.json`。四份配置当前均为 `false`。第一次训练 Guider1/Generator1 没有旧 adapter，直接设 true 会报错；需要重置旧 adapter 时先填入对应来源。

保存 `model.pt` 时生成一份合并副本用于推理，**不属于 ReLoRA**：不会改变正在训练的 A/B、冻结 base 或优化器。SFT 训练结束公共入口导出最后一个检查点的 `model.pt`；BiOPD 按 `save_merged: true` 导出。

GPU 优先级：`--gpus` > `CUDA_VISIBLE_DEVICES` > YAML 的 `gpus`。`gpus: [2, 3]` 是默认选择，不表示已确认当前空闲。训练进程数由所选 GPU 数量决定，不再手填 `NPROC_PER_NODE`。使用 `effective_batch_size: 4`、`batch_size: 1` 自动推导梯度累积：1 卡→4，2 卡→2，4 卡→1；3 卡不能整除，会提前报错。推理只使用所选列表的第一张卡。改卡数不保证显存或速度相同，入口不抢占或终止其他任务。

```bash
# 不修改 YAML 的临时覆盖
bash Wan21/scripts/train.sh guider2 --gpus 0,1,2,3 --override anchor=false --dry-run
CUDA_VISIBLE_DEVICES=2,3 bash Wan21/scripts/train.sh generator2 --dry-run

# 同一次训练断点续训：读取检查点旁/所属运行目录的 config.yaml。
# 必须保持原训练的卡数；沿用已准备好的 base，不重新合并/reset。
bash Wan21/scripts/train.sh generator2 --gpus 2,3 --resume /某次运行/step_0000100/training_state.pt
```

`--resume` 恢复同一次运行的训练状态；`initial_adapter` 是开启下一阶段，沿用权重并新建优化器。两者不要混用。断点续训仍由原 trainer 严格核对 base、缓存/数据、world size、训练目标等 contract。

## CPU 检查与底层依赖

```bash
bash Wan21/scripts/train.sh guider2 --check-init
bash Wan21/scripts/train.sh guider2 --check-cache
bash Wan21/scripts/train.sh generator2 --check-data
bash Wan21/scripts/infer.sh generator2 --dry-run
```

`--dry-run` 只预览并检查初始化/路径；推理 dry-run 还检查选样和动作，不启动 CUDA、不写视频。它不能证明实际显存、吞吐或视频质量。`--check-data` 读取训练数据；BiOPD 同时核验训练/验证文本缓存覆盖。`--check-cache` 对 SFT 的数据、encoder、视频身份及缓存 contract 做校验。

保留的底层实现：

| 流程 | 必需实现 |
| --- | --- |
| Guider 训练 | [wan_train_guider_sft.py](wan_train_guider_sft.py)、[wan_train_guider_sft_continue_lora.py](wan_train_guider_sft_continue_lora.py)、[camera_guider_sft.py](wan_trainer/camera_guider_sft.py) |
| Generator 训练 | [wan_train_biopd.py](wan_train_biopd.py)、[camera_biopd.py](wan_trainer/camera_biopd.py)、[camera_biopd.py 模型加载](model/camera_biopd.py) |
| Guider 推理 | [infer_guider_sft_validation_cfg.py](infer_guider_sft_validation_cfg.py)、[merge_guider_sft_lora.py](merge_guider_sft_lora.py) |
| Generator 推理 | [infer_rsi_biopd.py](infer_rsi_biopd.py)、[causal_inference.py](pipeline/causal_inference.py) |
| 数据预编码 | [preencode_guider_sft.py](scripts/data_preprocessing/preencode_guider_sft.py)、[preencode_biopd_text.py](scripts/data_preprocessing/preencode_biopd_text.py) |

`wan_utils/`、`model/`、`pipeline/`、`wan/` 、`Wan21/demo_utils/` 以及工程根目录的 `shared/` 等基础模块保留：实际加载及包初始化仍会依赖其中的模块，不能仅凭文件名判断为多余。`tests/` 保留本实验的全部回归测试。`configs/_base/` 为内部采样依赖，日常只改四个模型配置。

与本实验无关或已被统一入口替代的文件已移动到 [useless](../useless/README.md)：旧的卡数/检查点专用脚本、旧 YAML、其他 trainer、旧 CLI、评估/WorldPlayGen 脚本、旧说明与 `time-exp/`。迁移清单记录原路径、目标路径和 SHA256；只移动，不删除。代码修改前的完整文本快照在 [before](../outputs/experiment_refactor_20261017/before/)，验证记录在 [validation.json](../outputs/experiment_refactor_20261017/validation.json)。
