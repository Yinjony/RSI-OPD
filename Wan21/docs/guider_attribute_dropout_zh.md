# Guider SFT 单侧属性训练（2026-11-03）

`attribute_dropout: true` 启用按分支配额采样的属性 dropout。每个训练位置只输入
`(y, a, A_pos)` 或 `(y, a, A_neg)`，保留原视频、相机动作、加噪与 v 的监督目标。
负属性仍描述该视频中观察到的缺陷，负分支使用同样的正权重 MSE。

## 实际训练路径

1. `wan_utils/guider_sft_dataset.py::condition_texts` 分别构造
   `format_prompt(y, A_pos, [])`、`format_prompt(y, [], A_neg)`。
   `scripts/data_preprocessing/preencode_guider_sft.py` 将两段文字分别送入 T5，缓存为
   `positive` / `negative`。不在已混合的 embedding 上屏蔽属性。
2. `wan_utils/guider_sft_sampling.py::BalancedAttributeBatchSampler` 只使用训练划分，
   建立正属性非空池和负属性非空池。池可以重叠，不把视频固定分成正负两组。
3. 先在全局有效 batch 上分配分支，再从对应非空池取视频，随机打乱位置。
   有效 batch = 每卡 batch_size × 卡数 × grad_accum_steps。
   有效 batch 为偶数时严格一半正、一半负；奇数时相邻两次更新轮换多出的一个位置，
   累计差最多 1。任一池为空时直接报错，不切换到另一分支，也不使用空属性。
4. 每个池按打乱后的顺序访问，耗尽后重新打乱。跨 epoch 接续池的位置，
   避免在池未访问完前因 epoch 切换而重复。池顺序和配额均由 seed、epoch 决定，
   断点的 epoch/batch_in_epoch 可重建相同采样流。
5. Dataset 收到 `(index, branch)` 后只返回 `embeddings['selected']`。
   `wan_trainer/camera_guider_sft.py::train` 显式传 `condition='selected'`；
   现有 `backward_sft_with_anchor` 将这个 condition 同时用于 SFT 和 anchor。
   同一视频两种访问的 latent、viewmats、Ks 相同，文本条件不同。

缓存仍保留 `full` 及已有对照条件，方便诊断；启用 dropout 后训练不会读取 `full`。
loss 验证分别报告 `positive` / `negative`，跳过该侧标签为空的视频；无属性和打乱属性
仍是额外的验证对照，不反向传播。`lora_dropout: 0.0` 是 LoRA 层参数，与本开关无关。

## 配方与兼容性

- `configs/guider1.yaml`、`configs/guider2.yaml` 为新训练启用 dropout，并使用新的缓存、输出目录。
  已有第一轮权重和历史运行文件不变；修改 guider1 配方不表示重训已有第一轮。
- guider2 使用 `iteration001_replay_new/Guider_2_new.json`，初始化为原始 DMD + 新 LoRA：
  `initial_adapter: null`、`relora: false`、`resume: null`。原有 anchor 与 20 步 schedule 保留。
- 新缓存 contract 带 `attribute_dropout: true`。旧缓存缺少单侧文本 embedding，
  必须先用新配方重新预编码；不要直接复用完整属性 embedding。当前未执行 GPU 预编码或训练。
- 历史保存的配置没有此开关时默认 false，保留旧缓存与 resume contract。
  新旧训练模式不能互相直接 resume。
- 日志 `branch_samples` 是当前更新的全局正负次数，`branch_samples_seen` 是累计次数，
  后者也保存到 training_state 并随断点恢复。

本次实现范围为 SFT、预编码、loss 验证及相关配方；BiOPD 的 v+/v- 推理公式、
推理条件构造、memory 库和 Guider/Selector 标注文件没有在本次修改中重写。

## 检查证据

CPU 数据检查确认第二轮 2000 个不同视频；按现有 5% 验证划分得到 1900 训练 / 100 验证。
训练正池 1788、负池 1092。使用生产采样器模拟 1425 次更新、有效 batch 为 4：

| 卡数（模拟分片） | 每卡 batch | 梯度累积 | 正分支 | 负分支 | 空标签 / 不平衡更新 |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 4 | 2850 | 2850 | 0 / 0 |
| 2 | 1 | 2 | 2850 | 2850 | 0 / 0 |
| 4 | 1 | 1 | 2850 | 2850 | 0 / 0 |

详细计数、标签源 SHA256 和实现 SHA256 见
`outputs/guider_attribute_dropout_check_20261103/sampling_audit.json`。
这只是 CPU 采样证据，不是多 GPU 训练结果。

`tests/test_guider_attribute_dropout.py` 检查正负文本隔离、空标签排除、多卡与累积配额、
跨 epoch 池覆盖、断点采样流、实际训练循环计数、单侧 loss 验证、旧缓存拒绝，
以及 SFT 与真实 on-policy anchor 的单侧条件传递。
相关原有 CPU SFT / anchor / LoRA / 启动器回归也已检查，包含双进程 Gloo 梯度同步。
