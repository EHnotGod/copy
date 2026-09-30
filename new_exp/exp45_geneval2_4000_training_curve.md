# §45：GenEval2 GM 数据规模、稳定性与复现

[返回实验目录](README.md)。三份原记录合并；质量曲线与稳定 gate 是两个问题，分别保留。

<a id="exp45-main"></a>

<!-- report:exp45-main:start -->
## §45 GenEval2 GM Router 2400→4000 训练数据扩展

> 数据和标签准备时间线见 `exp45_geneval2_4000_training_curve.md#exp45-preparation`。

### 实验问题

§44 在 `train≤2400, safety≤400` 的网格中没有找到稳定上线组合。
本实验固定 deployment safety/test，只扩 construction 训练标签，寻找 Router 首次通过
预注册稳定 gate 的训练规模。

### Exp44 基线

- `k=2400` 的固定 raw400 test gain 为 `+.0138`，95% CI `[-.0033,+.0309]`。
- 5/5 construction split 的 test 点估计为正，但 safety gate 最多只有1/5 split通过。
- 因而点估计有希望，但统计证据不足，不能上线。

### 数据与协议

- 保持 §43 的 `k=2400` split 与缓存完全不变；新增1600条与旧 pool/raw 3-gram 去重的 prompt。
- 新数据的 `atom_count=3..10` 各200条；prompt embedding shape为1600×4096。
- `k={2400,2800,3200,3600,4000}`，固定排除旧 pool 的200条 construction safety reserve，
  对应 fit n={2200,2600,3000,3400,3800}；5个 construction seeds。
- 使用 §44 固定且互斥的官方 raw400 safety + raw400 final-test；
  safety `n={50,100,150,200,250,300,350,400}`。
- 稳定上线要求：final-test paired CI下界>0、5/5 split test点估计>0、
  且至少4/5 split的 safety CI下界>0。
- 5000次 paired bootstrap；公共A7、预算10、dense target-only Router。

### 结果

下表报告 safety=400；其他 safety n 的完整结果保存在结构化 JSON 中。

| k | fit n | test gain [95% CI] | 5/5 test正 | safety400通过 | sustained safety n | deployment-ready |
|---:|---:|---:|---:|---:|---:|---:|
| 2400 | 2200 | +0.0138 [-0.0036,+0.0308] | 是 | 0/5 | — | 否 |
| 2800 | 2600 | +0.0092 [-0.0073,+0.0258] | 是 | 1/5 | — | 否 |
| 3200 | 3000 | +0.0073 [-0.0100,+0.0254] | 是 | 1/5 | — | 否 |
| 3600 | 3400 | +0.0084 [-0.0095,+0.0265] | 是 | 0/5 | — | 否 |
| 4000 | 3800 | +0.0132 [-0.0054,+0.0328] | 是 | 0/5 | — | 否 |

### 稳定阈值

- 当前扩展网格仍未达到预注册稳定上线标准。

### 完整性审计

- construction pool score shape：`(4000, 7)`。
- prompt embedding shape：`(4000, 4096)`。
- 分数与 embedding 全部有限：`True`。
- 最大 final-test Router 成本：`10.0000`（预算10）。
- 三份扩展 per-seed 标签均通过1600条完整动作键审计后才启动训练。

### 复现注意

Qwen 生成日志提示命令行中的 `true_cfg_scale=4.0` 与 `guidance_scale=1.0`
在当前 pipeline 中被忽略；实际生效配置与完整运行时间线见准备日志。

### 产物

- 准备日志：`exp45_geneval2_4000_training_curve.md#exp45-preparation`
- 脚本：`run_exp45_geneval2_4000_training_curve.py`
- 结果：[results/exp45_geneval2_4000_training_curve.json](results/exp45_geneval2_4000_training_curve.json)
- 模型缓存：`results/exp45_target_gm_seed*_k*_dense.npz`
- 前序实验：`exp44_router_stability_surface.md`
<!-- report:exp45-main:end -->

<a id="exp45-quality"></a>

<!-- report:exp45-quality:start -->
## GM Router 数据量曲线：200→4000

![GM Router data-scale curve](results/exp45_gm_router_data_scale.png)

### 统一口径

- 模型：dense target-only Ridge Router，公共A7，预算10，construction seeds 42–46。
- 横轴 `k` 是总 construction prompts；括号内另列真正参与fit的prompt数。
- Exp42、Exp43、Exp45 的缓存训练过程保持各自预注册协议不变。
- 为避免跨实验测试集口径不同造成假趋势，本图不直接拼接旧表；所有prediction cache都重新评估在
  Exp44/45固定raw400 final-test上。
- 纵轴只展示Router自身的soft-TIFA GM平均质量分，不减LP或其他baseline；越高越好。
- 误差棒是5个construction splits平均后的prompt bootstrap 95% CI，共5000次。

### 精确结果

| k（fit n） | Router平均质量分 [95% CI] |
|---:|---:|
| 200（160） | 0.3326 [0.3057,0.3593] |
| 300（250） | 0.3375 [0.3116,0.3641] |
| 400（350） | 0.3363 [0.3089,0.3626] |
| 600（540） | 0.3442 [0.3176,0.3723] |
| 800（720） | 0.3457 [0.3190,0.3732] |
| 1200（1080） | 0.3481 [0.3208,0.3755] |
| 1800（1620） | 0.3516 [0.3229,0.3810] |
| 2400（2200） | 0.3637 [0.3341,0.3927] |
| 2800（2600） | 0.3591 [0.3309,0.3891] |
| 3200（3000） | 0.3573 [0.3273,0.3862] |
| 3600（3400） | 0.3583 [0.3294,0.3888] |
| 4000（3800） | 0.3632 [0.3337,0.3934] |

### 解读

1. Router平均质量从`k=200`的0.3326提高到`k=2400`的0.3637，绝对增加0.0312，
   即约3.12个soft-TIFA GM百分点；主要增益发生在200→2400。
2. 小范围并非严格单调，例如300→400略降；但200→1800的总体趋势向上，2400出现一次明显跃升。
3. `k=2400→4000`进入平台：2800/3200略降，4000恢复到0.3632，与2400几乎相同。
   继续增加数据没有产生持续的质量爬升。
4. 本图按要求只展示Router绝对质量，不能单独回答Router是否优于LP或是否达到部署稳定性；
   baseline差值与safety结论仍以Exp45主实验为准。

### 产物

- 绘图脚本：`plot_exp45_gm_router_data_scale.py`
- 结构化曲线：[results/exp45_gm_router_data_scale.json](results/exp45_gm_router_data_scale.json)
- 图片：[results/exp45_gm_router_data_scale.png](results/exp45_gm_router_data_scale.png)
<!-- report:exp45-quality:end -->

<a id="exp45-preparation"></a>

<!-- report:exp45-preparation:start -->
## 数据准备与复现

准备阶段已经完成。保留数据合同、文件、实际参数与最终产物；逐次失败/提交快照见 [合并前归档](../附件范围.md#attachment-2) 中原 `exp45_geneval2_4000_training_curve.md#exp45-preparation`。

### 新增 construction 数据

- 文件：`../data/geneval2_more_1600.jsonl`
- 规模：1600条；`atom_count=3..10` 各200条。
- SHA-256：`274c96cb7c5056ae6cb16570ff0aa9c66aae9f2afe4f92cfaf651519e2d3b9e4`
- 去重：按小写、单复数归一化后的连续3-gram，与官方 raw800、旧2400 construction
  pool 的交集均为0；新批次内部同样去重。
- Prompt embedding：`../codes/.cache_embed/emb_586a56d72a7489913c78c43b68fc3fc0.npy`，
  shape=`(1600, 4096)`、dtype=`float32`。
- 一次生成曾在1407行处被截断；损坏副本保存在
  `../data/geneval2_more_1600.corrupt-1407.bak.jsonl`，完整数据已重新生成。
- 生成器 [../data/build_geneval2_new_2000.py](../附件范围.md#attachment-3) 已改为：不足目标条数时失败、临时文件
  写完后用 `os.replace` 原子提交，避免再次留下“表面存在但内容不完整”的数据文件。

### A7 标签矩阵

所有动作均为5个 image seeds，评分器为 Qwen3-VL-8B-Instruct，聚合指标为
`soft_tifa_gm`，最终保留 per-seed 标签。

| 阶段 | 动作/步数 | 预计图片数 | Exp45 所需键 |
|---|---|---:|---|
| Qwen direct | 10steps、50steps | 16,000 | `10steps`, `50steps` |
| Qwen TTS edit | 50steps init + 50steps edit | 8,000 | `tts` |
| Flux direct | 10steps、50steps | 16,000 | `10steps`, `50steps` |
| Flux TTS edit | 50steps init + 50steps edit | 8,000 | `tts` |
| Flux4B direct | 4steps | 8,000 | `4steps` |
| 合计 | 7个公共 A7 动作 | 56,000 | — |

统一入口为 `../codes/geneval2/run_more1600_a7.sh`，Hope 配置位于
`../codes/geneval2/hope_more1600/`。

### 实际 guidance 配置

Qwen direct 的命令行记录了 `true_cfg_scale=4.0`、`guidance_scale=1.0`，但当前
pipeline 日志反复提示：没有 negative prompt，因此 classifier-free guidance 未启用，
`true_cfg_scale` 被忽略；模型也不是 guidance-distilled，`guidance_scale` 被忽略。
任务运行正常，但报告实际配置时必须同时保留这条 runtime warning，不能把两个命令行值解释为
实际生效的 guidance 设置。

### 标签复核

2026-09-11 重新检查三份 more1600 per-seed JSON：三者共有 1600 个唯一 prompt，A7 共 56,000 份 seed score_list 全部非空且有限。AM 可由同一批 score_list 重聚合；AM 结果仍单列在 [§46](exp46_geneval2_am_4000_training_curve.md)。
<!-- report:exp45-preparation:end -->
