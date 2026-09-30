# §18–22：9-cfg 跨模型路由历史记录

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

已按 prompt 对齐特征与标签，剔除 UniGen 与 raw600 重合的 400 条训练候选，并将路由选择改为仅使用预测分数。以下只包含重算的 Ridge，不混入旧 KNN/TFMLP。

GM/AM 为 pool2400/test800；UniGen 为 pool2000/test600。§18 的 UniGen 最大训练点为 1800 + val200。


| 实验 | 数据集 | 完成的条件 × seed 行数 | 当前明细 |
|---|---|---:|---|
| §18 | GM | 18 | [JSON](results/prompt_disjoint_v1/exp18_ridge_cpu_gm.json) |
| §18 | AM | 18 | [JSON](results/prompt_disjoint_v1/exp18_ridge_cpu_am.json) |
| §18 | UniGen | 18 | [JSON](results/prompt_disjoint_v1/exp18_ridge_cpu_unigen.json) |
| §19 | GM | 50 | [JSON](results/prompt_disjoint_v1/exp19_ridge_cpu_gm.json) |
| §19 | AM | 50 | [JSON](results/prompt_disjoint_v1/exp19_ridge_cpu_am.json) |
| §20 | GM | 3 | [JSON](results/prompt_disjoint_v1/exp20_ridge_cpu_gm.json) |
| §20 | AM | 3 | [JSON](results/prompt_disjoint_v1/exp20_ridge_cpu_am.json) |
| §20 | UniGen | 3 | [JSON](results/prompt_disjoint_v1/exp20_ridge_cpu_unigen.json) |
| §21 | GM | 9 | [JSON](results/prompt_disjoint_v1/exp21_ridge_cpu_gm.json) |
| §21 | AM | 9 | [JSON](results/prompt_disjoint_v1/exp21_ridge_cpu_am.json) |
| §21 | UniGen | 6 | [JSON](results/prompt_disjoint_v1/exp21_ridge_cpu_unigen.json) |
| §22 | GM | 9 | [JSON](results/prompt_disjoint_v1/exp22_ridge_cpu_gm.json) |
| §22 | AM | 9 | [JSON](results/prompt_disjoint_v1/exp22_ridge_cpu_am.json) |
| §22 | UniGen | 9 | [JSON](results/prompt_disjoint_v1/exp22_ridge_cpu_unigen.json) |

§19 用满固定训练池，10 个训练 seed 只改变行顺序，不构成 10 个独立训练划分；修复脚本将同一训练集合排序后复用解，因此相同条件下结果一致。图片 seed 数 1–5 的比较仍保留。
§20 的 mid 过滤会减少可训练样本，不能把与 all-pool 的差异全部归因为样本难度。旧测试均分 LP 也不作为部署基线；新明细的 LP 仅由训练标签确定。
新旧分数的差异同时包含行序、训练划分、策略选择等修复，不能归因于单一因素。
复现：`rerun_legacy_ridge_cpu.py --threads 4`，只读取已有 embedding。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

[返回实验目录](README.md)。本页合并原汇总与五份分实验记录；重复表格保留分实验版本。

## 核对结论与引用限制

- **§18–22 的旧 Ridge 结果均存在特征/标签行序问题。** 标签按 instruction 排序，Ridge embedding 按文件原序截取；本次逐条比对 GM/AM pool 有 2398/2400 行错位、test 有 798/800 行错位，UniGen 为 2399/2400 与 599/600。
- KNN/TFMLP 的 CLIP 输入直接使用同一份排序后的 instruction；没有发现相同的行序缺陷。此结论不等于对所有缓存内容完成重新验证。
- §18 的对齐后复核见 [§23](exp23_10action_distill.md#旧-18-复现审计)。§19–22 未见对齐后重跑产物，下面 Ridge 表仅作历史记录；旧“Ridge 类别结构有效”“Ridge seed 稳定”等推断撤回，不能从这些表重建有效结论。
- UniGen 旧 pool/test 另有 400 条 prompt 重合；因此该域的独立测试主张也不成立。详见 [核对证据](../修改日志.md#数据与对齐核对)。
- 旧汇总把 LP 称为“上界”不准确：LP 是静态混合基线，Oracle 才是使用真值的路由上界。

## 公共设置

Qwen/Flux/Boogu × 10/50/TTS 共 9 动作，成本 `[2.0,10.0,30.5,4.7,23.1,44.9,4.3,21.5,64.8]`，budget=10。

以下 §18 为原实验记录；旧 Ridge 数值及依赖它的解释受上述限制。

<a id="exp18"></a>

<!-- report:exp18:start -->
## §18 9-cfg 跨模型路由 LC 实验


### 实验动机

exp18 是 exp4 的 9-cfg 版本。单模型路由（exp4）中，预算固定为各模型 50steps 的 cost，每个 (dataset, model) 组合独立训练预测器，在 3 cfgs（10/50/tts）之间路由。

exp18 将三个模型的所有 cfg 合并为 9-cfg 跨模型路由（qwen10/50/tts + flux10/50/tts + boogu10/50/tts）并使用统一预算 budget=10.0（对应 qwen50 的 cost），predictor 同时在 9 个 cfg 之间做 Lagrangian 路由。

### 实验设置

- 路由策略：Lagrangian 松弛，扫描 λ，argmax(pred_j - λ×cost_j) for j=1..9
- Score mode：直接预测 9 个绝对分数；Delta mode：预测相对 qwen50（QWEN50_IDX=1）的差值
- Budget=10.0（统一，对应 qwen 50steps cost），COSTS_9=[2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8]
- Oracle：测试集真实分路由（上界）；LP：全 pool 均值凸包（下界）；Always-qwen50：全用 qwen 50steps
- 预测器：Ridge（M=9）, KNN（M=9）, TFMLP（M=9）
- 数据集：GM（test=800）, AM（test=800）, UniGen（test=600）

### 方法简称说明

| 简称 | 含义 |
|------|------|
| RS | Ridge Score Mode |
| RD | Ridge Delta Mode |
| KS | KNN Score Mode |
| KD | KNN Delta Mode |
| TS | TFMLP Score Mode |
| TD | TFMLP Delta Mode |

### 结果汇总（n=2200，routing quality 原始值）

| Dataset | Oracle | LP | Always-qwen50 | RS | RD | KS | KD | TS | TD |
|---------|--------|----|--------------|----|----|----|----|----|----|----|
| gm | 0.5489 | 0.3755 | 0.2482 | 0.3507 | 0.3507 | 0.3876 | 0.3876 | 0.3883 | 0.3823 |
| am | 0.8583 | 0.7947 | 0.7300 | 0.7806 | 0.7806 | 0.7993 | 0.7993 | 0.8000 | 0.8002 |
| unigen | 0.9626 | 0.8618 | 0.7518 | 0.8300 | 0.8300 | 0.8764 | 0.8764 | 0.8578 | 0.8768 |

_RS=Ridge-Score, RD=Ridge-Delta, KS=KNN-Score, KD=KNN-Delta, TS=TFMLP-Score, TD=TFMLP-Delta_

### 实验结论

- **9-cfg 跨模型路由全面超过 always_qwen50 基线**，KNN-Score 在 GM/0.3876、AM/0.7993、UniGen/0.8764，相对 always_qwen50 的提升分别为 +0.139、+0.069、+0.125。
- **Score mode 和 Delta mode 差异极小**（RS=RD，KS=KD），9-cfg 下 delta（相对 qwen50 的差值）与 score 完全等价，路由结果相同。
- **TFMLP 在 AM/UniGen 上略优**（TD 达到 0.8002/0.8768），KNN 在 GM 略优（0.3876），Ridge 在各数据集均低于 KNN/TFMLP。
- **学习曲线趋势稳定**：n=300 已有明显超越 always_qwen50 的路由收益，n>=1000 后趋于饱和。
- **GM 路由空间最大**（Oracle 0.5489 vs LP 0.3755 vs always_qwen50 0.2482），UniGen 次之（0.9626/0.8618/0.7518），AM 最小（0.8583/0.7947/0.7300）。

### 图表

#### GenEval-M (GM)
![exp18 routing gm](figs/exp18_9cfg_routing_gm.png)

#### AM-EditBench (AM)
![exp18 routing am](figs/exp18_9cfg_routing_am.png)

#### UniGenBench
![exp18 routing unigen](figs/exp18_9cfg_routing_unigen.png)
<!-- report:exp18:end -->

以下 §19 为原实验记录；旧 Ridge 数值及依赖它的解释受上述限制。

<a id="exp19"></a>

<!-- report:exp19:start -->
## §19 9-cfg 跨模型路由 Seed Sensitivity

### 实验动机

exp19 是 exp5 的 9-cfg 版本。9-cfg 跨模型路由中，每条样本的质量分是 5 个图片 seed 的均值，但实际部署时可能只跑 1-2 个 seed。本实验探究：**训练/验证集若只用 k（k=1..5）个图片 seed 均值作为 ground truth，9-cfg 路由效果会下降多少**？Budget=10.0（统一），覆盖 GM 和 AM 两个数据集（unigen 无 per-seed 数据）。

### 实验设置

- 固定 n=2200，10 个 train/val split seeds（取均值±std）
- 改变 train/val 每个样本用的图片 seed 数量 k=1..5（取前 k 个 seed 均值作为 score）
- test 固定用 5-seed avg 作为 ground truth
- Budget=10.0（统一跨模型），COSTS_9=[2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8]
- 横轴：图片 seed 数量 k；纵轴：routing quality 原始值
- 参考线：Oracle（9-cfg 真实分路由上界）、Always-qwen50
- 预测器：Ridge, KNN（score mode，9-cfg 统一预测）

### 数值汇总

#### GenEval-M (GM)（Oracle=0.5489, Always-qwen50=0.2482）

| k | Ridge | KNN |
|---|-------|-----|
| 1 | 0.3453 | 0.3793 |
| 2 | 0.3481 | 0.3741 |
| 3 | 0.3526 | 0.3819 |
| 4 | 0.3524 | 0.3863 |
| 5 | 0.3507 | 0.3876 |

#### AM-EditBench (AM)（Oracle=0.8583, Always-qwen50=0.7300）

| k | Ridge | KNN |
|---|-------|-----|
| 1 | 0.7763 | 0.7928 |
| 2 | 0.7851 | 0.7929 |
| 3 | 0.7865 | 0.7984 |
| 4 | 0.7835 | 0.7996 |
| 5 | 0.7806 | 0.7993 |

### 实验结论

- **k=1 到 k=5 路由质量曲线总体平坦**，增加图片 seed 数量带来的提升十分有限（通常 <0.005），说明用单张图片的分数训练预测器已足够稳定。
- **GM 上 KNN 随 k 增大略有提升**（k=1→0.3793 到 k=5→0.3876），Ridge 也类似但幅度更小。
- **AM 上各 predictor 对 k 的变化极不敏感**（差距 <0.003），基线分数高，路由空间有限。
- **9-cfg 与 exp5（单模型）结论一致**：标注时每条 query 只跑 1 个 seed 即可，对路由效果几乎无损，可显著降低数据采集成本。

### 图表

#### GenEval-M (GM)
![exp19 seed sens gm](figs/exp19_9cfg_seed_sens_gm.png)

#### AM-EditBench (AM)
![exp19 seed sens am](figs/exp19_9cfg_seed_sens_am.png)
<!-- report:exp19:end -->

以下 §20 为原实验记录；旧 Ridge 数值及依赖它的解释受上述限制。

<a id="exp20"></a>

<!-- report:exp20:start -->
## §20 9-cfg Mid-pool vs All-pool

### 实验动机

exp20 是 exp6 的 9-cfg 版本。9-cfg 跨模型路由中，训练池是否应只保留中等难度样本（qwen50分数 in (0.10, 0.90)）以减少极端样本噪声？本实验对比 mid-pool 与 all-pool（exp18 n=2200）的路由效果。

### 实验设置

- Mid-pool：train/val 只用 qwen50-score ∈ (0.10, 0.90) 的样本（QWEN50_IDX=1）
- All-pool（参照 exp18 n=2200 结果）
- Budget=10.0（统一跨模型），COSTS_9=[2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8]
- Score mode only，3 seeds

### 数值汇总

| Dataset | Oracle | qwen50 | n_mid | Ridge(mid) | KNN(mid) | TF(mid) |
|---------|--------|--------|-------|------------|----------|---------|
| gm | 0.5489 | 0.2482 | 1008 | 0.3412 | 0.3763 | 0.3803 |
| am | 0.8583 | 0.7300 | 2263 | 0.7774 | 0.7968 | 0.8000 |
| unigen | 0.9626 | 0.7518 | 1486 | 0.8441 | 0.8711 | 0.8673 |

_Mid-pool 过滤条件：train/val 中 qwen50（50steps）分数 ∈ (0.10, 0.90)_

### 实验结论

- **Mid-pool 整体略差于 All-pool（exp18 n=2200）**：GM 上 KNN mid=0.3763 vs all=0.3876（−0.011），TFMLP mid=0.3803 vs all=0.3883（−0.008）。
- **AM 上几乎持平**，三个 predictor 差距均 < 0.003（mid≈all）。
- **UniGen 上 Ridge mid 反而略好**（0.8441 vs all=0.8300，+0.014），但 KNN mid=0.8711 略差于 all=0.8764（−0.005），整体接近。
- **结论：9-cfg 下 mid-pool 不如 all-pool**，与 exp6（单模型）结论一致，过滤极端样本得不偿失，维持 all-pool 训练。

### 图表

#### GenEval-M (GM)
![exp20 mid pool gm](figs/exp20_9cfg_mid_pool_gm.png)

#### AM-EditBench (AM)
![exp20 mid pool am](figs/exp20_9cfg_mid_pool_am.png)

#### UniGenBench
![exp20 mid pool unigen](figs/exp20_9cfg_mid_pool_unigen.png)
<!-- report:exp20:end -->

以下 §21 为原实验记录；旧 Ridge 数值及依赖它的解释受上述限制。

<a id="exp21"></a>

<!-- report:exp21:start -->
## §21 9-cfg 类别内打乱消融实验

### 实验动机

exp21 是 exp7 的 9-cfg 版本。9-cfg 跨模型路由中，predictor 超过 always_qwen50 基线的原因有两种可能：
（1）学到了 prompt 级别的难度排序信号（真正有价值）；
（2）学到了训练集的类别级别分布偏置（可能虚高）。
通过类别内打乱（intra-shuffle）破坏 prompt 级信号但保留类别级分布，
与全局打乱（full-shuffle）和正常训练（real）对比，定位信号来源。

### 实验设置

- n=2200，score mode only，3 seeds
- **Real**：正常训练，predictor 能学到 per-prompt 难度
- **Intra-shuffle**：类别内随机打乱质量标签（position/count/other），破坏 prompt 级信号（GM/AM 有；UniGen 无类别结构跳过）
- **Full-shuffle**：全局随机打乱，彻底破坏所有信号
- Budget=10.0（统一），COSTS_9=[2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8]

### 数值汇总

#### GenEval-M (GM)（Oracle=0.5489, Always-qwen50=0.2482）

| 条件 | Ridge | KNN |
|------|-------|-----|
| Real | 0.3507 | 0.3876 |
| Intra-Shuffle | 0.3595±0.0059 | 0.3729±0.0051 |
| Full-Shuffle | 0.3436±0.0024 | 0.3620±0.0039 |

#### AM-EditBench (AM)（Oracle=0.8583, Always-qwen50=0.7300）

| 条件 | Ridge | KNN |
|------|-------|-----|
| Real | 0.7806 | 0.7993 |
| Intra-Shuffle | 0.7772±0.0046 | 0.7918±0.0005 |
| Full-Shuffle | 0.7793±0.0028 | 0.7903±0.0012 |

#### UniGenBench（Oracle=0.9626, Always-qwen50=0.7518）

| 条件 | Ridge | KNN |
|------|-------|-----|
| Real | 0.8300 | 0.8764 |
| Full-Shuffle | 0.8382±0.0015 | 0.8536±0.0008 |

### 实验结论

- **GM/AM 上 Real 优于 Full-shuffle（KNN 一致）**：GM KNN Real(0.3876) > Full-shuffle(0.3620)，AM KNN Real(0.7993) > Full-shuffle(0.7903)，说明 predictor 确实学到了有效的质量排序信号。
- **UniGen 上 KNN real(0.8764) > full-shuffle(0.8536)，但 Ridge real(0.8300) < full-shuffle(0.8382)**，Ridge 异常可能因无类别结构的 unigen 中全局打乱偶然引入有利正则化；KNN 更稳健，支持 real 有效。
- **AM 上各条件差距较小**（<0.01），always_qwen50=0.730 基线已高，路由空间有限。
- **GM 上 Intra-shuffle Ridge(0.3595) 略高于 Real(0.3507)**：类别内打乱保留了类别级分布（position/count/other），类别本身对预测有贡献，但 KNN 方向相反（intra=0.3729 < real=0.3876），说明 per-prompt 信号对 KNN 更关键。
- **结论：9-cfg 路由有效信号来自 prompt 级难度特征（KNN 一致支持），类别结构是补充信号**。与 exp7 单模型结论一致。

### 图表

#### GenEval-M (GM)
![exp21 shuffle gm](figs/exp21_9cfg_shuffle_gm.png)

#### AM-EditBench (AM)
![exp21 shuffle am](figs/exp21_9cfg_shuffle_am.png)

#### UniGenBench
![exp21 shuffle unigen](figs/exp21_9cfg_shuffle_unigen.png)
<!-- report:exp21:end -->

以下 §22 为原实验记录；旧 Ridge 数值及依赖它的解释受上述限制。

<a id="exp22"></a>

<!-- report:exp22:start -->
## §22 9-cfg Train/Val 划分比例实验

### 实验动机

exp22 是 exp9 的 9-cfg 版本。9-cfg 跨模型路由中，默认使用 val=200 / train=2200 的划分。本实验验证：改变 val 大小（200/400/800）时，9-cfg 路由效果是否显著变化？

### 实验设置

| 条件 | val_size | n_train |
|------|----------|--------|
| 默认 | 200 | 2200 |
| 中等 | 400 | 2000 |
| 大 val | 800 | 1600 |

- 固定总池 2400，3 seeds，取均值±std
- Budget=10.0（统一），COSTS_9=[2.0, 10.0, 30.5, 4.7, 23.1, 44.9, 4.3, 21.5, 64.8]
- Score mode only，Ridge + KNN

### 数值汇总

| Dataset | Oracle | qwen50 | val=200 (R) | val=400 (R) | val=800 (R) | val=200 (K) | val=400 (K) | val=800 (K) |
|---------|--------|--------|-------------|-------------|-------------|-------------|-------------|-------------|
| gm | 0.5489 | 0.2482 | 0.3479±0.0011 | 0.3476±0.0039 | 0.3480±0.0061 | 0.3855±0.0021 | 0.3841±0.0002 | 0.3806±0.0064 |
| am | 0.8583 | 0.7300 | 0.7791±0.0021 | 0.7773±0.0028 | 0.7768±0.0051 | 0.7963±0.0010 | 0.7970±0.0006 | 0.7968±0.0016 |
| unigen | 0.9626 | 0.7518 | 0.8353±0.0031 | 0.8320±0.0030 | 0.8305±0.0033 | 0.8744±0.0007 | 0.8721±0.0010 | 0.8689±0.0022 |

### 实验结论

- **三种划分的路由质量差异极小**（绝大多数组合差距 < 0.003），与 exp9（单模型）结论完全一致。
- **GM 上 val=200 (Ridge=0.3479) 和 val=800 (0.3480) 几乎持平**，说明 α 选择对 val 大小不敏感。
- **AM 上三种划分差距可忽略**（< 0.002），基线高，空间有限。
- **UniGen 上 val=200 略优于 val=800**（Ridge: 0.8353 vs 0.8305），与 exp9 一致：较少 val 保留更多 train 样本对高信号数据集更有利。
- **结论：维持现有 200 val / 2200 train 的默认划分**，9-cfg 下也无需改动。

### 图表

![exp22 val split](figs/exp22_9cfg_val_split.png)
<!-- report:exp22:end -->


## 历史产物

| 实验 | 原始结果 | 运行脚本 |
|---|---|---|
| §18 | [JSON](results/exp18_9cfg_routing_merged.json) | [脚本](run_exp18_9cfg_routing.py) |
| §19 | [JSON](results/exp19_9cfg_seed_sens.json) | [脚本](run_exp19_9cfg_seed_sens.py) |
| §20 | [JSON](results/exp20_9cfg_mid_pool.json) | [脚本](run_exp20_9cfg_mid_pool.py) |
| §21 | [JSON](results/exp21_9cfg_shuffle.json) | [脚本](run_exp21_9cfg_shuffle.py) |
| §22 | [JSON](results/exp22_9cfg_val_split.json) | [脚本](run_exp22_9cfg_val_split.py) |

</details>
