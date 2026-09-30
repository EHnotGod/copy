# exp_catimage818 汇总：COCO CATImage 复刻 exp_815 主线（7-cfg 直接预测 + 路由，score 模式）

主线设置：**train:12869 / val:1838 / test:1839，score 模式**（直接预测绝对分），主指标 **clip**，budgets = 路由池各成员成本档（1.5 / 1.54 / 23.92 / 119.7 / 210.0 / 598.5 TFLOPs）。
路由池 7 cfg（CATImage 论文 Table 1 成员，成本单位 TFLOPs，依据 [../codes/coco/catimage_7cfg_costs.json](../codes/coco/catimage_7cfg_costs.json)）：
`infinity_1step(1.5) / sdxl_turbo_1step(1.54) / sdxl_lightning_4steps(23.92) / sdxl_low_10steps(119.7) / sdxl_deepcache_50steps(210.0) / sdxl_ddim_50steps(598.5) / sdxl_high_50steps(598.5)`。
数据集：COCO CATImage（论文 2506.14753v3 Sec 5.2），论文将 COCO 18,384 prompts 划为 train/val/test = 14707/1838/1839；我们的 `coco_train.json`/`coco_test.json` 即论文 train/test 分片，论文 val 分片没有图和评分，因此 **val 从 train 14707 里切**（固定 seed=42 shuffle 后前 1838 为 val，剩余 12869 为 train pool，主实验 train = 全部 12869），test = 1839 不动。

> 注：7 个模型 × 4 指标（clip/sharp/aesthetic/imagereward）所有 prompt 全部有值、无缺失；per-prompt 标签 = 4 个生成 seed 的均值（对应论文 S=4，个别 prompt 单 cfg 不足 4 seed 时按可用 seed 求均值）。数据路径 `Results/coco_catimage/{model}/scores_{train,test}.json`；prompt 列表 [../codes/coco/coco_train.json](../codes/coco/coco_train.json)（14707）/ [../codes/coco/coco_test.json](../codes/coco/coco_test.json)（1839），task_id 与行号一一对齐（加载侧全量对账 prompt 文本，0 条错位）。exp_815 遗留的 train_size=2200 / val=200 旧协议已废弃（旧结果在 `results/backup_2200/`），本文件所有数字均为新协议单一设置。

seeds [0,1,2]；predictor = Ridge / KNN / TFMLP（同 exp_815 实现）。

---

## 实验一：直接预测 —— 四指标均无 0/1 堆积；中段预测力有限（sharp 除外）

做法：用 train（12869）训练 predictor（val 1838 选参/早停），在 test（1839）上逐条预测，计算 R² / Spearman。四个指标（clip/sharp/aesthetic/imagereward）各训各的 predictor（论文口径：每个质量指标单独训一个模型）。

- Predictor：Ridge（Qwen3-Embedding-8B 4096d + StandardScaler，α∈{1,10,100,1000} 按 val R² 逐 cfg 选）、KNN（CLIP mean-pool + StandardScaler，k=min(50, n//3)=50）、TFMLP（CATImage Appendix D：CLIP per-token → 2 层 self-attn + per-token MLP，M=7 一次训练，min-max 归一化用 train lo/hi，val 早停，seed=0）

### 核心现象

四个指标的 test 分布均为单峰、**均无 exp_815 里 GenEval 式指标的 0/1 两端堆积**（exp_815 的 R²(mid) 用 y∈(0.1,0.9) 绝对阈值，在这里失去意义），因此 **R²(mid) 统一用分位数定义：y_true ∈ (q10, q90)**，即去上下各 10%，n_mid=1471（全部指标 × cfg 相同，分位数定义下恒为 80%×1839 取整）。**R²(all) 中等偏好（sharp 最高 0.63~0.68，imagereward 最低 0.19~0.25），但 clip/aesthetic/imagereward 的 R²(mid) 大幅缩水（clip −0.014~+0.138、aesthetic −0.007~+0.188、imagereward Ridge/TFMLP 几乎全负）；唯一例外是 sharp，R²(mid) 仍有 +0.20~+0.41**——中段塌陷不是 clip 特有，而是除 sharp 外的共性；排序信号（Spearman）在各指标上都显著强于 R² 所显示的校准能力。

### 分布柱状图（四指标均单峰、无端点堆积）

test true 分数的 20-bin 直方图（自适应 x 范围，虚线为 q10/q90，标题标 μ/σ），每指标一张（7 cfg 子图）：

- **clip**：μ∈[0.268, 0.282]，σ≈0.03，近似正态；infinity_1step 均值最低（0.2683）、略左偏
- **sharp**：μ∈[0.108, 0.155]，σ∈[0.041, 0.074]，近似正态、右尾略长；infinity 均值最高（0.1552）
- **aesthetic**：μ∈[5.378, 5.763]，σ∈[0.349, 0.401]，近似正态；turbo 最低（5.3777）、lightning 最高（5.7632）
- **imagereward**：μ∈[0.620, 1.111]，σ∈[0.677, 0.750]，明显左偏（长左尾、高分侧陡）；low_10steps 最低（0.6199）、infinity 最高（1.1115）
- 四指标均无任何端点堆积（与 exp_815 GM 低分堆积 / UniGen 高分堆积完全不同）

![dist_clip](figs/exp1_dist_clip.png)
![dist_sharp](figs/exp1_dist_sharp.png)
![dist_aesthetic](figs/exp1_dist_aesthetic.png)
![dist_imagereward](figs/exp1_dist_imagereward.png)

### 拟合散点图

每图 3 行（Ridge/KNN/TFMLP）× 7 列（cfg），pred vs true，格内标 R²(all)/R²(mid)，红/橙竖线为 q10/q90，每指标一张：

![scatter_clip](figs/exp1_scatter_clip.png)
![scatter_sharp](figs/exp1_scatter_sharp.png)
![scatter_aesthetic](figs/exp1_scatter_aesthetic.png)
![scatter_imagereward](figs/exp1_scatter_imagereward.png)

形态（四指标一致）：Ridge 点云斜率正确但偏平；KNN 预测被压到很窄的一条带（邻居均值回归向 train 均值）；TFMLP 预测发散度大于真值（部分超出真值范围），但点云走向最陡。

### R² 表格

mid 子集定义：y_true ∈ (q10, q90)（分位数定义）；n_mid=1471（全部指标 × cfg 相同）。每指标 R²(all) + R²(mid) 两张全表。

**clip · R²(all)**（全 test 集 R²）：

| cfg | Ridge | KNN | TFMLP | n_mid |
|---|---|---|---|---|
| infinity_1step | +0.409 | +0.227 | +0.450 | 1471 |
| sdxl_turbo_1step | +0.297 | +0.150 | +0.402 | 1471 |
| sdxl_lightning_4steps | +0.358 | +0.206 | +0.461 | 1471 |
| sdxl_low_10steps | +0.357 | +0.183 | +0.476 | 1471 |
| sdxl_deepcache_50steps | +0.352 | +0.169 | +0.463 | 1471 |
| sdxl_ddim_50steps | +0.355 | +0.165 | +0.461 | 1471 |
| sdxl_high_50steps | +0.354 | +0.166 | +0.461 | 1471 |

**clip · R²(mid)**（仅 y_true ∈ (q10, q90) 子集）：

| cfg | Ridge | KNN | TFMLP | n_mid |
|---|---|---|---|---|
| infinity_1step | +0.098 | +0.129 | +0.129 | 1471 |
| sdxl_turbo_1step | -0.001 | +0.040 | +0.070 | 1471 |
| sdxl_lightning_4steps | +0.009 | +0.083 | +0.099 | 1471 |
| sdxl_low_10steps | -0.014 | +0.056 | +0.138 | 1471 |
| sdxl_deepcache_50steps | -0.007 | +0.090 | +0.111 | 1471 |
| sdxl_ddim_50steps | +0.012 | +0.080 | +0.125 | 1471 |
| sdxl_high_50steps | +0.016 | +0.082 | +0.121 | 1471 |

**sharp · R²(all)**：

| cfg | Ridge | KNN | TFMLP | n_mid |
|---|---|---|---|---|
| infinity_1step | +0.637 | +0.386 | +0.663 | 1471 |
| sdxl_turbo_1step | +0.572 | +0.281 | +0.634 | 1471 |
| sdxl_lightning_4steps | +0.606 | +0.399 | +0.675 | 1471 |
| sdxl_low_10steps | +0.592 | +0.358 | +0.659 | 1471 |
| sdxl_deepcache_50steps | +0.554 | +0.336 | +0.647 | 1471 |
| sdxl_ddim_50steps | +0.549 | +0.342 | +0.634 | 1471 |
| sdxl_high_50steps | +0.548 | +0.341 | +0.633 | 1471 |

**sharp · R²(mid)**：

| cfg | Ridge | KNN | TFMLP | n_mid |
|---|---|---|---|---|
| infinity_1step | +0.386 | +0.291 | +0.405 | 1471 |
| sdxl_turbo_1step | +0.299 | +0.211 | +0.310 | 1471 |
| sdxl_lightning_4steps | +0.343 | +0.297 | +0.405 | 1471 |
| sdxl_low_10steps | +0.329 | +0.245 | +0.381 | 1471 |
| sdxl_deepcache_50steps | +0.231 | +0.228 | +0.318 | 1471 |
| sdxl_ddim_50steps | +0.205 | +0.219 | +0.286 | 1471 |
| sdxl_high_50steps | +0.202 | +0.219 | +0.285 | 1471 |

**aesthetic · R²(all)**：

| cfg | Ridge | KNN | TFMLP | n_mid |
|---|---|---|---|---|
| infinity_1step | +0.515 | +0.347 | +0.493 | 1471 |
| sdxl_turbo_1step | +0.432 | +0.250 | +0.421 | 1471 |
| sdxl_lightning_4steps | +0.560 | +0.361 | +0.569 | 1471 |
| sdxl_low_10steps | +0.536 | +0.320 | +0.529 | 1471 |
| sdxl_deepcache_50steps | +0.526 | +0.314 | +0.528 | 1471 |
| sdxl_ddim_50steps | +0.499 | +0.292 | +0.499 | 1471 |
| sdxl_high_50steps | +0.516 | +0.308 | +0.516 | 1471 |

**aesthetic · R²(mid)**：

| cfg | Ridge | KNN | TFMLP | n_mid |
|---|---|---|---|---|
| infinity_1step | +0.078 | +0.188 | +0.024 | 1471 |
| sdxl_turbo_1step | +0.015 | +0.143 | -0.007 | 1471 |
| sdxl_lightning_4steps | +0.098 | +0.185 | +0.145 | 1471 |
| sdxl_low_10steps | +0.052 | +0.092 | +0.064 | 1471 |
| sdxl_deepcache_50steps | +0.044 | +0.119 | +0.060 | 1471 |
| sdxl_ddim_50steps | +0.042 | +0.120 | +0.084 | 1471 |
| sdxl_high_50steps | +0.040 | +0.112 | +0.099 | 1471 |

**imagereward · R²(all)**：

| cfg | Ridge | KNN | TFMLP | n_mid |
|---|---|---|---|---|
| infinity_1step | +0.272 | +0.179 | +0.249 | 1471 |
| sdxl_turbo_1step | +0.231 | +0.141 | +0.193 | 1471 |
| sdxl_lightning_4steps | +0.216 | +0.150 | +0.209 | 1471 |
| sdxl_low_10steps | +0.285 | +0.176 | +0.240 | 1471 |
| sdxl_deepcache_50steps | +0.249 | +0.170 | +0.237 | 1471 |
| sdxl_ddim_50steps | +0.237 | +0.157 | +0.239 | 1471 |
| sdxl_high_50steps | +0.242 | +0.159 | +0.234 | 1471 |

**imagereward · R²(mid)**：

| cfg | Ridge | KNN | TFMLP | n_mid |
|---|---|---|---|---|
| infinity_1step | -0.036 | +0.068 | +0.023 | 1471 |
| sdxl_turbo_1step | -0.077 | +0.012 | -0.079 | 1471 |
| sdxl_lightning_4steps | -0.071 | +0.061 | +0.001 | 1471 |
| sdxl_low_10steps | -0.066 | +0.055 | -0.078 | 1471 |
| sdxl_deepcache_50steps | -0.084 | +0.062 | -0.066 | 1471 |
| sdxl_ddim_50steps | -0.089 | +0.043 | -0.060 | 1471 |
| sdxl_high_50steps | -0.097 | +0.039 | -0.078 | 1471 |

Spearman 参考（同 7 cfg 全算，按 cfg 范围汇总）：

- clip：Spearman(all) Ridge +0.544~+0.627、KNN +0.383~+0.482、TFMLP +0.626~+0.673；Spearman(mid) Ridge +0.436~+0.500、KNN +0.270~+0.372、TFMLP +0.506~+0.529
- sharp：Spearman(all) Ridge +0.733~+0.802、KNN +0.590~+0.660、TFMLP +0.794~+0.819；Spearman(mid) Ridge +0.606~+0.701、KNN +0.473~+0.549、TFMLP +0.680~+0.716
- aesthetic：Spearman(all) Ridge +0.632~+0.710、KNN +0.502~+0.588、TFMLP +0.628~+0.715；Spearman(mid) Ridge +0.488~+0.574、KNN +0.373~+0.474、TFMLP +0.471~+0.581
- imagereward：Spearman(all) Ridge +0.490~+0.557、KNN +0.388~+0.454、TFMLP +0.463~+0.532；Spearman(mid) Ridge +0.375~+0.430、KNN +0.258~+0.330、TFMLP +0.339~+0.405

**解读**：

- **可预测性排序：sharp > aesthetic > clip > imagereward**。R²(all) 分别为 0.63~0.68 / 0.42~0.57 / 0.30~0.48 / 0.19~0.25（按 predictor 最好档），Spearman(all) 同序（sharp 最高 TFMLP 0.794~0.819，imagereward 最低 0.463~0.532）。全量训练（12869，论文口径）下四个指标的 predictor 都可学，排序信号均显著强于 R² 所显示的校准能力。
- **"中段有限"是三指标的共性，sharp 是唯一例外**：clip 的 R²(mid) 缩水到 −0.014~+0.138、imagereward 的 Ridge/TFMLP 中段几乎全负（Ridge −0.097~−0.036、TFMLP −0.079~+0.023）、aesthetic 也只有 −0.007~+0.188；而 sharp 的 R²(mid) 仍有 +0.202~+0.405（保留 R²(all) 的约六成），Spearman(mid) 0.61~0.72 几乎不缩水。sharpness 类底层视觉特征从中段样本里仍可稳定学习，语义对齐（clip）与偏好评分（imagereward）的中段区分则困难得多。
- **predictor 格局随指标变化**：R²(all) 上 TFMLP 在 clip/sharp 最好、aesthetic 上与 Ridge 互有胜负、imagereward 上 Ridge 反超（0.216~0.285 vs TFMLP 0.193~0.249）；KNN 的 R²(all) 恒最低，但 R²(mid) 上 KNN 在 aesthetic 全 7 cfg 最高（+0.092~+0.188）、在 imagereward 是唯一全正的 predictor——窄带保守预测在均方误差口径下占便宜，其 Spearman(mid) 实则各指标全最低（如 imagereward +0.258~+0.330）。
- 与 exp_815 结论差异：exp_815 的 R²(all) 虚高由 0/1 堆积贡献、中段全军覆没；本数据四指标均无堆积，R²(all) 是"真实的中等"。结合废弃的 2200 小训练量对照（旧结果见 `results/backup_2200/`：2200 时 clip 的 Ridge/TFMLP R²(mid) 大面积为负、full 时转正），**"R² 虚高"现象是 GenEval 式 0/1 指标特有的；本池表现为"小样本中段不可学、全量可学"**，与论文"predictor 可学"建立在全量 train 上的口径一致。

---

## 实验二：带路由（score 模式，四指标 × 6 个成员成本档）

做法：Lagrangian 路由 `argmax(pred − λ·cost)`，λ 网格照搬 exp_815（0~0.05 两千点细扫 + 至 1e6），quality@budget 线性插值，**六个 budget 档 = 路由池成员成本 1.5 / 1.54 / 23.92 / 119.7 / 210.0 / 598.5 TFLOPs**（predictor 的 test 预测只算一次，六档复用）。**四指标各自独立训 predictor、按同指标评估路由质量**；predictor 配置与实验一完全一致，预测不重复训练——Ridge/KNN（确定性）与 TFMLP seed=0 复用 [results/exp1_preds.json](results/exp1_preds.json)，clip 的 TFMLP seed1/2 复用 `results/backup_exp5/exp5_preds/tf_{1,2}.npy`（与旧实验二逐位核对过），sharp/aesthetic/imagereward 的 TFMLP seed1/2 按实验一配置现场补训、缓存到 `results/exp2_preds/`。Baseline：**Fixed**（该 budget 买得起的成员中该指标 test 均值最好者，表内标注成员名）、**LP**（train 均值上凸包混合，约束 **E[c] ≤ B 允许预算闲置**：budget < 峰值顶点成本时在上升段两档按 alpha=(B−c_lo)/(c_hi−c_lo) 随机混合，budget ≥ 峰值顶点成本时纯分配质量峰值顶点成员——sharp/imagereward 峰值为 infinity、clip 为 deepcache、aesthetic 为 lightning）、**OC**（train oracle 分布随机分配，超预算向最便宜档收缩）——LP/OC 框架从 exp_815 迁移，但 LP 在本复刻中于 2026-08-21 从旧实现的 E[c]=B（强制花满预算、高 budget 停在最贵 hull 顶点）修正为 E[c]≤B，旧实现下 sharp/imagereward/aesthetic 的 LP 随 budget 反常下降（旧结果备份在 `results/backup_lp_equality/`）；**Oracle** 为 true 路由上界。seeds [0,1,2] 报均值±标准差（Ridge/KNN 确定性，三次同值；LP 纯分配档无采样随机性）。复现校验：本节非 LP 字段与修正前版本逐值一致、LP 四指标 × 6 budget 单调不降（脚本内置 assert；更早的 clip-only 版备份在 `results/backup_exp2_cliponly/`）。

costs = [1.5, 1.54, 23.92, 119.7, 210.0, 598.5, 598.5]（与路由池 7 cfg 同序；ddim 与 high 同价 598.5）

### budget = 1.5

只有 infinity_1step 买得起，四指标所有方法均退化为 fixed infinity。

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2683 | 0.2683 (±0.0000) | 0.2683 (±0.0000) | 0.2683 (infinity_1step) | 0.2683 | 0.2683 | 0.2683 (±0.0000) |
| sharp | 0.1552 | 0.1552 (±0.0000) | 0.1552 (±0.0000) | 0.1552 (infinity_1step) | 0.1552 | 0.1552 | 0.1552 (±0.0000) |
| aesthetic | 5.6792 | 5.6792 (±0.0000) | 5.6792 (±0.0000) | 5.6792 (infinity_1step) | 5.6792 | 5.6792 | 5.6792 (±0.0000) |
| imagereward | 1.1115 | 1.1115 (±0.0000) | 1.1115 (±0.0000) | 1.1115 (infinity_1step) | 1.1115 | 1.1115 | 1.1115 (±0.0000) |

### budget = 1.54

clip：turbo 只比 infinity 贵 0.04 TFLOPs 但均值高 0.0089，Oracle 跳到 0.2835（每条 query 二选一取优），三 predictor 全部打平或略超 fixed turbo（0.2772）。其余三指标：**infinity 仍是最好成员**（贵成员均值反而更低），最好 fixed 不变，但 Oracle 仍明显高出 fixed——per-query 取优本身有增益。

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2835 | 0.2772 (±0.0000) | 0.2683 (±0.0000) | 0.2772 (sdxl_turbo_1step) | 0.2777 | 0.2775 | 0.2774 (±0.0001) |
| sharp | 0.1645 | 0.1552 (±0.0000) | 0.1552 (±0.0000) | 0.1552 (infinity_1step) | 0.1583 | 0.1566 | 0.1577 (±0.0004) |
| aesthetic | 5.7166 | 5.6792 (±0.0002) | 5.6792 (±0.0002) | 5.6792 (infinity_1step) | 5.6833 | 5.6805 | 5.6828 (±0.0007) |
| imagereward | 1.2694 | 1.1115 (±0.0000) | 1.1116 (±0.0001) | 1.1115 (infinity_1step) | 1.1185 | 1.1163 | 1.1177 (±0.0011) |

### budget = 23.92

lightning（23.92）进入预算：aesthetic 的最好 fixed 换成 lightning（5.7632）；clip 最好 fixed 仍是 turbo；sharp/imagereward 最好 fixed 仍是 infinity。**Ridge×imagereward 从本档起低于最好 fixed**（1.1052 < 1.1115）——花钱买了错误路由。

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2898 | 0.2778 (±0.0002) | 0.2696 (±0.0001) | 0.2772 (sdxl_turbo_1step) | 0.2796 | 0.2793 | 0.2791 (±0.0000) |
| sharp | 0.1718 | 0.1552 (±0.0000) | 0.1524 (±0.0002) | 0.1552 (infinity_1step) | 0.1615 | 0.1594 | 0.1607 (±0.0004) |
| aesthetic | 5.8612 | 5.7632 (±0.0000) | 5.6794 (±0.0022) | 5.7632 (sdxl_lightning_4steps) | 5.7761 | 5.7798 | 5.7804 (±0.0020) |
| imagereward | 1.3321 | 1.1115 (±0.0000) | 1.0885 (±0.0032) | 1.1115 (infinity_1step) | 1.1052 | 1.1165 | 1.1203 (±0.0021) |

### budget = 119.7

clip 最好 fixed 换成 low_10steps（0.2801）；其余三指标最好 fixed 不变。sharp/aesthetic/imagereward 的 predictor 曲线从本档起饱和（与 210/598.5 档基本同值）。

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2934 | 0.2801 (±0.0000) | 0.2739 (±0.0002) | 0.2801 (sdxl_low_10steps) | 0.2820 | 0.2815 | 0.2817 (±0.0001) |
| sharp | 0.1729 | 0.1552 (±0.0000) | 0.1443 (±0.0004) | 0.1552 (infinity_1step) | 0.1622 | 0.1596 | 0.1611 (±0.0005) |
| aesthetic | 5.8700 | 5.7632 (±0.0000) | 5.6857 (±0.0013) | 5.7632 (sdxl_lightning_4steps) | 5.7768 | 5.7798 | 5.7803 (±0.0019) |
| imagereward | 1.3478 | 1.1115 (±0.0000) | 1.0018 (±0.0043) | 1.1115 (infinity_1step) | 1.1025 | 1.1163 | 1.1223 (±0.0046) |

### budget = 210.0

clip 最好 fixed 换成 deepcache（0.2818），clip predictor 到本档基本到顶；三指标的 predictor 均已饱和。

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2940 | 0.2818 (±0.0000) | 0.2771 (±0.0006) | 0.2818 (sdxl_deepcache_50steps) | 0.2825 | 0.2824 | 0.2826 (±0.0002) |
| sharp | 0.1729 | 0.1552 (±0.0000) | 0.1443 (±0.0004) | 0.1552 (infinity_1step) | 0.1622 | 0.1596 | 0.1611 (±0.0005) |
| aesthetic | 5.8700 | 5.7632 (±0.0000) | 5.6857 (±0.0013) | 5.7632 (sdxl_lightning_4steps) | 5.7768 | 5.7798 | 5.7803 (±0.0019) |
| imagereward | 1.3478 | 1.1115 (±0.0000) | 1.0018 (±0.0043) | 1.1115 (infinity_1step) | 1.1025 | 1.1163 | 1.1198 (±0.0064) |

### budget = 598.5

ddim/high 同价 598.5（clip fixed 分别 0.2804 / 0.2808），但 clip 最好 fixed 仍是 210 档的 deepcache（0.2818）；Oracle 与 210 档持平（0.2940，无可再选）。三指标同样无新增收益；修正后的 LP 在本档全部纯分配峰值顶点成员，与最好 fixed 同值（clip 0.2818 / sharp 0.1552 / aesthetic 5.7632 / imagereward 1.1115）。

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2940 | 0.2818 (±0.0000) | 0.2771 (±0.0006) | 0.2818 (sdxl_deepcache_50steps) | 0.2826 | 0.2827 | 0.2826 (±0.0002) |
| sharp | 0.1729 | 0.1552 (±0.0000) | 0.1443 (±0.0004) | 0.1552 (infinity_1step) | 0.1622 | 0.1596 | 0.1611 (±0.0005) |
| aesthetic | 5.8700 | 5.7632 (±0.0000) | 5.6857 (±0.0013) | 5.7632 (sdxl_lightning_4steps) | 5.7768 | 5.7798 | 5.7803 (±0.0019) |
| imagereward | 1.3478 | 1.1115 (±0.0000) | 1.0018 (±0.0043) | 1.1115 (infinity_1step) | 1.1025 | 1.1163 | 1.1198 (±0.0064) |

### deferral curve（论文 Fig 3a 式，四指标 2×2 面板）

budget 从 1.5 到 598.5 几何取 28 点，quality-cost 曲线（x 轴 log），每指标一面板（实线 = Oracle / Ridge / KNN / TFMLP，虚线 = LP / OC，红点 = 7 个 fixed 成员）：

![deferral_full](figs/exp2_deferral_full.png)

代表预算点数值（clip 主指标）：

| budget | Oracle | Ridge | KNN | TFMLP | LP | OC |
|---|---|---|---|---|---|---|
| 1.50 | 0.2683 | 0.2683 | 0.2683 | 0.2683 | 0.2683 | 0.2683 |
| 7.09 | 0.2867 | 0.2784 | 0.2782 | 0.2780 | 0.2774 | 0.2686 |
| 26.82 | 0.2901 | 0.2798 | 0.2795 | 0.2793 | 0.2779 | 0.2697 |
| 126.69 | 0.2935 | 0.2820 | 0.2816 | 0.2817 | 0.2802 | 0.2742 |
| 479.44 | 0.2940 | 0.2826 | 0.2827 | 0.2826 | 0.2818 | 0.2771 |
| 598.50 | 0.2940 | 0.2826 | 0.2827 | 0.2826 | 0.2818 | 0.2771 |

四指标端点摘要（@1.5 全员退化；@598.5 列最好 predictor 并括注方法名）：

| 指标 | Oracle @1.5→@598.5 | 最好 predictor @598.5 | 最好 fixed @598.5 | LP @598.5 | OC @598.5 |
|---|---|---|---|---|---|
| clip | 0.2683 → 0.2940 | 0.2827 (KNN) | 0.2818 (deepcache) | 0.2818 | 0.2771 |
| sharp | 0.1552 → 0.1729 | 0.1622 (Ridge) | 0.1552 (infinity) | 0.1552 | 0.1443 |
| aesthetic | 5.6792 → 5.8700 | 5.7803 (TFMLP) | 5.7632 (lightning) | 5.7632 | 5.6857 |
| imagereward | 1.1115 → 1.3478 | 1.1198 (TFMLP) | 1.1115 (infinity) | 1.1115 | 1.0018 |

**结论**：

- **budget=1.5 全员退化**：只有 infinity 买得起，四指标所有方法 = fixed infinity，无需比较。
- **clip 行与旧 clip-only 版结论完全一致**（逐值 assert 通过）：1.54 档三 predictor 打平或略超 fixed turbo（0.2772，旧 2200 协议下 Ridge/TFMLP 曾在此档反亏，见 backup）；≥23.92 起稳定超过最好 fixed 与 LP/OC；最好档（598.5）增益 +0.0009（KNN 0.2827 vs deepcache 0.2818），Oracle gap ~0.011——**clip 上"Ours > Fixed"成立但增益很小**，瓶颈是 predictor 精度（实验一：中段预测力有限）。
- **三指标的成本-质量关系与 clip 相反**：sharp/imagereward 上最便宜的 infinity 就是最好成员（0.1552 / 1.1115，贵成员全线更差），aesthetic 上 23.92 的 lightning 最好（5.7632）、infinity 次之（5.6792）——"成本↑ ⇒ 质量↑"只在 clip 上成立。直接后果：修正后的 LP 在这些指标上几乎不花钱（停在峰值顶点、与最好 fixed 同值），额外 budget 对 LP 零增益；OC 不优化质量（只把 train oracle 分布按预算向下压），在三指标上随 budget 降到渐近值后持平（sharp 0.1552→0.1443、imagereward 1.1115→1.0018）。
- **LP 修正（E[c]=B → E[c]≤B）改变高 budget 档的 predictor−LP 格局**：旧实现强制花满预算，在三指标上 LP 越花越差（旧值备份在 `results/backup_lp_equality/`：sharp LP@598.5 = 0.1325、imagereward 0.9634、aesthetic 5.6058）；修正后 LP 单调不降（脚本内置 assert：四指标 × 6 budget 单调性 + 峰值纯分配 == 成员 test 均值，全过），budget ≥ 峰值顶点成本即纯分配峰值成员、与最好 fixed 同值（clip 自 210 档起 LP=0.2818=deepcache，sharp/imagereward 全程 = infinity 0.1552/1.1115，aesthetic 自 23.92 档起 = lightning 5.7632）。**于是高 budget 档的 predictor−LP 差收敛为 predictor−最好fixed 差**（clip +0.0009、sharp +0.0070、aesthetic +0.0171、imagereward +0.0083），不再像旧实现那样被 LP 的虚低放大；中段 budget（上升段混合区）LP 数值与修正前逐位一致。"predictor > LP"的成立范围与"predictor > 最好 fixed"完全重合（同样只有 Ridge×imagereward 四格例外）。
- **815 不受此 bug 影响（已核实）**：exp_815 的 LP 是同一等成本混合实现，但其 budgets=[5,10,20] 全部落在 hull 上升段（gm/am/unigen 三数据集的 hull 峰值顶点均为 boogu_50steps@21.5，没有任何 budget 档超过它），E[c]=B 与 E[c]≤B 在上升段等价，故 815 的结果无需修正（其池本身也并非成本单调——如 flux_50steps@23.1 差于 boogu_50steps@21.5——只是 budget 从未触及峰值顶点之上）。本次只修 catimage 复刻，不动 815。
- **predictor 路由在四指标上普遍超过最好 fixed**：非退化 60 格（5 budget × 4 指标 × 3 predictor）中 56 格胜出，增益比 clip 大一个量级——sharp 最好 Ridge +0.0070（0.1622 vs 0.1552）、aesthetic 最好 TFMLP +0.0171（5.7803 vs 5.7632）、imagereward 最好 TFMLP +0.0107（1.1223 vs 1.1115）。**唯一例外是 Ridge×imagereward**：budget ≥ 23.92 后 4 格全部低于 fixed infinity（1.1025 vs 1.1115），KNN/TFMLP 无此问题——与实验一"imagereward 最难预测、Ridge 中段 R² 为负"互洽。
- **Oracle gap 与实验一可预测性排序互为印证**：@598.5 的 Oracle − 最好 predictor 为 clip 0.011 / sharp 0.011 / aesthetic 0.090 / imagereward 0.228；predictor 实际填上 Oracle−最好fixed 空间的比例为 clip 7%、sharp 40%、aesthetic 16%、imagereward 3.5%——填得最多的是 sharp（实验一中 R²(mid) 唯一不塌的指标），填得最少的是 imagereward（可预测性最差）。**路由收益空间本身（Oracle − 最好 fixed）排序为 imagereward 0.236 > aesthetic 0.107 > sharp 0.018 > clip 0.012**，与"贵即好"的直觉完全脱钩。
- **曲线形态**：clip 上 Oracle 与 predictor 随 budget 单调爬升（成本-质量正相关，210 到顶）；三指标上 predictor 曲线低 budget 段即跳起、119.7 饱和（aesthetic 23.92 即饱和），更高 budget 纯浪费——fixed 散点横排甚至下行（sharp/imagereward 的 598.5 成员垫底）；修正后的 LP 曲线全程单调不降、高 budget 段与最好 fixed 点重合（三指标为平线、clip 升到 210 后持平），OC 在三指标上降到渐近值后平躺。
- **7 个 fixed 点在 clip 上几乎排成水平线**（0.2683~0.2818，cross 400 倍成本只差 0.0135），这是本池子的本质特征；但三指标上成员均值差大得多（imagereward 最好 1.1115 vs 最差 0.6199），只是方向不再随成本单调——**路由收益空间小是 clip 特有的，predictor 精度不足才是四指标共性**。

---

## 实验三：掐头去尾（budget=598.5）

**动机**：与 exp_815 实验三同构——把两端极端样本从训练集去掉、让 predictor 专注中段，路由是否变好。815 在 budget=10（anchor=qwen_50steps 成本）上跑；本复现统一用 **budget=598.5**（= sdxl_ddim_50steps/sdxl_high_50steps 成本，全池买得起、路由空间最大的一档，即实验二 predictor 增益最好所在档），后同。

**设计**：**仅主指标 clip（与 815 同口径）**。极端样本定义需适配——catimage 的 clip 无 0/1 堆积，直接把 815 的绝对阈值（≤0.05/≥0.95）打在原始 clip（范围 ~[0.14,0.40]）上会得到"全部样本都是中段"。按论文口径把标签**逐 cfg min-max 缩放到 [0,1]**（lo/hi 取 train pool 12869 的逐 cfg 最小/最大）后再用同样阈值：任一 cfg 缩放后 ≤0.05 或 ≥0.95 即极端样本。去除比例 X ∈ {0%, 20%, 40%, 60%, 80%, 100%}，用 `np.random.default_rng(seed)` 随机移除极端样本中的 X%。Ridge/KNN/TFMLP 在裁剪子集上重训（Ridge 的 α 选择、TFMLP 的早停仍用不裁剪的 val 1838；TFMLP 的 min-max lo/hi 用裁剪后训练集），test 1839 不变。baselines（Oracle/LP/OC/最好 fixed）用完整训练集、固定为实验二 budget=598.5 的值。seeds [0,1,2] 报均值±标准差（移除采样与 TFMLP 都按 seed）。frac=0 行与实验二 budget=598.5 的 Ridge/KNN/TFMLP 逐位一致（复现校验通过，脚本内置 assert）。

**n_extreme = 27 / 12869（0.2%）**——min-max 缩放到 [0,1] 后，单峰近似正态的 clip 分布只有极少数样本落在距两端 5% 量程以内。

baselines（固定）：Oracle=0.2940，LP=0.2818，OC=0.2771，最好 fixed=0.2818（sdxl_deepcache_50steps）（LP 为 E[c]≤B 修正版：budget≥210 纯分配 deepcache，与最好 fixed 同值）

| 去除比例 | n_train | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| 0% | 12869 | 0.2826 (±0.0000) | 0.2827 (±0.0000) | 0.2826 (±0.0002) |
| 20% | 12864 | 0.2827 (±0.0000) | 0.2827 (±0.0000) | 0.2828 (±0.0000) |
| 40% | 12858 | 0.2826 (±0.0000) | 0.2827 (±0.0000) | 0.2826 (±0.0001) |
| 60% | 12853 | 0.2827 (±0.0001) | 0.2827 (±0.0000) | 0.2826 (±0.0001) |
| 80% | 12847 | 0.2827 (±0.0000) | 0.2827 (±0.0000) | 0.2828 (±0.0001) |
| 100% | 12842 | 0.2827 (±0.0000) | 0.2827 (±0.0000) | 0.2825 (±0.0003) |

![exp3_trim](figs/exp3_trim.png)

**结论**：掐头去尾**没有带来路由提升，也没有伤害**——全程在 ±0.0003 种子噪声内波动（Ridge 0.2826~0.2827、KNN 恒 0.2827、TFMLP 0.2825~0.2828）。

- 与 815 的定性结论同向（"掐头去尾无提升"），但机制不同：815 的 GM/UniGen 极端样本占 63~81%，去掉后路由单调变差（极端样本携带路由信号）；本数据按同一阈值口径**根本没有两端堆积可去**——n_extreme 只有 27 条（0.2%），100% 去除也只动 27/12869 条训练样本，实验近乎恒等。
- 这是实验一分布结论的直接推论：CLIPScore 单峰、无 0/1 极端群体，"掐头去尾"这个在 GenEval 式指标上有操作空间的手段，在 CATImage 池上从定义起就不起作用。

---

## 实验四：训练标签 seed 数消融（budget=598.5）

思路与实验三同构（也与 815 实验四同构），**仅主指标 clip（与 815 同口径）**，改的是**训练标签的生成 seed 数**：现状标签是 4 个生成 seed 的均值（对应论文 S=4）；本实验让 train/val 标签只用前 k 个可用 seed 的均值，k ∈ {1,2,3,4}（k=4 = baseline；个别 cell 缺某个 seed 时取可用的前 k 个，全池仅 ddim 缺 7 格、high 缺 2 格）。test 标签不变（4-seed 均值作真值）。baselines 固定为实验二 budget=598.5 的数值。k=4 行与实验二 budget=598.5 的 Ridge/KNN/TFMLP 逐位一致（k=4 标签直接用 common.load_split 的 float32 数组，与实验二逐位同源；复现校验通过，脚本内置 assert）。

n_train = 12869；baselines（固定）：Oracle=0.2940，LP=0.2818，OC=0.2771，最好 fixed=0.2818（sdxl_deepcache_50steps）（LP 为 E[c]≤B 修正版：budget≥210 纯分配 deepcache，与最好 fixed 同值）

| k | Ridge | KNN | TFMLP |
|---|---|---|---|
| 1 | 0.2817 (±0.0000) | 0.2824 (±0.0000) | 0.2817 (±0.0007) |
| 2 | 0.2819 (±0.0000) | 0.2825 (±0.0000) | 0.2819 (±0.0003) |
| 3 | 0.2823 (±0.0000) | 0.2824 (±0.0000) | 0.2825 (±0.0004) |
| 4 | 0.2826 (±0.0000) | 0.2827 (±0.0000) | 0.2826 (±0.0002) |

![exp4_seeds](figs/exp4_seeds.png)

**结论**：单 seed 标签的噪声确实伤路由，但程度很小且主要在 Ridge/TFMLP 上——

- **Ridge** 0.2817→0.2826（+0.0009，k=3 后基本收敛）；**TFMLP** 0.2817→0.2826（+0.0009，同步收敛且种子噪声最大 ±0.0007）；**KNN** 几乎不动（0.2824~0.2827）——邻居均值本身已把 seed 噪声摊薄。
- 与 815 定性一致（单 seed 伤 Ridge、k≈3 收敛、KNN 稳），但量级差一个数量级：815 GM Ridge 单 seed 代价 0.024，这里只有 0.0009——clip 标签的 seed 间噪声小（均值统计误差 ~±0.0007），且本池路由增益空间总共只有 0.012，标签噪声可侵蚀的部分本来就少。
- **k=3 即收敛**，从 k=3 到 k=4 的增益 ≤0.0003；论文取 S=4 已足够。

---

## 验证节：fixed 成员 test clip vs 论文 Table 1 Fixed 行

test n=1839，指标 clip。对应关系：sdxl_low_10steps↔SDXL-9（我们 10 步 vs 论文 9 步）、sdxl_high_50steps↔SDXL-65（我们 50 步 vs 论文 65 步）；论文 SDXL-22（0.2805）我们无对应档。

| cfg | 论文成员 | 论文 Fixed | 本复现 test clip | 差值 | 备注 |
|---|---|---|---|---|---|
| infinity_1step | Infinity | 0.2816 | 0.2683 | **-0.0133** | |
| sdxl_turbo_1step | Turbo | 0.2794 | 0.2772 | -0.0022 | |
| sdxl_lightning_4steps | Lightning | 0.2798 | 0.2723 | -0.0075 | |
| sdxl_low_10steps | SDXL-9 | 0.2773 | 0.2801 | +0.0028 | steps 10 vs 9 |
| sdxl_deepcache_50steps | DeepCache | 0.2749 | 0.2818 | **+0.0069** | |
| sdxl_ddim_50steps | DDIM | 0.2820 | 0.2804 | -0.0016 | |
| sdxl_high_50steps | SDXL-65 | 0.2819 | 0.2808 | -0.0011 | steps 50 vs 65 |
| （无对应） | SDXL-22 | 0.2805 | — | — | |

**结论：大致正确、有系统性小偏差。**

- 7 个成员中 5 个差值在 ±0.008 以内（turbo/lightning/low/ddim/high），方向有正有负，均值 −0.0023——整体水平与论文 Table 1 一致。
- 两个例外：**infinity 偏低 0.0133**（0.2683 vs 0.2816）和 **deepcache 偏高 0.0069**（0.2818 vs 0.2749）。这两个偏差改变了池内排序：论文里 DeepCache 是 Fixed 最差（0.2749）、Infinity 居中；本复现里 infinity 最差（0.2683）、deepcache 最好（0.2818）。
- 偏差来源讨论：① prompt 子集不同——论文用完整 COCO 评测集，我们是 coco_test.json 1839 条；② 评测实现差异（CLIP 模型版本、图像分辨率、seed 数 S=4 的抽样噪声，clip 均值 σ/√(S·N) 量级的统计误差约 ±0.0007，不足以解释 0.013）；③ infinity 的生成配置（论文 vs 本地 ckpt/推理参数）差异最可能放大到 0.01 量级。排序差异（deepcache 最优 vs 最差）会直接影响路由实验的基线强度，阅读实验二数字时应注意。

---

## 产物清单

- 代码：`common.py`（7-cfg 加载/论文 80/10/10 划分/特征/R²+Spearman 指标）、`embed_coco.py`（Qwen3-Embedding-8B 现算并缓存到 codes/.cache_embed）、`run_exp1_predict.py`（四指标，per_metric 结构）、`plot_exp1.py`（四指标出图/表）、`run_exp2_routing.py`、`plot_exp2.py`、`run_exp3_trim.py`、`run_exp4_seeds.py`、`plot_exp345.py`、`validate_fixed.py`
- 结果：[results/exp1_r2.json](results/exp1_r2.json)、`exp1_preds.json`、`exp1_tables.md`、`exp2_routing.json`、`exp2_routing.md`、`exp2_deferral.json`、`validation.json`、`exp3_trim.json`、`exp3_trim.md`、`exp4_seeds.json`、`exp4_seeds.md`、`exp2_preds/{metric}_{seed}.npy`（sharp/aesthetic/imagereward 的 TFMLP seed1/2 补训缓存；clip 三 seed 复用 backup_exp5/exp5_preds，Ridge/KNN 复用 exp1_preds.json）
- 旧协议备份（train 2200 / val 200）：`results/backup_2200/`（含当时全部 results json/md 及两张 2200 图）
- 下线备份：`results/backup_exp5/`（实验五「Oracle gap 反事实分解」的全部代码、结果 json/md、base 预测缓存 exp5_preds/、图与日志；因 min-max 阈值口径下极端集为空、实验退化而下线，备查）
- 旧版备份：`results/backup_exp2_cliponly/`（实验二四指标扩展前的 clip-only 版 exp2_routing.json/md 与 exp2_deferral.json；新文件 clip 块已 assert 与其逐值一致）、`results/backup_appendix_knn/`（已删除的「附录：KNN×四指标路由」的结果 json，其内容已被实验二四指标主表覆盖）、`results/backup_lp_equality/`（LP 旧实现 E[c]=B 等成本混合版的 exp2 三件套与 deferral 图，附 README 注明下线原因；其中 predictor/Oracle/OC/fixed 数字与现行版逐值一致，仅 LP 列作废）
- 图：`figs/exp1_dist_{clip,sharp,aesthetic,imagereward}.png`、`figs/exp1_scatter_{clip,sharp,aesthetic,imagereward}.png`、[figs/exp2_deferral_full.png](figs/exp2_deferral_full.png)（四指标 2×2 面板）、[figs/exp3_trim.png](figs/exp3_trim.png)、[figs/exp4_seeds.png](figs/exp4_seeds.png)
- 日志：`logs/embed_coco.log`、`logs/exp1.log`、`logs/exp1_metrics.log`、`logs/exp2.log`、`logs/exp2_metrics.log`、`logs/exp2_lpfix.log`、`logs/plot_exp1.log`、`logs/plot_exp2.log`、`logs/validate.log`、`logs/exp3_trim.log`、`logs/exp4_seeds.log`
- 缓存（复用既有目录）：`codes/.cache_embed/emb_{md5}.npy`（coco_train/test 两条缓存）、`codes/.cache_clip/cliptok_*`（expcat818_train_n14707 / expcat818_test_n1839）
