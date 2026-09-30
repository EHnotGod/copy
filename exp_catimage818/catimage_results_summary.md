# CATImage 复刻实验结果汇总（exp_catimage818）

> 单独抽取自 `exp_catimage818/exp_catimage818.md`，覆盖截至 2026-08-21 的全部现行结果（含 LP 修正版）。
> 详细口径、产物清单与备份说明见原文档。

## 0. 实验设置

- **数据**：COCO CATImage（论文 2506.14753v3 Sec 5.2）。train 12869 / val 1838 / test 1839（论文 val 分片无评分，val 从论文 train 14707 中固定 seed=42 切出；test 1839 为论文 test 分片，不动）。
- **路由池 7 cfg**（CATImage 论文 Table 1 成员，成本单位 TFLOPs）：
  `infinity_1step(1.5) / sdxl_turbo_1step(1.54) / sdxl_lightning_4steps(23.92) / sdxl_low_10steps(119.7) / sdxl_deepcache_50steps(210.0) / sdxl_ddim_50steps(598.5) / sdxl_high_50steps(598.5)`
- **标签**：7 模型 × 4 指标（clip / sharp / aesthetic / imagereward）全量无缺失；per-prompt 标签 = 4 个生成 seed 均值（论文 S=4）。
- **Predictor**：Ridge（Qwen3-Embedding-8B 4096d，α 按 val R² 逐 cfg 选）、KNN（CLIP mean-pool，k=50）、TFMLP（CATImage Appendix D：CLIP per-token → 2 层 self-attn + per-token MLP）。seeds [0,1,2]。
- **score 模式**：直接预测绝对分；四指标各训各的 predictor（论文口径）。
- **路由**：Lagrangian `argmax(pred − λ·cost)`，quality@budget 线性插值；6 个 budget 档 = 池成员成本（1.5 / 1.54 / 23.92 / 119.7 / 210.0 / 598.5）。
- **Baselines**：Fixed（预算内 test 均值最好成员）、**LP（E[c]≤B 修正版：允许预算闲置，budget ≥ 峰值顶点成本时纯分配质量峰值成员）**、OC（train oracle 分布随机分配）、Oracle（true 路由上界）。

---

## 1. 实验一：绝对分直接预测

**核心结论：四指标分布均为单峰、无 0/1 两端堆积；R²(all) 中等，但中段（y∈(q10,q90)，n=1471）大幅缩水——除 sharp 外是共性。**

### R²(all)（全 test，各指标取 7 cfg 范围）

| 指标 | Ridge | KNN | TFMLP |
|---|---|---|---|
| clip | +0.297~+0.409 | +0.150~+0.227 | +0.402~+0.476 |
| sharp | +0.548~+0.637 | +0.281~+0.399 | +0.633~+0.675 |
| aesthetic | +0.432~+0.560 | +0.250~+0.361 | +0.421~+0.569 |
| imagereward | +0.216~+0.285 | +0.141~+0.179 | +0.193~+0.249 |

### R²(mid)（y_true ∈ (q10, q90) 子集）

| 指标 | Ridge | KNN | TFMLP |
|---|---|---|---|
| clip | −0.014~+0.098 | +0.040~+0.129 | +0.070~+0.138 |
| sharp | +0.202~+0.386 | +0.211~+0.297 | +0.285~+0.405 |
| aesthetic | +0.015~+0.098 | +0.092~+0.188 | −0.007~+0.145 |
| imagereward | −0.097~−0.036 | +0.012~+0.068 | −0.079~+0.023 |

Spearman（all / mid，按最好 predictor 档）：

- clip：TFMLP 0.626~0.673 / 0.506~0.529
- sharp：TFMLP 0.794~0.819 / 0.680~0.716
- aesthetic：TFMLP 0.628~0.715 / 0.471~0.581
- imagereward：Ridge 0.490~0.557 / 0.375~0.430

**解读**：

1. **可预测性排序：sharp > aesthetic > clip > imagereward**（R² 与 Spearman 同序）。
2. **"中段塌陷"是 clip / aesthetic / imagereward 的共性，sharp 是唯一例外**（R²(mid) 保留 R²(all) 约六成，Spearman(mid) 几乎不缩水）——底层视觉特征可从 prompt 稳定学习，语义对齐与偏好评分的中段区分困难。
3. 排序信号（Spearman）在各指标上均显著强于 R² 所显示的校准能力。
4. 与 exp_815 结论差异：815 的 R²(all) 虚高来自 GenEval 式 0/1 堆积；本数据无堆积，R²(all) 是"真实的中等"。结合旧 2200 小训练量对照（`results/backup_2200/`），本池表现为**"小样本中段不可学、全量可学"**。

图：`figs/exp1_dist_{clip,sharp,aesthetic,imagereward}.png`（分布）、`figs/exp1_scatter_*.png`（拟合散点）。

---

## 2. 实验二：路由（四指标 × 6 budget）

### 各 budget 档数值（mean，±std 见原文档）

**budget = 1.5**：只有 infinity 买得起，全员退化 = fixed infinity。

**budget = 1.54**：

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2835 | 0.2772 | 0.2683 | 0.2772 (turbo) | 0.2777 | 0.2775 | 0.2774 |
| sharp | 0.1645 | 0.1552 | 0.1552 | 0.1552 (infinity) | 0.1583 | 0.1566 | 0.1577 |
| aesthetic | 5.7166 | 5.6792 | 5.6792 | 5.6792 (infinity) | 5.6833 | 5.6805 | 5.6828 |
| imagereward | 1.2694 | 1.1115 | 1.1116 | 1.1115 (infinity) | 1.1185 | 1.1163 | 1.1177 |

**budget = 23.92**：

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2898 | 0.2778 | 0.2696 | 0.2772 (turbo) | 0.2796 | 0.2793 | 0.2791 |
| sharp | 0.1718 | 0.1552 | 0.1524 | 0.1552 (infinity) | 0.1615 | 0.1594 | 0.1607 |
| aesthetic | 5.8612 | 5.7632 | 5.6794 | 5.7632 (lightning) | 5.7761 | 5.7798 | 5.7804 |
| imagereward | 1.3321 | 1.1115 | 1.0885 | 1.1115 (infinity) | **1.1052** | 1.1165 | 1.1203 |

**budget = 119.7**：

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2934 | 0.2801 | 0.2739 | 0.2801 (low_10steps) | 0.2820 | 0.2815 | 0.2817 |
| sharp | 0.1729 | 0.1552 | 0.1443 | 0.1552 (infinity) | 0.1622 | 0.1596 | 0.1611 |
| aesthetic | 5.8700 | 5.7632 | 5.6857 | 5.7632 (lightning) | 5.7768 | 5.7798 | 5.7803 |
| imagereward | 1.3478 | 1.1115 | 1.0018 | 1.1115 (infinity) | **1.1025** | 1.1163 | 1.1223 |

**budget = 210.0 / 598.5**（predictor 已饱和，两档同值）：

| 指标 | Oracle | LP | OC | 最好 fixed | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|---|---|
| clip | 0.2940 | 0.2818 | 0.2771 | 0.2818 (deepcache) | 0.2826 | 0.2827 | 0.2826 |
| sharp | 0.1729 | 0.1552 | 0.1443 | 0.1552 (infinity) | 0.1622 | 0.1596 | 0.1611 |
| aesthetic | 5.8700 | 5.7632 | 5.6857 | 5.7632 (lightning) | 5.7768 | 5.7798 | 5.7803 |
| imagereward | 1.3478 | 1.1115 | 1.0018 | 1.1115 (infinity) | **1.1025** | 1.1163 | 1.1198 |

加粗 = Ridge×imagereward 在 budget ≥ 23.92 后 4 格全部**低于最好 fixed**（花钱买了错误路由）——非退化 60 格中仅有的例外，与实验一"imagereward 最难预测"互洽。

### 端点摘要（@598.5）

| 指标 | Oracle | 最好 predictor | 最好 fixed / LP | Oracle gap 填补率 |
|---|---|---|---|---|
| clip | 0.2940 | 0.2827 (KNN) | 0.2818 (deepcache) | **7%** |
| sharp | 0.1729 | 0.1622 (Ridge) | 0.1552 (infinity) | **40%** |
| aesthetic | 5.8700 | 5.7803 (TFMLP) | 5.7632 (lightning) | **16%** |
| imagereward | 1.3478 | 1.1198 (TFMLP) | 1.1115 (infinity) | **3.5%** |

**解读**：

1. **Predictor 路由普遍超过最好 fixed**（56/60 格），增益比 clip 大一个量级：sharp +0.0070、aesthetic +0.0171、imagereward +0.0107；clip 仅 +0.0009。
2. **Oracle gap 填补率与实验一可预测性排序互为印证**：填得最多的是 sharp（唯一中段不塌的指标），最少的是 imagereward。
3. **路由收益空间（Oracle − 最好 fixed）排序为 imagereward 0.236 > aesthetic 0.107 > sharp 0.018 > clip 0.012**，与"贵即好"的直觉完全脱钩——sharp/imagereward 上最便宜的 infinity 就是最好成员，aesthetic 上 23.92 的 lightning 最好，"成本↑⇒质量↑"只在 clip 上成立。
4. clip 上 7 个 fixed 点几乎水平（跨 400 倍成本只差 0.0135）——**路由收益空间小是 clip 特有，predictor 精度不足才是四指标共性**。
5. **LP 修正（E[c]=B → E[c]≤B）后单调不降**（脚本内置 assert 全过），高 budget 档 LP = 最好 fixed；predictor−LP 差不再被旧实现的虚低放大。旧值备份 `results/backup_lp_equality/`；exp_815 不受影响（其 budget 全在 hull 上升段）。

图：[figs/exp2_deferral_full.png](figs/exp2_deferral_full.png)（四指标 2×2 面板 deferral curve）。

---

## 3. 实验三：掐头去尾（budget=598.5，仅 clip）

min-max 缩放后按 ≤0.05/≥0.95 阈值定义极端样本：**n_extreme = 27 / 12869（0.2%）**。

| 去除比例 | Ridge | KNN | TFMLP |
|---|---|---|---|
| 0% | 0.2826 | 0.2827 | 0.2826 |
| 100% | 0.2827 | 0.2827 | 0.2825 |

**结论：无提升也无伤害**（全程 ±0.0003 种子噪声内）。与 815 定性同向但机制不同：815 的极端样本占 63~81%、去掉伤路由；本数据 clip 单峰无堆积，"掐头去尾"从定义起就不起作用。图：[figs/exp3_trim.png](figs/exp3_trim.png)。

---

## 4. 实验四：训练标签 seed 数消融（budget=598.5，仅 clip）

train/val 标签只用前 k 个生成 seed 的均值，test 标签不变（4-seed）：

| k | Ridge | KNN | TFMLP |
|---|---|---|---|
| 1 | 0.2817 | 0.2824 | 0.2817 |
| 2 | 0.2819 | 0.2825 | 0.2819 |
| 3 | 0.2823 | 0.2824 | 0.2825 |
| 4 | 0.2826 | 0.2827 | 0.2826 |

**结论**：单 seed 标签确实伤路由，但代价很小（Ridge/TFMLP +0.0009，比 815 GM 的 0.024 低一个数量级——clip 标签 seed 间噪声小、且路由增益空间总共只有 0.012）；**k=3 即收敛**，论文 S=4 足够；KNN 几乎不动（邻居均值已摊薄噪声）。图：[figs/exp4_seeds.png](figs/exp4_seeds.png)。

---

## 5. 验证节：fixed 成员 test clip vs 论文 Table 1

| cfg | 论文成员 | 论文 Fixed | 本复现 | 差值 |
|---|---|---|---|---|
| infinity_1step | Infinity | 0.2816 | 0.2683 | **−0.0133** |
| sdxl_turbo_1step | Turbo | 0.2794 | 0.2772 | −0.0022 |
| sdxl_lightning_4steps | Lightning | 0.2798 | 0.2723 | −0.0075 |
| sdxl_low_10steps | SDXL-9 | 0.2773 | 0.2801 | +0.0028 |
| sdxl_deepcache_50steps | DeepCache | 0.2749 | 0.2818 | **+0.0069** |
| sdxl_ddim_50steps | DDIM | 0.2820 | 0.2804 | −0.0016 |
| sdxl_high_50steps | SDXL-65 | 0.2819 | 0.2808 | −0.0011 |

**结论：大致正确、有系统性小偏差。** 5/7 成员差值在 ±0.008 内；两个例外（infinity 偏低 0.0133、deepcache 偏高 0.0069）改变了池内排序——论文里 DeepCache 最差、Infinity 居中；本复现里 infinity 最差、deepcache 最好。阅读实验二数字时应注意此排序差异。

---

## 6. 对论文主线的支撑点

- **绝对预测"看似可学"但实例级有效性有限**：R²(all) 中等（0.19~0.68），R²(mid) 除 sharp 外塌陷或为负——全量 R² 主要由分布两端贡献，中段（真正需要细粒度路由决策的区间）近乎不可学。
- **路由收益 ≠ 准确预测**：predictor 路由虽普遍超过最好 fixed，但 Oracle gap 填补率仅 3.5%~40%，且填补率与 R²(mid) 排序严格对应——收益大小由"可预测的信号有多少"决定，而非路由机制本身。
- **成本-质量非单调**：7 成员跨 400 倍成本，clip 上质量只差 0.0135；sharp/imagereward 上最便宜成员最好。这对 CATImage 式"给复杂 prompt 分贵配置"的直觉是直接反例。
- **LP 基线修正的教训**：E[c]=B 等成本混合在非单调池中不是 feasible optimum，会虚低基线、夸大 predictor 优势；审计类论文必须用 E[c]≤B 形式。
