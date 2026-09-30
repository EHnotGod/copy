# 实验五：Oracle gap 的反事实分解（COCO CATImage，budget=598.5）

极端样本 = test 真实 anchor（sdxl_high_50steps）clip 按 train pool min-max 缩放后 ≤0.05 或 ≥0.95；
CF-extreme/CF-mid = 极端/中段样本的预测换成真值后的路由质量（同 λ 扫描口径）。
gap 分解百分比 = (CF − base) / (Oracle − base)。

n_test = 1839，极端样本 0 条（0.0%），中段 1839 条；
Oracle=0.2940，最好 fixed=0.2818（sdxl_deepcache_50steps）

| 路由方式 | Ridge | KNN | TFMLP |
|---|---|---|---|
| 最好 fixed | 0.2818 | 0.2818 | 0.2818 |
| base（全预测） | 0.2826 (±0.0000) | 0.2827 (±0.0000) | 0.2826 (±0.0002) |
| CF-extreme（极端给真值） | 0.2826 (±0.0000) | 0.2827 (±0.0000) | 0.2826 (±0.0002) |
| CF-mid（中段给真值） | 0.2940 (±0.0000) | 0.2940 (±0.0000) | 0.2940 (±0.0000) |
| Oracle | 0.2940 | 0.2940 | 0.2940 |

gap 分解（占 Oracle−base 的百分比）：

| CF | Ridge | KNN | TFMLP |
|---|---|---|---|
| CF-extreme | 0.0% | 0.0% | 0.0% |
| CF-mid | 100.0% | 100.0% | 100.0% |

极端/中段样本上各档真实质量均值：

| 子集 | infinity_1step | sdxl_turbo_1step | sdxl_lightning_4steps | sdxl_low_10steps | sdxl_deepcache_50steps | sdxl_ddim_50steps | sdxl_high_50steps |
|---|---|---|---|---|---|---|---|
| extreme | — | — | — | — | — | — | — |
| mid | 0.2683 | 0.2772 | 0.2723 | 0.2801 | 0.2818 | 0.2804 | 0.2808 |
