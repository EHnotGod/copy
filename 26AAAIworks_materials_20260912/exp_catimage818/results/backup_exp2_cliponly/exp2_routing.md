# 实验二：带路由（COCO CATImage，clip 主指标）

cfg = ['infinity_1step', 'sdxl_turbo_1step', 'sdxl_lightning_4steps', 'sdxl_low_10steps', 'sdxl_deepcache_50steps', 'sdxl_ddim_50steps', 'sdxl_high_50steps']
costs = [1.5, 1.54, 23.92, 119.7, 210.0, 598.5, 598.5]

budget = 各成员 cost（论文 Table 1 式）；Fixed = 该成本成员的 test clip 均值。
598.5 同时对应 sdxl_ddim_50 / sdxl_high_50 两行（Fixed 不同）。
Ridge/KNN 确定性，TFMLP/LP/OC 为 seeds [0,1,2] 均值±std。

## train_size = full（n_train=12869）

### budget = 1.5

Fixed: infinity_1step=0.2683

| Oracle | LP | OC | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|
| 0.2683 | 0.2683 (±0.0000) | 0.2683 (±0.0000) | 0.2683 | 0.2683 | 0.2683 (±0.0000) |

### budget = 1.54

Fixed: sdxl_turbo_1step=0.2772

| Oracle | LP | OC | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|
| 0.2835 | 0.2772 (±0.0000) | 0.2683 (±0.0000) | 0.2777 | 0.2775 | 0.2774 (±0.0001) |

### budget = 23.92

Fixed: sdxl_lightning_4steps=0.2723

| Oracle | LP | OC | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|
| 0.2898 | 0.2778 (±0.0002) | 0.2696 (±0.0001) | 0.2796 | 0.2793 | 0.2791 (±0.0000) |

### budget = 119.7

Fixed: sdxl_low_10steps=0.2801

| Oracle | LP | OC | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|
| 0.2934 | 0.2801 (±0.0000) | 0.2739 (±0.0002) | 0.2820 | 0.2815 | 0.2817 (±0.0001) |

### budget = 210.0

Fixed: sdxl_deepcache_50steps=0.2818

| Oracle | LP | OC | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|
| 0.2940 | 0.2818 (±0.0000) | 0.2771 (±0.0006) | 0.2825 | 0.2824 | 0.2826 (±0.0002) |

### budget = 598.5

Fixed: sdxl_ddim_50steps=0.2804 / sdxl_high_50steps=0.2808

| Oracle | LP | OC | Ridge | KNN | TFMLP |
|---|---|---|---|---|---|
| 0.2940 | 0.2808 (±0.0000) | 0.2771 (±0.0006) | 0.2826 | 0.2827 | 0.2826 (±0.0002) |
