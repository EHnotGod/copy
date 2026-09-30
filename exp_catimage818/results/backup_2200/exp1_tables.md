# 实验一：直接预测（COCO CATImage，clip 主指标）

mid 子集定义：y_true ∈ (q10, q90)（去上下各 10% 分位数；CLIPScore 范围 ~[0.2,0.3]，
没有 exp_815 的 0/1 堆积，故不用 (0.1, 0.9) 绝对阈值）。n_mid=1469（全部 7 cfg 同，
分位数定义下恒为 80%×1837 取整）。

## R²(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| 2200 | infinity_1step | +0.303 | +0.160 | +0.377 |
| 2200 | sdxl_turbo_1step | +0.211 | +0.105 | +0.298 |
| 2200 | sdxl_lightning_4steps | +0.274 | +0.140 | +0.366 |
| 2200 | sdxl_low_10steps | +0.270 | +0.145 | +0.382 |
| 2200 | sdxl_deepcache_50steps | +0.275 | +0.124 | +0.359 |
| 2200 | sdxl_ddim_50steps | +0.273 | +0.112 | +0.346 |
| 2200 | sdxl_high_50steps | +0.273 | +0.113 | +0.341 |
| full | infinity_1step | +0.418 | +0.234 | +0.435 |
| full | sdxl_turbo_1step | +0.310 | +0.156 | +0.396 |
| full | sdxl_lightning_4steps | +0.373 | +0.211 | +0.458 |
| full | sdxl_low_10steps | +0.369 | +0.191 | +0.465 |
| full | sdxl_deepcache_50steps | +0.368 | +0.180 | +0.453 |
| full | sdxl_ddim_50steps | +0.371 | +0.175 | +0.452 |
| full | sdxl_high_50steps | +0.370 | +0.175 | +0.459 |

## Spearman(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| 2200 | infinity_1step | +0.537 | +0.420 | +0.601 |
| 2200 | sdxl_turbo_1step | +0.474 | +0.324 | +0.562 |
| 2200 | sdxl_lightning_4steps | +0.524 | +0.391 | +0.609 |
| 2200 | sdxl_low_10steps | +0.525 | +0.385 | +0.623 |
| 2200 | sdxl_deepcache_50steps | +0.521 | +0.363 | +0.606 |
| 2200 | sdxl_ddim_50steps | +0.514 | +0.348 | +0.605 |
| 2200 | sdxl_high_50steps | +0.515 | +0.350 | +0.601 |
| full | infinity_1step | +0.634 | +0.491 | +0.667 |
| full | sdxl_turbo_1step | +0.556 | +0.392 | +0.634 |
| full | sdxl_lightning_4steps | +0.601 | +0.465 | +0.678 |
| full | sdxl_low_10steps | +0.603 | +0.430 | +0.678 |
| full | sdxl_deepcache_50steps | +0.601 | +0.439 | +0.673 |
| full | sdxl_ddim_50steps | +0.602 | +0.433 | +0.678 |
| full | sdxl_high_50steps | +0.601 | +0.434 | +0.682 |

## R²(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| 2200 | infinity_1step | -0.059 | +0.077 | +0.088 |
| 2200 | sdxl_turbo_1step | -0.130 | +0.030 | -0.098 |
| 2200 | sdxl_lightning_4steps | -0.111 | +0.057 | -0.012 |
| 2200 | sdxl_low_10steps | -0.137 | +0.056 | -0.072 |
| 2200 | sdxl_deepcache_50steps | -0.108 | +0.057 | -0.083 |
| 2200 | sdxl_ddim_50steps | -0.085 | +0.054 | -0.152 |
| 2200 | sdxl_high_50steps | -0.103 | +0.045 | -0.138 |
| full | infinity_1step | +0.105 | +0.133 | +0.073 |
| full | sdxl_turbo_1step | +0.018 | +0.052 | +0.090 |
| full | sdxl_lightning_4steps | +0.024 | +0.096 | +0.088 |
| full | sdxl_low_10steps | +0.003 | +0.066 | +0.081 |
| full | sdxl_deepcache_50steps | +0.015 | +0.102 | +0.083 |
| full | sdxl_ddim_50steps | +0.037 | +0.092 | +0.084 |
| full | sdxl_high_50steps | +0.039 | +0.088 | +0.104 |

## Spearman(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| 2200 | infinity_1step | +0.421 | +0.293 | +0.475 |
| 2200 | sdxl_turbo_1step | +0.373 | +0.223 | +0.453 |
| 2200 | sdxl_lightning_4steps | +0.394 | +0.266 | +0.476 |
| 2200 | sdxl_low_10steps | +0.397 | +0.267 | +0.483 |
| 2200 | sdxl_deepcache_50steps | +0.383 | +0.251 | +0.461 |
| 2200 | sdxl_ddim_50steps | +0.397 | +0.255 | +0.463 |
| 2200 | sdxl_high_50steps | +0.384 | +0.245 | +0.457 |
| full | infinity_1step | +0.510 | +0.380 | +0.538 |
| full | sdxl_turbo_1step | +0.450 | +0.283 | +0.517 |
| full | sdxl_lightning_4steps | +0.467 | +0.336 | +0.544 |
| full | sdxl_low_10steps | +0.473 | +0.308 | +0.534 |
| full | sdxl_deepcache_50steps | +0.464 | +0.338 | +0.525 |
| full | sdxl_ddim_50steps | +0.482 | +0.327 | +0.538 |
| full | sdxl_high_50steps | +0.477 | +0.325 | +0.542 |

