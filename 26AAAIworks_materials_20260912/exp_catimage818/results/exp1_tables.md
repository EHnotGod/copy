# 实验一：直接预测（COCO CATImage，四指标）

mid 子集定义：y_true ∈ (q10, q90)（去上下各 10% 分位数；各指标均无
exp_815 式 0/1 堆积，故不用 (0.1, 0.9) 绝对阈值）。n_mid=1471（全部
指标 × cfg 相同，分位数定义下恒为 80%×1839 取整）。

## clip

### R²(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.409 | +0.227 | +0.450 |
| full | sdxl_turbo_1step | +0.297 | +0.150 | +0.402 |
| full | sdxl_lightning_4steps | +0.358 | +0.206 | +0.461 |
| full | sdxl_low_10steps | +0.357 | +0.183 | +0.476 |
| full | sdxl_deepcache_50steps | +0.352 | +0.169 | +0.463 |
| full | sdxl_ddim_50steps | +0.355 | +0.165 | +0.461 |
| full | sdxl_high_50steps | +0.354 | +0.166 | +0.461 |

### Spearman(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.627 | +0.482 | +0.650 |
| full | sdxl_turbo_1step | +0.544 | +0.383 | +0.626 |
| full | sdxl_lightning_4steps | +0.588 | +0.457 | +0.665 |
| full | sdxl_low_10steps | +0.595 | +0.419 | +0.673 |
| full | sdxl_deepcache_50steps | +0.588 | +0.422 | +0.665 |
| full | sdxl_ddim_50steps | +0.590 | +0.417 | +0.666 |
| full | sdxl_high_50steps | +0.588 | +0.421 | +0.667 |

### R²(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.098 | +0.129 | +0.129 |
| full | sdxl_turbo_1step | -0.001 | +0.040 | +0.070 |
| full | sdxl_lightning_4steps | +0.009 | +0.083 | +0.099 |
| full | sdxl_low_10steps | -0.014 | +0.056 | +0.138 |
| full | sdxl_deepcache_50steps | -0.007 | +0.090 | +0.111 |
| full | sdxl_ddim_50steps | +0.012 | +0.080 | +0.125 |
| full | sdxl_high_50steps | +0.016 | +0.082 | +0.121 |

### Spearman(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.500 | +0.372 | +0.518 |
| full | sdxl_turbo_1step | +0.436 | +0.270 | +0.506 |
| full | sdxl_lightning_4steps | +0.455 | +0.322 | +0.529 |
| full | sdxl_low_10steps | +0.465 | +0.295 | +0.521 |
| full | sdxl_deepcache_50steps | +0.450 | +0.321 | +0.511 |
| full | sdxl_ddim_50steps | +0.468 | +0.310 | +0.523 |
| full | sdxl_high_50steps | +0.463 | +0.314 | +0.523 |

## sharp

### R²(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.637 | +0.386 | +0.663 |
| full | sdxl_turbo_1step | +0.572 | +0.281 | +0.634 |
| full | sdxl_lightning_4steps | +0.606 | +0.399 | +0.675 |
| full | sdxl_low_10steps | +0.592 | +0.358 | +0.659 |
| full | sdxl_deepcache_50steps | +0.554 | +0.336 | +0.647 |
| full | sdxl_ddim_50steps | +0.549 | +0.342 | +0.634 |
| full | sdxl_high_50steps | +0.548 | +0.341 | +0.633 |

### Spearman(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.802 | +0.651 | +0.815 |
| full | sdxl_turbo_1step | +0.764 | +0.590 | +0.803 |
| full | sdxl_lightning_4steps | +0.777 | +0.660 | +0.819 |
| full | sdxl_low_10steps | +0.771 | +0.634 | +0.811 |
| full | sdxl_deepcache_50steps | +0.739 | +0.611 | +0.799 |
| full | sdxl_ddim_50steps | +0.733 | +0.608 | +0.794 |
| full | sdxl_high_50steps | +0.733 | +0.607 | +0.794 |

### R²(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.386 | +0.291 | +0.405 |
| full | sdxl_turbo_1step | +0.299 | +0.211 | +0.310 |
| full | sdxl_lightning_4steps | +0.343 | +0.297 | +0.405 |
| full | sdxl_low_10steps | +0.329 | +0.245 | +0.381 |
| full | sdxl_deepcache_50steps | +0.231 | +0.228 | +0.318 |
| full | sdxl_ddim_50steps | +0.205 | +0.219 | +0.286 |
| full | sdxl_high_50steps | +0.202 | +0.219 | +0.285 |

### Spearman(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.701 | +0.545 | +0.709 |
| full | sdxl_turbo_1step | +0.655 | +0.473 | +0.699 |
| full | sdxl_lightning_4steps | +0.666 | +0.549 | +0.716 |
| full | sdxl_low_10steps | +0.664 | +0.515 | +0.705 |
| full | sdxl_deepcache_50steps | +0.618 | +0.483 | +0.686 |
| full | sdxl_ddim_50steps | +0.607 | +0.482 | +0.681 |
| full | sdxl_high_50steps | +0.606 | +0.481 | +0.680 |

## aesthetic

### R²(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.515 | +0.347 | +0.493 |
| full | sdxl_turbo_1step | +0.432 | +0.250 | +0.421 |
| full | sdxl_lightning_4steps | +0.560 | +0.361 | +0.569 |
| full | sdxl_low_10steps | +0.536 | +0.320 | +0.529 |
| full | sdxl_deepcache_50steps | +0.526 | +0.314 | +0.528 |
| full | sdxl_ddim_50steps | +0.499 | +0.292 | +0.499 |
| full | sdxl_high_50steps | +0.516 | +0.308 | +0.516 |

### Spearman(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.689 | +0.586 | +0.664 |
| full | sdxl_turbo_1step | +0.632 | +0.502 | +0.628 |
| full | sdxl_lightning_4steps | +0.710 | +0.588 | +0.715 |
| full | sdxl_low_10steps | +0.688 | +0.536 | +0.675 |
| full | sdxl_deepcache_50steps | +0.679 | +0.527 | +0.678 |
| full | sdxl_ddim_50steps | +0.665 | +0.523 | +0.668 |
| full | sdxl_high_50steps | +0.678 | +0.529 | +0.677 |

### R²(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.078 | +0.188 | +0.024 |
| full | sdxl_turbo_1step | +0.015 | +0.143 | -0.007 |
| full | sdxl_lightning_4steps | +0.098 | +0.185 | +0.145 |
| full | sdxl_low_10steps | +0.052 | +0.092 | +0.064 |
| full | sdxl_deepcache_50steps | +0.044 | +0.119 | +0.060 |
| full | sdxl_ddim_50steps | +0.042 | +0.120 | +0.084 |
| full | sdxl_high_50steps | +0.040 | +0.112 | +0.099 |

### Spearman(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.564 | +0.474 | +0.524 |
| full | sdxl_turbo_1step | +0.488 | +0.399 | +0.471 |
| full | sdxl_lightning_4steps | +0.574 | +0.457 | +0.581 |
| full | sdxl_low_10steps | +0.531 | +0.373 | +0.506 |
| full | sdxl_deepcache_50steps | +0.529 | +0.384 | +0.525 |
| full | sdxl_ddim_50steps | +0.512 | +0.383 | +0.516 |
| full | sdxl_high_50steps | +0.521 | +0.377 | +0.528 |

## imagereward

### R²(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.272 | +0.179 | +0.249 |
| full | sdxl_turbo_1step | +0.231 | +0.141 | +0.193 |
| full | sdxl_lightning_4steps | +0.216 | +0.150 | +0.209 |
| full | sdxl_low_10steps | +0.285 | +0.176 | +0.240 |
| full | sdxl_deepcache_50steps | +0.249 | +0.170 | +0.237 |
| full | sdxl_ddim_50steps | +0.237 | +0.157 | +0.239 |
| full | sdxl_high_50steps | +0.242 | +0.159 | +0.234 |

### Spearman(all)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.557 | +0.454 | +0.532 |
| full | sdxl_turbo_1step | +0.501 | +0.388 | +0.463 |
| full | sdxl_lightning_4steps | +0.490 | +0.395 | +0.475 |
| full | sdxl_low_10steps | +0.529 | +0.404 | +0.511 |
| full | sdxl_deepcache_50steps | +0.515 | +0.425 | +0.511 |
| full | sdxl_ddim_50steps | +0.508 | +0.407 | +0.512 |
| full | sdxl_high_50steps | +0.509 | +0.407 | +0.509 |

### R²(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | -0.036 | +0.068 | +0.023 |
| full | sdxl_turbo_1step | -0.077 | +0.012 | -0.079 |
| full | sdxl_lightning_4steps | -0.071 | +0.061 | +0.001 |
| full | sdxl_low_10steps | -0.066 | +0.055 | -0.078 |
| full | sdxl_deepcache_50steps | -0.084 | +0.062 | -0.066 |
| full | sdxl_ddim_50steps | -0.089 | +0.043 | -0.060 |
| full | sdxl_high_50steps | -0.097 | +0.039 | -0.078 |

### Spearman(mid)

| train_size | cfg | Ridge | KNN | TFMLP |
|---|---|---|---|---|
| full | infinity_1step | +0.430 | +0.330 | +0.405 |
| full | sdxl_turbo_1step | +0.375 | +0.258 | +0.339 |
| full | sdxl_lightning_4steps | +0.379 | +0.290 | +0.368 |
| full | sdxl_low_10steps | +0.382 | +0.279 | +0.382 |
| full | sdxl_deepcache_50steps | +0.383 | +0.309 | +0.387 |
| full | sdxl_ddim_50steps | +0.382 | +0.291 | +0.392 |
| full | sdxl_high_50steps | +0.375 | +0.283 | +0.382 |

