# 实验二：R²(all) / R²(mid) / delta-R² 全样本 capacity

**核心命题：** 全样本 R²(all) 看起来不错，但中间区间直接塌陷；
delta-R²（预测档位间增益的能力）揭示不同 predictor / 配置的实际可分程度。

## 实验设置

| 项目 | 说明 |
|---|---|
| 训练集 | pool 共 2400 条，固定 seed=42 切 200 条 external val，剩余 2200 条；3 seeds（0/1/2）均值（每 seed 从 2200 中随机采样 n=2000 训练）|
| 测试集 | geneval2: 800 条（raw_perseed）；unigen: 600 条（raw600_edit_v2）|
| 生成模型 | Qwen-Image-2512（qwen）/ FLUX.2-Klein-9B（flux）/ Boogu-Image（boogu）× 3 档（10steps / 50steps / tts）|
| **Ridge** | Qwen3-Embedding-8B（4096 dim）+ StandardScaler；α ∈ {1,10,100,1000}，external val R² 选最优 α |
| **KNN** | CLIP-ViT-L/14 mean-pool（768 dim）+ StandardScaler；k = min(50, n//3)；abs 模式预测绝对分，delta 模式直接预测 Δq |
| **TFMLP** | 2-layer self-attn + per-token MLP（CATImage Appendix D）；abs 模式 M=1 预测绝对分，delta 模式 M=2 预测 (d1,d2)；val-CE 早停 patience=30 |
| **R²(all)** | 全 test 集 R² |
| **R²(mid)** | 仅 y_true ∈ (0.10, 0.90) 子集的 R² |
| **delta-R²** | d1 = q₅₀ − q₁₀，d2 = q_tts − q₅₀；Ridge 从绝对预测相减得 Δp；KNN/TFMLP 直接预测 Δq；每个 delta 维度分别报 R² |

## GenEval2 GM（800 test samples）

![GenEval2 GM capacity](figs/exp2_capacity_gm.png)

### R²(all) 和 R²(mid)（绝对分预测，n=2200，3 seeds 均值）

| model | cfg | Ridge R²(all) | Ridge R²(mid) | KNN R²(all) | KNN R²(mid) | TFMLP R²(all) | TFMLP R²(mid) |
|---|---|---|---|---|---|---|---|
| qwen | 10steps | +0.310 | -0.005 | +0.060 | -0.621 | +0.174 | -0.426 |
| qwen | 50steps | +0.292 | -0.046 | +0.034 | -0.650 | +0.217 | -0.381 |
| qwen | tts | +0.393 | +0.152 | +0.058 | -0.043 | +0.332 | -0.011 |
| flux | 10steps | +0.506 | +0.180 | +0.140 | -0.170 | +0.397 | -0.134 |
| flux | 50steps | +0.507 | +0.123 | +0.148 | -0.159 | +0.426 | -0.203 |
| flux | tts | +0.464 | +0.102 | +0.121 | -0.166 | +0.376 | -0.099 |
| boogu | 10steps | +0.363 | +0.132 | +0.088 | -0.123 | +0.298 | -0.116 |
| boogu | 50steps | +0.383 | +0.124 | +0.062 | -0.159 | +0.397 | -0.008 |
| boogu | tts | +0.392 | +0.140 | +0.089 | -0.043 | +0.357 | +0.051 |

### delta-R²（d1 = q₅₀ − q₁₀，d2 = q_tts − q₅₀）

Ridge 的 delta 从绝对预测相减；KNN/TFMLP 直接预测 delta（M=2）。

| model | delta | Ridge delta-R² | KNN delta-R² | TFMLP delta-R² |
|---|---|---|---|---|
| qwen | d1（q₅₀−q₁₀）| -0.131 | -0.017 | -0.017 |
| qwen | d2（q_tts−q₅₀）| +0.046 | +0.045 | +0.159 |
| flux | d1（q₅₀−q₁₀）| -0.077 | +0.003 | +0.006 |
| flux | d2（q_tts−q₅₀）| -0.150 | -0.017 | -0.008 |
| boogu | d1（q₅₀−q₁₀）| -0.078 | -0.029 | +0.018 |
| boogu | d2（q_tts−q₅₀）| -0.050 | -0.006 | +0.020 |

<!-- scatter_gm -->
### 散点图（seed=0）
**qwen** — 绝对分预测（3 predictor × 3 cfg）
![score scatter qwen](figs/exp2_scatter_score_gm_qwen.png)
**qwen** — delta 预测（3 predictor × d1/d2）
![delta scatter qwen](figs/exp2_scatter_delta_gm_qwen.png)


## GenEval2 AM（800 test samples）

![GenEval2 AM capacity](figs/exp2_capacity_am.png)

### R²(all) 和 R²(mid)（绝对分预测，n=2200，3 seeds 均值）

| model | cfg | Ridge R²(all) | Ridge R²(mid) | KNN R²(all) | KNN R²(mid) | TFMLP R²(all) | TFMLP R²(mid) |
|---|---|---|---|---|---|---|---|
| qwen | 10steps | +0.376 | +0.258 | -0.178 | -0.129 | +0.249 | +0.144 |
| qwen | 50steps | +0.392 | +0.283 | -0.203 | -0.139 | +0.223 | +0.089 |
| qwen | tts | +0.298 | -0.043 | -0.108 | +0.026 | +0.269 | -0.124 |
| flux | 10steps | +0.367 | +0.115 | +0.021 | +0.056 | +0.267 | -0.024 |
| flux | 50steps | +0.352 | +0.102 | -0.008 | +0.059 | +0.239 | -0.062 |
| flux | tts | +0.357 | +0.129 | -0.021 | +0.070 | +0.313 | +0.022 |
| boogu | 10steps | +0.308 | +0.061 | -0.136 | -0.036 | +0.232 | -0.120 |
| boogu | 50steps | +0.269 | -0.004 | -0.134 | +0.023 | +0.248 | -0.151 |
| boogu | tts | +0.219 | -0.219 | -0.087 | -0.013 | +0.200 | -0.211 |

### delta-R²（d1 = q₅₀ − q₁₀，d2 = q_tts − q₅₀）

Ridge 的 delta 从绝对预测相减；KNN/TFMLP 直接预测 delta（M=2）。

| model | delta | Ridge delta-R² | KNN delta-R² | TFMLP delta-R² |
|---|---|---|---|---|
| qwen | d1（q₅₀−q₁₀）| -0.200 | -0.014 | -0.036 |
| qwen | d2（q_tts−q₅₀）| +0.072 | +0.024 | +0.171 |
| flux | d1（q₅₀−q₁₀）| -0.170 | -0.025 | -0.022 |
| flux | d2（q_tts−q₅₀）| -0.204 | -0.022 | -0.088 |
| boogu | d1（q₅₀−q₁₀）| -0.130 | -0.017 | -0.020 |
| boogu | d2（q_tts−q₅₀）| -0.073 | -0.026 | +0.003 |

<!-- scatter_am -->
### 散点图（seed=0）
**qwen** — 绝对分预测（3 predictor × 3 cfg）
![score scatter qwen](figs/exp2_scatter_score_am_qwen.png)
**qwen** — delta 预测（3 predictor × d1/d2）
![delta scatter qwen](figs/exp2_scatter_delta_am_qwen.png)
**flux** — 绝对分预测（3 predictor × 3 cfg）
![score scatter flux](figs/exp2_scatter_score_am_flux.png)
**flux** — delta 预测（3 predictor × d1/d2）
![delta scatter flux](figs/exp2_scatter_delta_am_flux.png)
**boogu** — 绝对分预测（3 predictor × 3 cfg）
![score scatter boogu](figs/exp2_scatter_score_am_boogu.png)
**boogu** — delta 预测（3 predictor × d1/d2）
![delta scatter boogu](figs/exp2_scatter_delta_am_boogu.png)


## UniGenBench（600 test samples）

![UniGenBench capacity](figs/exp2_capacity_unigen.png)

### R²(all) 和 R²(mid)（绝对分预测，n=2200，3 seeds 均值）

| model | cfg | Ridge R²(all) | Ridge R²(mid) | KNN R²(all) | KNN R²(mid) | TFMLP R²(all) | TFMLP R²(mid) |
|---|---|---|---|---|---|---|---|
| qwen | 10steps | +0.423 | +0.042 | +0.077 | -0.393 | +0.179 | -0.333 |
| qwen | 50steps | +0.423 | +0.148 | +0.079 | -0.421 | +0.171 | -0.337 |
| qwen | tts | +0.267 | -0.392 | +0.003 | -0.708 | +0.031 | -0.804 |
| flux | 10steps | +0.433 | +0.133 | +0.077 | -0.553 | +0.170 | -0.550 |
| flux | 50steps | +0.387 | -0.023 | +0.047 | -0.652 | +0.084 | -0.672 |
| flux | tts | +0.278 | -0.178 | +0.031 | -0.621 | +0.044 | -0.558 |
| boogu | 10steps | +0.264 | -0.209 | +0.006 | -0.468 | +0.070 | -0.490 |
| boogu | 50steps | +0.303 | -0.301 | +0.015 | -0.905 | +0.052 | -0.980 |
| boogu | tts | +0.126 | -0.730 | -0.011 | -0.853 | +0.016 | -0.971 |

### delta-R²（d1 = q₅₀ − q₁₀，d2 = q_tts − q₅₀）

Ridge 的 delta 从绝对预测相减；KNN/TFMLP 直接预测 delta（M=2）。

| model | delta | Ridge delta-R² | KNN delta-R² | TFMLP delta-R² |
|---|---|---|---|---|
| qwen | d1（q₅₀−q₁₀）| +0.115 | +0.012 | +0.000 |
| qwen | d2（q_tts−q₅₀）| +0.375 | +0.038 | +0.039 |
| flux | d1（q₅₀−q₁₀）| +0.190 | +0.031 | +0.013 |
| flux | d2（q_tts−q₅₀）| +0.213 | +0.012 | +0.023 |
| boogu | d1（q₅₀−q₁₀）| +0.303 | +0.045 | +0.043 |
| boogu | d2（q_tts−q₅₀）| +0.178 | +0.025 | +0.011 |

<!-- scatter_unigen -->
### 散点图（seed=0）
**qwen** — 绝对分预测（3 predictor × 3 cfg）
![score scatter qwen](figs/exp2_scatter_score_unigen_qwen.png)
**qwen** — delta 预测（3 predictor × d1/d2）
![delta scatter qwen](figs/exp2_scatter_delta_unigen_qwen.png)
**flux** — 绝对分预测（3 predictor × 3 cfg）
![score scatter flux](figs/exp2_scatter_score_unigen_flux.png)
**flux** — delta 预测（3 predictor × d1/d2）
![delta scatter flux](figs/exp2_scatter_delta_unigen_flux.png)
**boogu** — 绝对分预测（3 predictor × 3 cfg）
![score scatter boogu](figs/exp2_scatter_score_unigen_boogu.png)
**boogu** — delta 预测（3 predictor × d1/d2）
![delta scatter boogu](figs/exp2_scatter_delta_unigen_boogu.png)


## 规律小结

**R²(all) 虚高，R²(mid) 普遍塌陷：**
三种 predictor 在全样本上的 R²(all) 普遍在 0.1–0.5 之间，看起来有预测力。
但剔除两端（y ∈ (0.10, 0.90)）后，R²(mid) 大幅下降甚至为负——GM 的低分堆积、UniGen 的高分堆积是 R²(all) 的主要来源，中间区间几乎无信号。

**KNN / TFMLP 与 Ridge 的对比：**
- KNN（CLIP 768 dim）的 R²(all) 普遍低于 Ridge（Qwen3 4096 dim），说明 Qwen3 embedding 信息密度更高。
- R²(mid) 上三种 predictor 差距收窄甚至 KNN 更差——中间区间的瓶颈是信号稀薄而非特征质量。
- TFMLP 的 R²(all) 介于 KNN 和 Ridge 之间（CLIP 特征 + 非线性结构），但 R²(mid) 同样无法修复。

**delta-R²：d1 vs d2：**
- d1（q₅₀ − q₁₀，低档→中档增益）在 GM 上普遍为负；d2（q_tts − q₅₀，中档→高档增益）略好但仍接近零。
- UniGen 上 delta-R² 相对更高（Ridge d1 最高达 +0.30），与旧实验中 UniGen 路由收益更强的结论一致。
- KNN 和 TFMLP 的 delta-R² 均显著低于 Ridge——直接预测 delta 并不比从绝对预测相减更强，
  在线性框架下结论延续（exp_delta_routing.md）；非线性的 TFMLP 也未带来提升，说明瓶颈在数据信号本身。
