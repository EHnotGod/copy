# 实验二 (Ridge-only)：R²(all) / R²(mid) / delta-R²

**预测器固定**：Ridge + Qwen3-Embedding-8B (4096 dim)，α ∈ {1,10,100,1000}，external val R² 选最优。
KNN / TFMLP 已去除，保证控制变量。

## 实验设置
| 项目 | 说明 |
|---|---|
| 训练集 | pool 共 2400 条，seed=42 切 200 external val，剩余 2200；3 seeds(0/1/2) 均值 |
| 测试集 | GenEval2 GM/AM: 800 条；UniGenBench: 600 条 |
| 模型 | qwen / flux / boogu × 10steps / 50steps / tts |
| Predictor | Ridge + Qwen3-Embedding-8B |
| delta | d1=q50-q10，d2=qtts-q50，从绝对预测相减 |

## GenEval2 GM（800 test samples）

| model | cfg | R²(all) | R²(mid) |
|---|---|---|---|
| boogu | 10steps | +0.363 | +0.132 |
| boogu | 50steps | +0.383 | +0.124 |
| boogu | tts | +0.392 | +0.140 |
| flux | 10steps | +0.506 | +0.180 |
| flux | 50steps | +0.507 | +0.123 |
| flux | tts | +0.464 | +0.102 |
| qwen | 10steps | +0.310 | -0.005 |
| qwen | 50steps | +0.292 | -0.046 |
| qwen | tts | +0.393 | +0.152 |

| model | d1 (q50-q10) | d2 (qtts-q50) |
|---|---|---|
| boogu | -0.078 | -0.050 |
| flux | -0.077 | -0.150 |
| qwen | -0.131 | +0.046 |

## GenEval2 AM（800 test samples）

| model | cfg | R²(all) | R²(mid) |
|---|---|---|---|
| qwen | 10steps | +0.376 | +0.258 |
| qwen | 50steps | +0.392 | +0.283 |
| qwen | tts | +0.298 | -0.043 |
| boogu | 10steps | +0.308 | +0.061 |
| boogu | 50steps | +0.269 | -0.004 |
| boogu | tts | +0.219 | -0.219 |
| flux | 10steps | +0.367 | +0.115 |
| flux | 50steps | +0.352 | +0.102 |
| flux | tts | +0.357 | +0.129 |

| model | d1 (q50-q10) | d2 (qtts-q50) |
|---|---|---|
| qwen | -0.200 | +0.072 |
| boogu | -0.130 | -0.073 |
| flux | -0.170 | -0.204 |

## UniGenBench（600 test samples）

| model | cfg | R²(all) | R²(mid) |
|---|---|---|---|
| qwen | 10steps | +0.423 | +0.042 |
| qwen | 50steps | +0.423 | +0.148 |
| qwen | tts | +0.267 | -0.392 |
| flux | 10steps | +0.433 | +0.133 |
| flux | 50steps | +0.387 | -0.023 |
| flux | tts | +0.278 | -0.178 |
| boogu | 10steps | +0.264 | -0.209 |
| boogu | 50steps | +0.303 | -0.301 |
| boogu | tts | +0.126 | -0.730 |

| model | d1 (q50-q10) | d2 (qtts-q50) |
|---|---|---|
| qwen | +0.115 | +0.375 |
| flux | +0.190 | +0.213 |
| boogu | +0.303 | +0.178 |

