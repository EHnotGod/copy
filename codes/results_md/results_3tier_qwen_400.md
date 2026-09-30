# geneval2 · 3-tier router · qwen 和 flux 分开，delta vs score 对照

每个模型 3 档（10 / 50 / tts）。两种 predictor 目标：

- **delta**: 预测 (d1=S_50-S_10, d2=S_tts-S_50)，路由分数 s0=0, s1=d1, s2=d1+d2

- **score**: 直接预测 3 档绝对分数 [S_10, S_50, S_tts]


| model | cost 10 | cost 50 | cost tts | budget |
|---|---|---|---|---|
| qwen | 10 | 50 | 100 | 50 |
| flux | 23.1 | 115.5 | 231 | 115.5 |

## 决策收益（AUC @ budget，%oracle 括号内为 argmax accuracy）

| model | train pool | mode | fixed-10 | fixed-50 | fixed-tts | oracle@B | LP | OC | KNN | TFMLP |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen | 400 | delta | 24.21 | 24.82 | 44.82 | 44.76 | 33.36 (44.5%) | 32.87 (42.7%) | 35.10 (51.1%) | 36.07 (54.3%) |
| qwen | 400 | score | 24.21 | 24.82 | 44.82 | 44.76 | 33.36 (44.5%) | 32.87 (42.7%) | 35.10 (51.1%) | 35.21 (52.8%) |

## Delta 预测本身（Spearman ρ / MAE / WSA，d1=50-10, d2=tts-50）

score 模式下的 d1/d2 是从预测的绝对分数相减得到的。


| model | train pool | mode | predictor | Sp d1 | Sp d2 | MAE d1 | MAE d2 | WSA d1 | WSA d2 |
|---|---|---|---|---|---|---|---|---|---|
| qwen | 400 | delta | KNN | +0.013 | +0.168 | 0.086 | 0.196 | 0.525 | 0.905 |
| qwen | 400 | delta | TFMLP | +0.008 | +0.262 | 0.089 | 0.192 | 0.528 | 0.905 |
| qwen | 400 | score | KNN | +0.013 | +0.168 | 0.086 | 0.196 | 0.525 | 0.905 |
| qwen | 400 | score | TFMLP | +0.017 | +0.165 | 0.087 | 0.202 | 0.509 | 0.905 |
