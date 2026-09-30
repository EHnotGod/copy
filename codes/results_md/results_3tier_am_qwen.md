# geneval2 · 3-tier router · qwen 和 flux 分开，delta vs score 对照

每个模型 3 档（10 / 50 / tts）。两种 predictor 目标：

- **delta**: 预测 (d1=S_50-S_10, d2=S_tts-S_50)，路由分数 s0=0, s1=d1, s2=d1+d2

- **score**: 直接预测 3 档绝对分数 [S_10, S_50, S_tts]


| model | cost 10 | cost 50 | cost tts | budget |
|---|---|---|---|---|
| qwen | 10 | 50 | 100 | 50 |
| flux | 23.1 | 115.5 | 231 | 115.5 |

## 决策收益（quality @ budget，0-1 分。括号内为 argmax accuracy）

| model | train pool | mode | fixed-10 | fixed-50 | fixed-tts | oracle@B | LP | OC | KNN | TFMLP |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen | 2400 | delta | 0.7221 | 0.7300 | 0.8147 | 0.8136 | 0.7642 (44.7%) | 0.7629 (43.2%) | 0.7760 (51.7%) | 0.7783 (54.2%) |
| qwen | 2400 | score | 0.7221 | 0.7300 | 0.8147 | 0.8136 | 0.7642 (44.7%) | 0.7629 (43.2%) | 0.7760 (51.7%) | 0.7784 (55.1%) |

## Delta 预测本身（Spearman ρ / MAE / WSA，d1=50-10, d2=tts-50）

score 模式下的 d1/d2 是从预测的绝对分数相减得到的。


| model | train pool | mode | predictor | Sp d1 | Sp d2 | MAE d1 | MAE d2 | WSA d1 | WSA d2 |
|---|---|---|---|---|---|---|---|---|---|
| qwen | 2400 | delta | KNN | -0.005 | +0.220 | 0.045 | 0.090 | 0.562 | 0.886 |
| qwen | 2400 | delta | TFMLP | -0.014 | +0.301 | 0.046 | 0.081 | 0.507 | 0.885 |
| qwen | 2400 | score | KNN | -0.005 | +0.220 | 0.045 | 0.090 | 0.562 | 0.886 |
| qwen | 2400 | score | TFMLP | -0.020 | +0.295 | 0.046 | 0.082 | 0.533 | 0.886 |
