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
| flux | 2400 | delta | 31.63 | 33.21 | 37.21 | 43.31 | 34.06 (37.9%) | 33.95 (36.7%) | 34.37 (36.7%) | 34.56 (39.0%) |
| flux | 2400 | score | 31.63 | 33.21 | 37.21 | 43.31 | 34.06 (37.9%) | 33.95 (36.7%) | 34.37 (36.7%) | 34.87 (41.9%) |

## Delta 预测本身（Spearman ρ / MAE / WSA，d1=50-10, d2=tts-50）

score 模式下的 d1/d2 是从预测的绝对分数相减得到的。


| model | train pool | mode | predictor | Sp d1 | Sp d2 | MAE d1 | MAE d2 | WSA d1 | WSA d2 |
|---|---|---|---|---|---|---|---|---|---|
| flux | 2400 | delta | KNN | +0.057 | +0.009 | 0.104 | 0.128 | 0.547 | 0.645 |
| flux | 2400 | delta | TFMLP | +0.084 | -0.001 | 0.104 | 0.133 | 0.561 | 0.650 |
| flux | 2400 | score | KNN | +0.057 | +0.009 | 0.104 | 0.128 | 0.547 | 0.645 |
| flux | 2400 | score | TFMLP | +0.087 | +0.036 | 0.103 | 0.132 | 0.565 | 0.672 |
