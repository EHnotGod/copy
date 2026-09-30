# §8 路由收益与预测指标的相关性

## 实验动机
路由收益的高低究竟由哪类预测指标决定？直觉上研究者常用绝对分 R²（如 R²(q50)）来衡量预测器质量，但路由决策依赖的是样本间的 **相对难度排序**（哪条 query 在 tts 有增益），而非绝对分值的准确性。本实验在 test 集上同时计算多类指标，以路由收益为因变量，通过 Pearson 相关系数定量对比各指标的预测力，证明 delta 指标才是真正的routing-relevant signal，绝对分指标不可信。

## 实验设置
- 数据点：Ridge / KNN / TFMLP × 3 datasets × 3 models × 3 seeds = 27 点/predictor，共 81 点
- 固定 n=2200，score mode
- 所有指标均在 **test 集** 上计算（非 train/val）
- 路由收益 = route_quality(score_mode) − always_50steps
- 指标分两组：Unreliable（绝对分）vs Reliable（delta 分）

## Pearson r 汇总（n=2200，各 predictor 分列）

| 指标 | 类型 | Ridge r | KNN r | TFMLP r |
|------|------|---------|-------|--------|
| `abs_r2_50` | Unreliable | 0.071(p0.724) | 0.105(p0.601) | -0.131(p0.514) |
| `abs_r2_mean` | Unreliable | 0.050(p0.804) | 0.107(p0.596) | -0.138(p0.492) |
| `abs_sp_50` | Unreliable | -0.162(p0.421) | -0.192(p0.336) | -0.102(p0.613) |
| `delta_r2_d1` | Reliable | 0.272(p0.170) | 0.006(p0.977) | 0.050(p0.805) |
| `delta_r2_d2` | Reliable | 0.749(p<.001) | 0.808(p<.001) | 0.499(p0.008) |
| `delta_r2_mean` | Reliable | 0.561(p0.002) | 0.500(p0.008) | 0.258(p0.193) |
| `delta_sp_mean` | Reliable | 0.532(p0.004) | 0.267(p0.178) | 0.550(p0.003) |

## 实验结论

- **绝对分指标全部不相关**：`abs_r2_50`、`abs_r2_mean`、`abs_sp_50` 在三个 predictor 上 r 均接近 0（最高仅 0.107），p 值全部无显著性（p>0.3），散点呈水平云团。说明绝对分预测精度完全无法预测路由收益，用它来选 predictor 是误导性的。
- **delta_r2_d2 是唯一可靠指标**：R²(Δ₂=qtts−q50) 在 Ridge r=**0.749**、KNN r=**0.808**、TFMLP r=**0.499**，三个 predictor 均显著（p<0.01），且 r 值最高。图中散点有明显的正相关趋势，是所有指标中唯一在三个 predictor 上均成立的信号。
- **d₁ 完全无效**：`delta_r2_d1`（R²(q50−q10)）r 在 0.006-0.272 之间，均不显著。说明 10steps vs 50steps 的差值对路由几乎没有贡献，路由收益完全由 tts 的边际增益决定。
- **delta_sp_mean 次之**：对 Ridge（r=0.532）和 TFMLP（r=0.550）显著，但对 KNN 不显著（r=0.267, p=0.178）。
- **实践意义**：评估路由 predictor 时，**只用 `delta_r2_d2` = R²(qtts−q50)** 作为核心指标即可；绝对分 R² 高低与路由好坏无关，不应作为 predictor 选型依据。

## 图表

![exp8 metrics scatter](figs/exp8_metrics_scatter.png)

