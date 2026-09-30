# §28 Conditional Flux4B + QEval Probe

## 设置

- Stage 1 使用 prompt-only Ridge；以 cost-adjusted top-2 action margin 作为不确定性，margin 最低的 prompt 才进入 Flux4B + QEval。
- 候选 validation probe coverage：0/10%/25%/50%/75%/100%。每个 evaluator 成本情景只用 200 条 validation true scores 选 coverage 和阈值，然后在 test 冻结。
- 未 probe 样本使用原动作成本；probe 样本计入 Flux4B 0.455、假设的 evaluator cost，并在最终选 Flux4B 时复用 probe 图。
- test 侧 λ/混合比例仍只读取 prediction 和逐样本成本；test true scores 只用于冻结后的评价。
- GenEval2 GM，A10，预算 10，split seeds 42/43/44，paired bootstrap 2000 次。

![conditional probe](figs/exp28_conditional_probe.svg)

## 主要结果

| 假设 evaluator cost | Val-selected test coverage | Conditional quality | 相对 Prompt | 95% CI | Always-probe 增益 | Fixed-50% 增益 |
|---:|---:|---:|---:|---|---:|---:|
| 0 | 74.3% | 0.4003 | +0.0197 | [+0.0089, +0.0310] | +0.0227 | +0.0151 |
| 0.5 | 74.3% | 0.3977 | +0.0171 | [+0.0063, +0.0279] | +0.0192 | +0.0140 |
| 1 | 74.3% | 0.3954 | +0.0148 | [+0.0041, +0.0262] | +0.0162 | +0.0127 |
| 2 | 62.9% | 0.3916 | +0.0110 | [+0.0012, +0.0216] | +0.0104 | +0.0096 |
| 3 | 74.3% | 0.3874 | +0.0068 | [-0.0044, +0.0174] | +0.0048 | +0.0073 |
| 4 | 54.6% | 0.3847 | +0.0040 | [-0.0047, +0.0123] | -0.0006 | +0.0051 |

## 结论

1. **简单 margin gate 有效。** evaluator cost=2 时，平均只 probe 62.9%，仍获得 +0.0110，paired CI 为 [+0.0012, +0.0216]，显著高于 prompt-only。
2. **它确实延长了 break-even。** evaluator cost=4 时，always-probe 已为 -0.0006，固定 50% gate 仍为 +0.0051。
3. **当前可靠区间大约到 cost=2。** cost=3 的点估计仍有 +0.0068，但 CI 已跨 0；不能据此声称稳定净收益。
4. **精确 coverage 还不稳定。** 只有 200 条 validation，三个 seeds 选择的覆盖率有波动。因此现阶段应把 25%/50%/75% 看成成本曲线，而不是宣称某一个覆盖率最优。

这说明后续轻量化目标可以更具体：如果 evaluator 成本能压到约 2 个 action-cost 单位以内，简单 conditional probe 已有稳定净收益；若仍在 3–4 左右，则需要更轻的 evaluator 或更好的 benefit gate。

## 数值审计

- 每个 conditional policy 均使用逐样本成本，最终平均成本为 10（浮点误差范围内）。
- Coverage 只由 validation 选择；所有 test fixed-coverage 曲线均为预先给定的诊断网格。
- evaluator cost 是与现有 action TFLOPs 相同单位的敏感性参数，不是当前 Qwen3-VL 的实测成本。
