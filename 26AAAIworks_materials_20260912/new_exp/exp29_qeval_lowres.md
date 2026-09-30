# §29 QEval 分辨率—计算量 Pareto

## 设置

- Qwen3-VL-8B、评价 prompt、final query token、Ridge、A10 和 budget=10 全部不变，只修改输入图像上限。
- 256/320/1024 对应视觉 patch 数 256/400/4096，LLM 序列约 117/153/1077。
- 解析 matmul FLOPs：256=1.873、320=2.521、1024=21.443 TFLOPs；不运行文本生成或 LM head。
- GenEval2 GM，split seeds 42/43/44，paired bootstrap 2000 次。

![resolution Pareto](figs/exp29_qeval_resolution_pareto.svg)

## 表征与路由结果

| 输入 | TFLOPs | Macro R² | Info-only Router | 计入 evaluator 后 | 相对 Prompt |
|---|---:|---:|---:|---:|---:|
| Prompt-only | — | 0.3958 | — | 0.3806 | — |
| QEval 256 | 1.873 | 0.4681 | 0.3943 | 0.3857 | +0.0050 |
| QEval 320 | 2.521 | 0.4755 | 0.3973 | 0.3867 | +0.0060 |
| QEval 1024 | 21.443 | 0.4777 | 0.4033 | 不可行 | — |

计入 evaluator 后的 paired 区间：

- 256：+0.0050，95% CI [-0.0079, +0.0176]。
- 320：+0.0060，95% CI [-0.0074, +0.0187]。

## Conditional probe

- 256：validation-selected coverage 平均 74.3%，质量 0.3882，相对 Prompt +0.0075，CI [-0.0033, +0.0176]。
- 320：validation-selected coverage 平均 11.7%，质量 0.3810，相对 Prompt +0.0004，CI [-0.0029, +0.0039]。

320 的简单 margin gate 在小 validation 上过于保守；三个 seeds 最终平均只 probe 11.7%。因此该结果说明当前 gate 不稳，不代表 320 图像没有价值。固定 always-probe 的点估计仍为 +0.0060。

## 结论

1. **256 确实压得偏狠。** 它保住了大部分整体 R²，但相对同一 VLM text-only 的图像增量只剩 +0.0043，CI 跨 0。
2. **320 是更好的表征 Pareto 点。** 只用全分辨率约 11.8% 的 FLOPs，Macro R² 已达到 0.4755，接近 1024 的 0.4777；但 Router information quality 仍低约 0.0060。
3. **完整计费后两者都只有正向点估计，尚未显著。** 256/320 分别为 +0.0050/+0.0060，不能写成稳定净收益。
4. **不建议继续直接上 448。** 320→1024 的 R² 上限只剩约 0.0022，而 448 成本会升到约 4.28 TFLOPs。下一步应改进 benefit gate 或做 early-exit/小 VLM，而不是单纯增加分辨率。

成本估算是保守口径：向新方案收取完整 Qwen3-VL FLOPs，没有减去当前系统本来就要支付的 Qwen3-Embedding prompt encoder 成本。若要得到最终系统净成本，需要统一核算两套 router encoder。

## 数值审计

- 256/320 缓存均为 2400 pool + 800 test、4096 维，无 NaN/Inf，instruction 与 1024 缓存逐条一致。
- 所有可行策略的实际平均成本均为 10；1024 evaluator + Flux probe 的最低成本约 21.90，因此 always-probe 在 budget=10 下不可行。
- FLOPs 只统计主要矩阵乘，不含 norm/activation/softmax/preprocessing；适合相对比较，不应当作硬件延迟。
