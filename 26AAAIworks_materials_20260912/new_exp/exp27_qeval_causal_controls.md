# §27 QEval 图像信息因果对照

## 问题与设置

§26 的旧 Prompt embedding 来自 Qwen3-Embedding，而 QEval 来自 Qwen3-VL，因此原提升可能混合了“VLM 文本表征更强”和“正确结果图有信息”两部分。§27 固定 Qwen3-VL-8B、评价 template、4096 维 final-query-token、Ridge、训练划分和路由规则，只改变视觉输入：

- `text-only`：不给图片；
- `shuffled image`：原 prompt 配同 split 内另一条 prompt 的 Flux4B 图，使用确定性 derangement，零自匹配；
- `matched image`：原 prompt 配自己的 Flux4B seed_0 图。

三组使用相同的 cascade cost 评价。Flux probe 成本 0.455 已计入；Qwen evaluator TFLOPs 仍未计入。split seeds 为 42/43/44，每个差值做 2000 次 paired bootstrap。

![QEval causal controls](figs/exp27_qeval_causal_controls.svg)

## 三个 split 的平均结果

| 输入 | Macro test R² | Router quality | seed std |
|---|---:|---:|---:|
| 旧 Prompt embedding | 0.3958 | 0.3788 | 0.0023 |
| Qwen3-VL text-only | 0.4336 | 0.3900 | 0.0003 |
| Qwen3-VL shuffled image | 0.3758 | 0.3738 | 0.0029 |
| **Qwen3-VL matched image** | **0.4777** | **0.4033** | 0.0032 |

## 因果差值

| 对照 | 3-split mean | Paired 95% CI | 解释 |
|---|---:|---|---|
| Matched − shuffled | +0.0295 | [+0.0165, +0.0419] | prompt-image 对齐信息 |
| Matched − text-only | +0.0133 | [+0.0009, +0.0260] | 正确结果图相对同一 VLM 文本特征的增量 |
| Shuffled − text-only | -0.0162 | [-0.0277, -0.0040] | 无关图片的干扰 |
| VLM text-only − 旧 Prompt | +0.0113 | [+0.0004, +0.0223] | 更换文本 backbone/template 的贡献 |
| Matched − 旧 Prompt | +0.0245 | [+0.0104, +0.0383] | §26 总信息增益 |

## 结论

1. **正确结果图具有真实、显著的增量信息。** Matched 相对同一 Qwen3-VL text-only 提升 +0.0133，3-split paired CI 不跨 0。
2. **§26 的总提升并非全部来自图片。** 更换到 Qwen3-VL text-only 本身贡献 +0.0113；正确图在总增益 +0.0245 中贡献约 54%，其余主要来自更强的文本表征。
3. **模型确实读取并依赖图像内容。** 错配图相对 text-only 降低 -0.0162，而 matched−shuffled 达 +0.0295。如果 embedding 只编码 prompt，不会出现这一模式。
4. **可以继续做轻量化和 conditional probe。** 因果对照已经通过，不需要转向更复杂 Router；下一步应减少 VLM 读取图像的成本，并只在值得的 prompt 上触发。

## seed=42 复核

| 条件 | Router quality |
|---|---:|
| 旧 Prompt（cascade） | 0.3755 |
| VLM text-only（cascade） | 0.3896 |
| Shuffled image（cascade） | 0.3701 |
| Matched image（cascade） | 0.3997 |

## 数值审计

- 两个新缓存均为 pool 2400 / test 800 / 4096 维，无 NaN/Inf，并与 §26 instruction 逐条一致。
- shuffled pool/test 各自一一置换，分别有 2400/800 张唯一图片，且不存在 prompt 与 donor instruction 相同的样本。
- 所有冻结策略实际平均成本为 10（仅有浮点误差）；策略选择阶段不读取 test true scores。
