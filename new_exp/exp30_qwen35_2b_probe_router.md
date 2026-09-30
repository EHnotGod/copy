# §30 Qwen3.5-2B 融合 Flux 出图信息的 Router

## 设置

- 数据：GenEval2 GM，固定 800 条 test；pool 严格拆成 2000 Ridge train / 200 alpha-val / 200 不使用的 library-val，split seeds 为 42/43/44。
- probe：每条 prompt 先运行 `flux4b_4steps / seed_0`，成本 0.455；最终选择 F4B4 时直接复用 probe 图。
- evaluator：Qwen3.5-2B 同时读取原 prompt 和 320px probe 图，不生成文字或 vocabulary logits，只取 evaluation query 最后一个非 padding token 的 2048 维 hidden state。
- Router 固定为 Ridge。比较完整 A10、预筛选核心库 `QTTS+F10+B10`，以及核心库加 probe 动作 `QTTS+F10+B10+F4B4`。
- 每个条件均只用预测值和动作成本冻结路由策略，test label 只在冻结后评价。

## 计算成本

- 320px 输入为 400 个视觉 patch、100 个 merged image tokens；3200 条 query 的平均总序列长度为 154.96。
- 解析 dominant forward cost 为 **0.6948 TFLOPs/样本**，计入视觉 tower、dense attention、gated-delta recurrence 和 MLP；不含 norm、activation、softmax、预处理等小项。
- 旧 Qwen3-VL-8B@320px 为 2.5215 TFLOPs；2B 是其 **27.6%**。
- H100 单卡、batch=16、当前 torch fallback 环境下，160 条稳态前向共 8.928 秒，即 **55.8 ms/样本**；包含模型预热和加载的命令总耗时为 29.27 秒。
- `charged` 条件向每个动作完整加收 0.6948，并同时计入 Flux probe 0.455；没有减去旧 Prompt Router 本来需要的 Qwen3-Embedding 成本，因此是偏保守口径。

## 主结果

| 动作库 | Prompt 原成本 | 2B text + probe成本 | 2B 图文 info-only | 2B 图文 charged | 8B@320 图文 info-only |
|---|---:|---:|---:|---:|---:|
| 完整 A10 | 0.3816 | 0.3806 | 0.3929 | **0.3909** | 0.3965 |
| `QTTS+F10+B10` | **0.3916** | 0.3850 | 0.3924 | 0.3888 | 0.3937 |
| `QTTS+F10+B10+F4B4` | 0.3870 | 0.3809 | 0.3977 | **0.3946** | 0.3987 |

三列图文结果中，`info-only` 已计 Flux probe、未计 evaluator；`charged` 两者都计。所有冻结策略的实际平均成本均为 10。

## 配对差值

| 动作库 | 图像信息：2B matched − 2B text | 同 probe 成本：2B matched − Prompt | 完整计费：2B charged − Prompt 原成本 | 2B matched − 8B@320 matched |
|---|---:|---:|---:|---:|
| 完整 A10 | **+0.0123** `[+0.0006,+0.0240]` | **+0.0137** `[+0.0024,+0.0264]` | +0.0094 `[-0.0025,+0.0212]` | −0.0036 `[-0.0136,+0.0074]` |
| `QTTS+F10+B10` | +0.0075 `[-0.0027,+0.0179]` | +0.0038 `[-0.0057,+0.0136]` | −0.0028 `[-0.0126,+0.0071]` | −0.0013 `[-0.0105,+0.0081]` |
| `QTTS+F10+B10+F4B4` | **+0.0168** `[+0.0049,+0.0276]` | **+0.0126** `[+0.0018,+0.0230]` | +0.0076 `[-0.0031,+0.0185]` | −0.0010 `[-0.0116,+0.0093]` |

区间为三个 split 预测取均值后的 paired prompt bootstrap 95% CI，2000 次。

## 诊断与结论

1. **2B 确实读到了图片中的新增信息。** 完整 A10 的 Macro R² 从 text-only 0.4031 提升到 matched 0.4478；F4B4 动作的 R² 从 0.3819 大幅升到 0.6096，是最明显的来源。
2. **图片信息最适合判断 probe 图能否直接交付。** 核心三动作不包含 F4B4 时，matched 增益不显著；把 F4B4 加回后，图像增量达到 +0.0168 且 CI 不跨 0。
3. **2B 保留了大部分 8B 信息价值。** 同为 320px、同 split、同 Ridge 时，三种动作库中 2B 均只低 0.0010–0.0036，差值全部不显著，而解析 FLOPs 降低约 72.4%。
4. **当前只能声称信息增益稳定，不能声称净收益稳定。** 计入 0.455 probe 和 0.6948 evaluator 后，A10 / core4 相对 Prompt 仍为正向点估计 +0.0094 / +0.0076，但 CI 跨 0；core3 为 −0.0028。
5. **不要再拼原 Prompt embedding。** `Prompt + 2B matched` 在 A10/core4 分别比 2B matched 低 0.0025/0.0033；完整计费后也没有改善。Qwen3.5 图文 embedding 已包含 prompt 信息，直接拼接只增加维度和估计方差。

推荐部署候选是 **2B@320 + core4**，因为它最符合“先看快速 Flux 图，再判断直接交付还是转到 QTTS/F10/B10”的因果结构；但正式结论仍需独立 held-out 或更多 benchmark 复验 charged gain。

## 产物与审计

- 特征缓存：`results/exp30_qwen35_2b_qeval_gm_seed0_320px.npz`、`results/exp30_qwen35_2b_text_only_gm.npz`。
- 数值：[results/exp30_qwen35_2b_probe_router.json](results/exp30_qwen35_2b_probe_router.json)；FLOPs：[results/exp30_qwen35_2b_flops.json](results/exp30_qwen35_2b_flops.json)。
- 缓存均为 2400 pool + 800 test、2048 维、无 NaN/Inf/零行，matched/text instruction 与 GenEval2 数据逐条一致。
- 所有 condition 的 allocation 和为 1，平均动作成本不超过 budget=10；JSON 无 NaN/Inf。
