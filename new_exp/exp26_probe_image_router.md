# §26 结果图评价 embedding 辅助 Router

[返回实验目录](README.md)。结果与设计已合并；设计中的可选扩展不表示已运行。

<a id="exp26-result"></a>

<!-- report:exp26-result:start -->
## §26 结果图评价 embedding 辅助 Router

### 设置

- GenEval2 GM、完整 A10、预算 `E[c]≤10`，Ridge-only。
- 每条 prompt 先用 `flux4b_4steps / seed_0` 生成 probe 图；Flux 成本 0.455 已计入，选择 Flux 时复用该图。
- Qwen3-VL-8B-Instruct 同时读取原 prompt 和 probe 图。不生成评价文字或 vocabulary logits，只取固定评价 query 最后一个非 padding token 的 4096 维 hidden state。
- 目标标签仍是每动作 5 个 image seeds 的平均质量。训练划分为 2200/200，测试 800；split seeds 为 42, 43, 44。
- Qwen evaluator 自身计算成本尚未换算为 TFLOPs，因此以下端到端收益是乐观上界。

![probe image router](figs/exp26_probe_image_router.svg)

### seed=42 主结果

| Router 输入 | Test macro R² | Top-1 action acc | Router quality | 实际成本 |
|---|---:|---:|---:|---:|
| Prompt（原成本） | 0.3972 | 0.2150 | 0.3788 | 10.0000 |
| Prompt（计 probe 成本） | 0.3972 | 0.2150 | 0.3755 | 10.0000 |
| **QEval embedding only** | **0.4737** | **0.2437** | **0.3997** | 10.0000 |
| Prompt + QEval | 0.4707 | 0.2238 | 0.3970 | 10.0000 |
| Prompt + true Flux score（不可部署） | 0.4971 | 0.2387 | 0.4139 | 10.0000 |

QEval-only 是正式主条件；该 embedding 已由 prompt+image 联合编码。直接再拼 4096 维 prompt embedding 将维度翻倍，seed=42 的质量反而从 0.3997 降到 0.3970，因此只保留为拼接消融。

### 三种收益口径

| 比较 | seed=42 | 95% CI | 3-split mean | 3-split paired 95% CI |
|---|---:|---|---:|---|
| 信息增益：QEval(cascade) − Prompt(cascade) | +0.0242 | [+0.0098, +0.0398] | +0.0245 | [+0.0112, +0.0387] |
| 乐观端到端：QEval(cascade) − Prompt(original) | +0.0209 | [+0.0050, +0.0353] | +0.0227 | [+0.0097, +0.0373] |
| Learned gain：QEval(cascade) − LP(cascade) | +0.0258 | [+0.0129, +0.0397] | +0.0294 | [+0.0184, +0.0402] |

### 诊断

- 相对 Flux4B 的 pairwise 动作排序准确率从 0.6349 提高到 0.7006；相对 Qwen50 从 0.6294 提高到 0.6426。
- 不是只靠改变全局动作比例：固定为 QEval router 的同一动作 allocation、但随机分配给 prompt 时，质量只有 0.3328；实例级路由达到 0.3997，差 +0.0669。
- 按 probe 真值质量分桶：

| Flux probe quartile | n | Prompt(cascade) | QEval(cascade) | 信息增益 |
|---|---:|---:|---:|---:|
| Q1 | 200 | 0.2145 | 0.2468 | +0.0323 |
| Q2 | 200 | 0.2692 | 0.2814 | +0.0122 |
| Q3 | 200 | 0.3675 | 0.3716 | +0.0041 |
| Q4 | 200 | 0.6509 | 0.6990 | +0.0481 |

### 结论

这个预实验支持原假设：prompt-only 的信息确实不够，快速生成结果能暴露 prompt 本身无法体现的实例难度和生成失败模式。QEval embedding 对全部 10 个动作的预测均有帮助，其中 Flux4B 动作 R² 提升最大（约 0.357→0.685）。

但它目前只证明了信息价值，尚未证明最终部署净收益。Flux probe 成本已经完整计入，而 Qwen3-VL-8B evaluator 的 TFLOPs 尚未计入；8B VLM 若每次都运行，很可能过重。下一步应先准确核算 evaluator FLOPs，再决定采用轻量 VLM、低层/低分辨率特征，或把 QEval embedding 蒸馏到小型 image-prompt encoder。

### 数值审计

- seed=42/43/44 的所有冻结路由策略实际平均成本均为 10（浮点误差小于 `2e-15`）。
- λ 与混合比例仅由 prediction 和 action cost 选择；test true score 只在策略冻结后进入评价。
- 3200 个 probe 图均使用 seed_0；同一 prompt 没有跨 train/test 泄漏。
<!-- report:exp26-result:end -->

## 产物

- [结果JSON](results/exp26_probe_image_router_gm.json)
- [运行脚本](run_exp26_probe_image_router.py)
- [制图与报告脚本](plot_exp26_probe_image_router.py)

<a id="exp26-design"></a>

<!-- report:exp26-design:start -->
## §26 快速生成结果辅助 Router：预实验设计

### 核心问题

Prompt-only router 的信息是否不足？如果先运行最便宜动作 `flux4b_4steps`
（cost=0.455），再把它生成的图像作为额外观测，能否更准确地选择最终动作，并在计入
probe 成本后提高端到端质量？

现有图像已经齐全，不需要重新推理：

- GenEval2：2400 pool + 800 test，每条均有 5 个 flux4b seed 图像。
- UniGen：2400 pool + 600 test，每条均有 5 个 flux4b seed 图像。

### 最小预实验范围

- 数据：先只跑 **GenEval2 GM**。它的 prompt-only learned gain 最弱、Oracle gap 最大，最适合判断
  “prompt 信息不足”假设。
- 动作池：完整 A10。
- 主预算：总成本 `E[c]≤10`。
- split：与 §23/§24 相同，2200 train / 200 validation / 800 test，seed=42；结果为正后再补
  split seeds 43/44。
- Probe 图像：每条 prompt 固定使用 `flux4b_4steps / seed_0 / steps4.png`。
- Predictor：仍然只用 Ridge，保证唯一变化是输入信息。

### 特征对照

| 条件 | 输入 | 是否可部署 | 目的 |
|---|---|---|---|
| P | Qwen prompt embedding | 是 | 当前基线 |
| P-cost | 同 P，但强制计入 probe 成本 | 是 | 单独测 probe 成本惩罚 |
| E | Qwen3-VL 对 prompt+图的评价 hidden embedding | 是 | 主实验 |
| P+E | Qwen prompt embedding + QEval embedding | 是 | 检查显式拼接是否继续有益 |
| P+S* | prompt + flux4b true quality scalar | 否 | 非部署的 oracle-side-information 诊断上界 |

QEval 使用 `Qwen3-VL-8B-Instruct`：固定评价 query 同时包含原 prompt 和 Flux probe 图，不生成
评价文字、不计算 vocabulary logits，直接取 multimodal backbone 最后一个非 padding query token
的 4096 维 hidden state。E 与 P+E 均逐维 StandardScaler，再训练与当前完全相同的 10 个独立
Ridge score heads，alpha 仅按 validation R² 选择。由于 E 已联合编码 prompt 与图，E 为主条件，
P+E 仅作冗余拼接消融。

`P+S*` 不能作为正式结果，因为 true flux4b score 在线不可见。它只回答：如果准确知道 probe
质量，理论上是否有额外决策价值。正式结论只看 E。

### 正确的成本口径

Probe 是每个 prompt 必跑的第一阶段，因此不能继续使用原 A10 成本。

对于最终动作 a：

```text
cost_cascade(a) = 0.455                     if a = flux4b_4steps
                  0.455 + original_cost(a)  otherwise
```

- 最终选择 flux4b 时直接返回已经生成的 probe 图，不能重复收取 0.455。
- 最终选择其他动作时，成本是 probe + 第二阶段动作。
- P-cost 与 E 必须使用同一个 `cost_cascade`，二者差值才是纯信息增益。
- 原始 P 使用原动作成本，E 与原始 P 的差值才是实际端到端收益。
- Qwen evaluator 的计算必须最终单独计入；第一轮若尚未得到统一 TFLOPs，只能标为
  information-value screen，不能宣称最终净收益。

所有 λ/策略混合均使用 §25 的无标签 deploy-time calibration：只能看 prediction 和 cost，不能
根据 test true quality 挑 λ。

### 三个必须分开的收益

1. **图像信息增益**

   `Δinfo = Q(E, cascade cost) − Q(P-cost, cascade cost)`

   回答“看了快速结果后，router 是否真的分得更准”。

2. **端到端净收益**

   `Δe2e = Q(E, cascade cost) − Q(P, original cost)`

   回答图像信息能否抵消额外 0.455 成本。

3. **实例级 learned gain**

   `Q(E, cascade cost) − LP(cascade cost)`

   回答提升是否超越静态动作先验。

### 评价指标

#### 主指标

- budget=10 下的 Router quality。
- `Δinfo`、`Δe2e` 和 E−LP。
- Oracle−Router gap。
- 最终动作分配比例和实际平均成本。

#### 诊断指标

- 每动作 test R²，以及 10 动作 macro R²。
- pairwise ranking accuracy：预测的最终动作是否优于 Qwen50 / flux4b probe。
- oracle action top-1 accuracy（只作诊断，不作为主指标）。
- 按 probe true quality 分桶后的 router quality：检查图像是否主要帮助识别“probe 已经够好”或
  “probe 明显失败”的 prompt。

质量差值全部做 paired bootstrap 1000 次，报告 95% CI。

### 成功标准

预实验判为值得继续，需要同时满足：

1. `Δinfo ≥ +0.005`，且 bootstrap 95% CI 不跨 0；
2. 暂不计 Qwen evaluator 时 `Δe2e ≥ 0`，即先确认信息足以覆盖 probe 生成成本；
3. E 的 learned gain 高于当前 GM 的约 +0.005，并且不是单纯增加 flux4b 返回比例造成的假象。

结果解释：

| 结果 | 结论 | 下一步 |
|---|---|---|
| Δinfo>0，乐观 Δe2e>0 | 图像信息有效且覆盖 Flux probe 成本 | 核算 evaluator TFLOPs，再决定是否扩展 |
| Δinfo>0，Δe2e<0 | 信息有效但 always-probe 太贵 | 做 uncertainty-triggered conditional probe |
| Δinfo≈0，P+S*>0 | probe 有潜力，但当前 QEval 表征不够 | 换评价 token/轻量 VLM/image-text mismatch feature |
| Δinfo≈0，P+S*≈0 | 快速结果对动作选择帮助有限 | 停止该方向 |

### Seed 与标签注意事项

第一轮只用 seed_0 图像，但目标仍是当前五 seed 平均动作质量，测试“一次便宜采样是否能预测该
prompt 的期望动作排序”。若结果为正，再把 5 个 probe seeds 作为重复观测；数据划分必须按
prompt 分组，严禁同一 prompt 的不同 seed 落入 train/test 两侧。
<!-- report:exp26-design:end -->
