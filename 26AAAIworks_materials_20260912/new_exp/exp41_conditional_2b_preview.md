# §41 普通 Router 不确定度触发 2B Preview

## 协议

本实验在已有 GM 的真实 Qwen3.5-2B@320 特征上验证：是否只对普通 prompt Router 的不确定
case 运行 Flux4B preview + 2B evaluator，比 always-preview 更具成本效益。

- 动作库：完整 A10、`QTTS+F10+B10`（core3）、以及加 F4B4 的 core4；budget=10。
- 每个 split 使用2000条训练 Ridge、200条选 alpha、独立200条选择 gate type/coverage，最后在
  800条 test 上冻结评价；seeds 42/43/44。
- Gate 候选覆盖率预先固定为 `{0,25%,50%,75%,100%}`。
- `prompt_margin`：普通 Router 的 cost-adjusted top-2 utility margin，margin 小表示不确定。
- `committee_expected_regret`：5个只看 prompt 的 Ridge heads 对当前动作选择的平均预期后悔；
  committee 只增加很小的矩阵向量预测，不运行额外生成/VLM。
- 被 gate 的 case 换成真实2B图文预测，并逐 case 计入 Flux preview `0.455` 和2B evaluator
  `0.6948`；若最终选 F4B4，直接复用 preview 图。
- Gate type/coverage 只由独立 library-validation 标签选择；test 标签不参与 gate 或 λ 策略选择。

## Validation 选择后的主结果

| 动作库 | 平均 test preview coverage | 相对 Prompt 原成本 | 95% CI | Always-preview |
|---|---:|---:|---|---:|
| 完整 A10 | 66.2% | **+.0098** | **[+.0013,+.0188]** | +.0094 `[−.0029,+.0212]` |
| core3 | 5.8% | +.0009 | `[−.0012,+.0029]` | −.0028 `[−.0127,+.0067]` |
| core4 | 25.5% | +.0040 | `[−.0006,+.0086]` | +.0076 `[−.0028,+.0187]` |

完整 A10 的三个 split 分别选择 margin 50% / 100% / 50%，实际 coverage 为42.4% / 100% /
56.2%。平均只 preview 66.2%的 case，完整计费收益首次达到显著；同数据、同成本下
always-preview 仍不显著。

core3 中两个 split 直接选择0 coverage，说明不含 F4B4 时 preview 大多不值得。core4 的联合
gate/coverage 选择受200条 validation 波动影响，主点为正但尚不显著。

## 固定覆盖率诊断

固定 coverage 不读取 validation quality 选择该点，因此更清楚展示成本—收益曲线；以下 CI
未做多重比较校正，应作为诊断而非新增 confirmatory claims。

### 完整 A10

| Gate | 实际 coverage | 增益 | 95% CI |
|---|---:|---:|---|
| Committee 25% | 28.0% | +.0071 | [+.0016,+.0129] |
| Committee 50% | 51.4% | +.0093 | [+.0013,+.0177] |
| Margin 50% | 48.7% | +.0090 | [+.0019,+.0161] |
| Margin 75% | 70.3% | +.0100 | [+.0005,+.0203] |
| Always preview | 100% | +.0094 | [−.0029,+.0212] |

若还关心延迟/吞吐，Committee 25% 是最强低覆盖 Pareto 点：只 preview 约28%的 case，保留
约76%的 always-preview 点收益，而且 paired CI 不跨0。若只优化 budget=10 下质量，50%–75%
coverage 更高。

### Core4

| Gate | 实际 coverage | 增益 | 95% CI |
|---|---:|---:|---|
| Committee 25% | 28.4% | +.0052 | [−.0001,+.0104] |
| Committee 50% | 49.1% | +.0088 | [+.0021,+.0158] |
| Margin 50% | 44.2% | +.0051 | [−.0008,+.0113] |
| Margin 75% | 70.4% | +.0093 | [+.0010,+.0182] |
| Always preview | 100% | +.0076 | [−.0028,+.0187] |

Core4 的固定 Committee 50% 与 Margin 75% 都优于 always-preview 的统计稳定性，再次支持
“只看难判断 case”。

## 普通 Router 不确定度是否更好

1. **是，普通 Router 的不确定度足以形成有效 gate。** A10 的 margin validation-selected
   cascade 获得显著 +.0098；不需要先运行另一个视觉模型判断是否 preview。
2. **简单 margin 是更稳的默认值。** 它连续、无需额外 heads，并在 A10 主结果上优于 committee
   的 validation-selected 结果。
3. **Committee 在低覆盖区有价值。** A10 约28% coverage 已达到 +.0071；core4 约49%
   coverage 达到 +.0088。其 expected-regret 分数存在并列，requested 75% 在部分 split 会退化
   到100%，因此阈值校准还需改进。
4. **Preview 的主要价值与 F4B4 复用有关。** core3 最优策略几乎不 preview；加入 F4B4 后
   conditional preview 才形成清晰收益。

## 结论与限制

当前 GM 证据已经支持：**真实2B成本下，partial preview 比 always-preview 更值得；普通 Router
不确定度可以决定哪些 case 使用 preview。** 推荐优先验证两个冻结候选：A10+Margin 50%，以及
core4+Committee 50%。

目前仍只有 GM、3个 split；固定覆盖率表还属于多点诊断。要形成跨 benchmark 结论，需要为
AM、UniGen、CoreBench、WISE生成同模板2B preview embedding，并用相同冻结 gate 复验。

## 产物与审计

- 脚本：`run_exp41_conditional_2b_preview.py`。
- 数值：[results/exp41_conditional_2b_preview.json](results/exp41_conditional_2b_preview.json)。
- 3份五头 prompt committee 缓存：`results/exp41_prompt_committee_gm_seed*.npz`。
- 最大策略成本 `10.000000000000002`（浮点误差），全部满足 budget。
- 100% preview quality 与 §30 charged 结果逐 split 完全一致，最大误差0。
