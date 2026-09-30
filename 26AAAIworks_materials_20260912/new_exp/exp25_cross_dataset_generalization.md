# §25 Router 跨数据集泛化

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

§25 已完成 CPU 重算，当前结果为 [JSON](results/prompt_disjoint_v1/exp25_cross_dataset_generalization.json)。UniGen 使用 pool2000/raw600，所有新预测、基线与汇总来自同一数据版本。历史正文中的旧 UniGen 分数不再作为当前结论。GM/AM 是同一 prompt 来源的两种评分。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


[返回实验目录](README.md)。结果与设计已合并；设计中的可选扩展不表示已运行。

<a id="exp25-result"></a>

<!-- report:exp25-result:start -->
## §25 Router 跨数据集泛化

### 设置

- 完整 A10，预算 `E[c]≤10`，Ridge-only。
- source scaler、Ridge 和 alpha 只使用 source train/validation。
- target 侧仅用预测分数和成本选择 λ/混合策略；冻结策略后才读取 target true scores。
- split seeds：42, 43, 44, 45, 46；paired bootstrap：1000 次。
- GM/AM prompt 完全相同，二者互转属于 metric shift；与 UniGen 互转才是 domain shift。

![cross-dataset heatmap](figs/exp25_cross_dataset_heatmap.svg)

### 主要结果

| train→test | shift | Q50 | Source-LP | Cross-Ridge | 总收益 | Zero-shot learned | 95% CI | Target-LP | In-domain learned | Oracle |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| GM→GM | in-domain | 0.2482 | 0.3755 | 0.3808 | +0.1326 | +0.0053 | [-0.0058, +0.0166] | 0.3755 | +0.0053 | 0.5684 |
| GM→AM | metric shift | 0.7300 | 0.7919 | 0.7891 | +0.0591 | -0.0028 | [-0.0085, +0.0029] | 0.7919 | +0.0088 | 0.8665 |
| GM→UG | domain shift | 0.7518 | 0.8566 | 0.8326 | +0.0808 | -0.0240 | [-0.0340, -0.0139] | 0.8618 | +0.0550 | 0.9693 |
| AM→GM | metric shift | 0.2482 | 0.3755 | 0.3856 | +0.1374 | +0.0101 | [-0.0023, +0.0215] | 0.3755 | +0.0053 | 0.5684 |
| AM→AM | in-domain | 0.7300 | 0.7919 | 0.8007 | +0.0707 | +0.0088 | [+0.0034, +0.0136] | 0.7919 | +0.0088 | 0.8665 |
| AM→UG | domain shift | 0.7518 | 0.8566 | 0.8274 | +0.0756 | -0.0292 | [-0.0399, -0.0184] | 0.8618 | +0.0550 | 0.9693 |
| UG→GM | domain shift | 0.2482 | 0.3683 | 0.3631 | +0.1149 | -0.0052 | [-0.0159, +0.0051] | 0.3755 | +0.0053 | 0.5684 |
| UG→AM | domain shift | 0.7300 | 0.7947 | 0.7859 | +0.0559 | -0.0088 | [-0.0130, -0.0043] | 0.7919 | +0.0088 | 0.8665 |
| UG→UG | in-domain | 0.7518 | 0.8618 | 0.9167 | +0.1650 | +0.0550 | [+0.0450, +0.0657] | 0.8618 | +0.0550 | 0.9693 |

### 结论

1. **真正的跨 prompt-domain learned routing 没有泛化。** GM→UG 为 -0.0240，AM→UG 为 -0.0292，均显著低于 Source-LP。
2. **反向迁移也没有稳定收益。** UG→GM 为 -0.0052（CI 跨 0），UG→AM 为 -0.0088（显著为负）。
3. **迁移的主要是动作先验，不是 prompt-level 选择能力。** 所有跨域 Cross-Ridge 仍高于 Qwen50，但没有超过使用相同 source 标签的 Source-LP。
4. **同域 router 确实可学。** AM 和 UniGen 的对角线 learned gain 显著为正；GM 为小幅正值但 CI 跨 0。
5. **GM↔AM 也不稳定。** 即使 prompt 完全相同，GM→AM 没有超越 Source-LP；AM→GM 为正但 CI 跨 0，说明评分口径变化已经足以破坏部分动作排序。

因此当前证据支持：A10 的全局动作组合具有一定跨域可迁移性，但 Ridge 学到的实例级路由规律明显依赖训练数据集和评分口径。跨数据集部署时，Source-LP 比直接迁移 Ridge 更稳妥；若需要 learned routing，应至少做少量目标域校准。

### 数值审计

- 9 个方向、5 个 seeds 的部署策略成本均为 10.0000。
- seed=42 的自实现 Ridge 与 §23 sklearn 预测缓存 RMSE 均小于 `1.3e-6`。
- λ/混合策略选择函数不接受 target true scores；真实标签只进入冻结后的评价函数。
<!-- report:exp25-result:end -->

## 产物

- [结果JSON](results/exp25_cross_dataset_generalization.json)
- [运行脚本](run_exp25_cross_dataset_generalization.py)
- [制图与报告脚本](plot_exp25_cross_dataset_generalization.py)

<a id="exp25-design"></a>

<!-- report:exp25-design:start -->
## §25 Router 跨数据集泛化实验设计

### 研究问题

在动作池、成本、prompt embedding 和路由算法完全不变时，只使用源数据集标签训练的
Ridge router，能否零样本迁移到目标数据集？

本实验只研究 **closed-action / cross-dataset** 泛化。当前 router 为每个动作训练独立
Ridge head，无法为训练时未见过的新动作产生预测，因此不把 unseen-action 纳入本实验。

### 数据域

| 代码 | 数据 | 含义 |
|---|---|---|
| GM | GenEval2 GM | 几何均值评分 |
| AM | GenEval2 AM | 算术均值评分 |
| UG | UniGenBench | 不同 prompt/评分分布 |

GM 和 AM 的 2400 条 pool prompt、800 条 test prompt 均逐条相同，只是标签聚合方式不同。
因此 GM↔AM 是 **metric/label shift**，不能单独作为跨数据集泛化证据。
GM/AM↔UG 才是主要的 **domain shift** 结果。

### 固定设置

- 动作池：完整 A10，10 个动作和 §23/§24 完全一致。
- 主预算：`E[c]≤10`，即 qwen50 anchor cost。
- Predictor：仅 Ridge；每个动作一个独立回归头。
- 特征：按 instruction 对齐的 Qwen3 embedding。
- 源域划分：2200 train / 200 validation，seed=42；再用 5 个 split seed 做稳健性复现。
- 目标域：只使用固定 test（GenEval2 800，UniGen 600）做最终评价。
- 主实验不使用任何目标域质量标签训练、选超参数或选择 λ。

### 迁移矩阵

完整运行 3×3 矩阵：

| train \ test | GM | AM | UG |
|---|---:|---:|---:|
| GM | in-domain control | metric shift | domain shift |
| AM | metric shift | in-domain control | domain shift |
| UG | domain shift | domain shift | in-domain control |

对角线是同域上界；四个 GM/AM↔UG 方向是论文主结果；GM↔AM 只作为标签口径变化诊断。

### 单个 source→target 的执行流程

1. 仅在 source train 上拟合 `StandardScaler`。
2. 为 10 个动作分别训练 Ridge；`alpha∈{1,10,100,1000}` 仅按 source validation R² 选择。
3. 用 source scaler 和 Ridge 直接预测 target test 的 10 维动作分数。
4. 在 target predictions 上根据动作成本选择预算策略，但不读取 target true scores。
5. 冻结动作选择后，才读取 target true scores 计算最终质量。

#### 无标签预算校准

不能直接复用 §23 的 `best_policy_at_budget`：该函数会用 test true quality 从 λ 曲线中挑最好点，
用于跨域实验会产生目标域标签泄漏。

本实验使用 deploy-time calibration：

1. 扫描 λ，对 target prediction 执行 `argmax(pred−λ·cost)`。
2. 只根据预测动作产生的实际成本，找到预算 `B=10` 两侧相邻的两个 λ policy。
3. 按两者成本做解析混合，使 `E[c]=B`；若已有 policy 低于预算且不存在高侧点，则直接使用该 policy。
4. λ、混合比例及所有 tie-break 均不得使用 target true quality。

目标域的 prompt embedding、预测分数和动作成本在部署时可见，因此该预算校准不算标签适配。

### 对照方法

| 方法 | 使用的标签 | 作用 |
|---|---|---|
| Qwen50 | 无 | 固定动作 anchor |
| Source-LP | source train | 源域 action mean 学到的静态混合，在 target test 解析评估 |
| Cross-Ridge | source train/val | 主要零样本迁移结果 |
| Target-LP | target train | 目标域静态先验参考，不属于 zero-shot |
| In-domain Ridge | target train/val | 有监督目标域 router 上界 |
| Oracle | target test true scores | 系统机会理论上界 |

Cross-Ridge 必须与 **Source-LP** 比较，二者都只使用 source 标签；不能只拿 Target-LP 作为
zero-shot baseline，否则混入了目标域监督差异。

### 主要指标

对每个 source→target 方向报告：

- `Q_cross`：Cross-Ridge target quality。
- `total gain = Q_cross − Qwen50_target`：跨域 router 相对固定动作的总收益。
- `zero-shot learned gain = Q_cross − SourceLP_target`：是否超越源域静态动作先验，作为主指标。
- `in-domain learned gain = Q_indomain − TargetLP_target`：目标域可学收益参考。
- `transfer retention = zero-shot learned gain / in-domain learned gain`：迁移保留率；当分母≤0或很小时只报原始差值，不解释比率。
- `oracle gap = Q_oracle − Q_cross`：尚未利用的系统机会。
- 目标动作分配比例及实际 `E[c]`：检查迁移失败是否来自动作选择塌缩。

每个质量差值在 target prompt 上做 paired bootstrap 1000 次，报告 95% CI；再汇报 5 个
source split seed 的均值与标准差。

### 判定标准

- **有跨域泛化**：Cross-Ridge 显著高于 Source-LP，且 bootstrap CI 不跨 0。
- **只有动作先验迁移**：Cross-Ridge 高于 Qwen50，但不高于 Source-LP。
- **负迁移**：Cross-Ridge 低于 Source-LP，尤其低于 Qwen50。
- **系统有机会但 router 不泛化**：target Oracle 明显提高，而 Cross-Ridge/Source-LP 无差异。

重点结论以四个 GenEval2↔UniGen 方向为准。GM↔AM 即使很好，也只能说明 router 对评分聚合
口径变化有鲁棒性，不能说明跨 prompt domain 泛化。

### 可选的少样本扩展

若 zero-shot 结果较差，再增加 target `k∈{25,100,300}` 的校准曲线。冻结 source Ridge，
只用 k 条 target 样本为每个动作拟合 `y_target=a_j·pred_source+b_j`，比较其与 target-only
Ridge 的样本效率。该扩展与 zero-shot 主表分开报告。
<!-- report:exp25-design:end -->

</details>
