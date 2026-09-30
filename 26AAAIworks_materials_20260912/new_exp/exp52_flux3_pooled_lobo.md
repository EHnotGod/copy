# §52 Flux三动作混合Router对未见Benchmark的泛化

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

全部 5 个 seeds、5 个目标和 3 种标签变换均已在清洗后的统一数据版本上重跑。专用 Router 来自同版本 §51；UniGen 作为目标时 pooled fit 不使用 UniGen 标签。

| 目标 | raw-score Router | Pooled LP | Target LP | Router−Target LP (95% CI) |
|---|---:|---:|---:|---|
| GM | 0.324298 | 0.323623 | 0.323623 | +0.000676 [-0.002443, +0.003890] |
| AM | 0.752206 | 0.752023 | 0.752023 | +0.000183 [-0.001030, +0.001446] |
| UniGen | 0.823391 | 0.821896 | 0.821896 | +0.001495 [-0.002436, +0.005761] |
| CoreBench | 0.506050 | 0.507022 | 0.507022 | -0.000973 [-0.003260, +0.001303] |
| WISE | 0.775360 | 0.772644 | 0.772644 | +0.002716 [-0.001191, +0.006816] |

Raw-score 在 0/5 项评分上显著超过 Target LP。当前五个置信区间均跨零，旧版“WISE 显著正迁移”已不适用于修复后的主表；未显著不等于证明两者等价。

另外两种变换的全部分数见 [JSON](results/prompt_disjoint_v1/exp52_flux3_pooled_lobo.json)。![修复后的 LOBO](results/prompt_disjoint_v1/exp52_flux3_pooled_lobo.png)

这里留出的是评分任务；GM↔AM 仍属于同一 prompt 来源，不把它们算作两个独立的新数据域。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

> 2026-09-11 数据核对：本页涉及旧 UniGen gen2400/raw600 的训练或目标侧监督基线时，两者有 400 条 prompt 重合，不能作为独立 prompt 测试证据。纯 source→UniGen 的 pooled 模型是否受影响须与其目标侧对照分别判断。见 [核对记录](../修改日志.md#数据与对齐核对)。


## 问题

固定动作空间 `Flux-9B {10 steps, 50 steps, TTS/edit}`。每次完整留出一个目标benchmark，
只混合其余四个benchmark训练一个Router，目标benchmark标签不进入scaler、Ridge、alpha选择或
策略冻结。检验该Router能否在未见benchmark的新prompt上获得实例级routing收益。

## 协议

- 五个目标：GM、AM、UniGen、CoreBench、WISE。
- 每个source平衡抽取300 train + 50 alpha-val；每个目标的混合训练规模为1200 + 200。
- 5个split seeds：42–46；Qwen3 prompt embedding；预算10。
- 主比较按更严格的target-aware口径固定为：
  1. `Pooled Router − target-seen Static LP`；
  2. `Pooled Router − target-seen Specialist`。
  两个对比对象都使用了目标benchmark的300条训练标签。matched pooled LP仅保留为诊断。
- 同时测试raw score、相对Flux-50 advantage、按source尺度归一化advantage。

![Flux3 LOBO transfer](results/exp52_flux3_pooled_lobo.png)

## Raw-score混合Router结果

| Held-out target | Target Static LP | Pooled Router | Router−LP (95% CI) | Target Specialist | Router−Specialist (95% CI) |
|---|---:|---:|---:|---:|---:|
| GM | 0.323623 | 0.325112 | +0.001489 [-0.001429,+0.004526] | 0.322760 | +0.002352 [-0.000777,+0.005425] |
| AM | 0.752023 | 0.752871 | +0.000847 [-0.000375,+0.002117] | 0.753175 | -0.000305 [-0.001694,+0.001031] |
| UniGen | 0.821896 | 0.823391 | +0.001495 [-0.002436,+0.005761] | 0.829798 | **-0.006408 [-0.011880,-0.000945]** |
| CoreBench | 0.507022 | 0.505928 | -0.001095 [-0.003350,+0.001270] | 0.524027 | **-0.018099 [-0.024703,-0.011979]** |
| WISE | 0.772644 | 0.776912 | **+0.004269 [+0.000290,+0.008608]** | 0.784069 | **-0.007156 [-0.013716,-0.000485]** |

相对target-seen Static LP的macro gain为`+0.001401`；4/5目标为正点估计，只有WISE显著为正，
0/5显著为负。本动作空间和预算下，四源matched pooled LP碰巧与target Static LP选择了相同的
静态动作比例，因而两者质量数值相等；正式结论仍按target-seen LP口径表述。

## 标签变换敏感性

| 方法 | Macro learned gain | 正目标 | 显著正 | 显著负 |
|---|---:|---:|---:|---:|
| raw score | +0.001401 | 4/5 | 1/5 | 0/5 |
| Flux-50 advantage | +0.001401 | 4/5 | 1/5 | 0/5 |
| normalized Flux-50 advantage | +0.001436 | 4/5 | 0/5 | 0/5 |

减去Flux-50公共基线不改变策略；尺度归一化也没有稳定增强跨域泛化。

## 结论

1. **相对见过目标数据的Static LP，存在有限泛化。** WISE是唯一有统计证据的未见benchmark
   正迁移；GM、AM、UniGen只有小幅正点估计，CoreBench小幅负，四者CI均跨0。
2. **相对见过目标数据的Specialist，不能认为混合Router普遍泛化。** GM/AM与specialist统计
   持平；UniGen、CoreBench、WISE分别低`0.0064/0.0181/0.0072`，且差距显著。
3. 与§51联合监督结果组合起来看：已知benchmark纳入joint训练后，一个Router可以达到
   specialist水平；完全留出benchmark时，只能期待约千分之一量级的平均迁移收益。

## 产物

- [results/exp52_flux3_pooled_lobo.json](results/exp52_flux3_pooled_lobo.json)
- [results/exp52_flux3_pooled_lobo.png](results/exp52_flux3_pooled_lobo.png)
- `run_exp52_flux3_pooled_lobo.py`

</details>
