# §55–56：Flux4 公共 Router 的数据量实验

<!-- report:integrity-repair:start -->
<a id="corrected-results"></a>

## 修复后的当前结果（2026-09-11）

§55 原协议已重跑；另用 §56 的固定测试划分补齐 100/200/300/1000 同口径曲线。四个点使用相同的 target LP1000、alpha50、5 个 seeds 和嵌套 source train 前缀。

测试支持固定为 GM800、AM800、UniGen600、CoreBench400、WISE200；UniGen pool2000，CoreBench pool1080，WISE pool1200。


| 每源训练数 | 平均 Router | 固定 Target LP | Router−LP | 结果 |
|---:|---:|---:|---:|---|
| 100 | 0.632558 | 0.640257 | -0.007699 | [JSON](results/prompt_disjoint_v1/exp56_flux4_pooled_1k_n100.json) |
| 200 | 0.632464 | 0.640257 | -0.007793 | [JSON](results/prompt_disjoint_v1/exp56_flux4_pooled_1k_n200.json) |
| 300 | 0.633293 | 0.640257 | -0.006964 | [JSON](results/prompt_disjoint_v1/exp56_flux4_pooled_1k_n300.json) |
| 1000 | 0.635777 | 0.640257 | -0.004480 | [JSON](results/prompt_disjoint_v1/exp56_flux4_pooled_1k.json) |

1000 条时平均质量仍低于 LP；不把扩量解释为已经解决未见评分任务上的误路由。

![统一口径曲线](results/prompt_disjoint_v1/exp56_matched_data_curve.png)

§55 的原测试支持与 LP300 保留为另一协议：[JSON](results/prompt_disjoint_v1/exp55_flux4_pooled_router_data_curve.json)。它不参与上述四点连线。所有交集数来自实测断言。
<!-- report:integrity-repair:end -->

<details>
<summary>修复前记录：旧数据或旧评价口径，仅供追溯</summary>

[返回实验目录](README.md)。合并每源 100/200/300 与每源 1000 的记录，表格按原实验分别保留。

## 核对后的结论边界

- 两个实验的已存 JSON 都未显示稳定、普遍的跨目标正收益。
- §56 改变 CoreBench/WISE 的 construction/test 支持和 Target LP 标签量；两个 macro 不能直接作为同口径学习曲线比较。
- 旧 UniGen 2400/600 有 400 条 prompt 重合：留出 UniGen 的 pooled 模型没有使用 UniGen pool，但 Target LP/specialist 的目标侧对照不是独立 prompt 划分；其他目标与 UniGen 的身份关系也应单独检查。不能将该问题解释为 pooled UniGen 模型必然读取了自身 test 标签。
- 原 §56 JSON 的 overlap=0 不是全域实际核验结果。详细证据见 [核对记录](../修改日志.md#数据与对齐核对)。

<a id="exp55"></a>

<!-- report:exp55:start -->
## §55 蒸馏Flux动作扩容后的跨Benchmark数据量实验

### 目的与动作空间

在§54纯Flux三动作数据量实验上新增已有完整五benchmark评分的快速蒸馏动作
`FLUX.2-klein-4B @ 4 steps`，检验动作空间丰富后公共Router的数据量趋势：

| 动作 | 成本 |
|---|---:|
| FLUX.2-klein-4B 4-step | 0.455 |
| Flux-9B 10-step | 4.7 |
| Flux-9B 50-step | 23.1 |
| Flux-9B TTS/edit | 44.9 |

五个benchmark的pool/test分数均已存在且有限，因此本实验不重新生成图片，只复用逐prompt评分。

### 协议

- 外层依次完整留出GM、AM、UniGen、CoreBench、WISE。
- 公共pooled Router只混合其余四个source，benchmark等权。
- 每source使用`n={100,200,300}`条Ridge train和50条alpha validation。
- 5个split seeds（42–46），Qwen3 prompt embedding，平均budget=10。
- Target Static LP见过目标benchmark的300条训练标签；目标标签不参与公共Router拟合或策略冻结。
- §39已有的n=300 A7 raw-score Ridge各action head相互独立，因此直接抽取四个Flux head，与重新拟合
  Flux4完全等价；n=100/200从头拟合Flux4。

![Flux4 data curve](results/exp55_flux4_pooled_router_data_curve.png)

### 数据量结果

| 每source训练量 | 总train / alpha-val | Flux4 Router macro | Target LP macro | Router−LP | 正目标 | 显著正/负 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 400 / 200 | 0.628872 | 0.635442 | -0.006569 | 2/5 | 0/3 |
| 200 | 800 / 200 | 0.628101 | 0.635442 | -0.007340 | 0/5 | 0/2 |
| 300 | 1200 / 200 | 0.627693 | 0.635442 | -0.007748 | 0/5 | 0/2 |

新增动作后，增加source数据没有修复跨域误路由，macro反而略降。对应Flux3的100/200/300
gain为`-0.000458/+0.000317/+0.001401`，因此失败由动作扩容引入，而不是原数据量曲线本身。

### n=300逐Benchmark结果

| Held-out target | Target Static LP | Flux4 Router | Router−LP (95% CI) | Router中4B比例 |
|---|---:|---:|---:|---:|
| GM | 0.323623 | 0.304893 | **-0.018729 [-0.028604,-0.009100]** | 47.6% |
| AM | 0.752023 | 0.751150 | -0.000874 [-0.005128,+0.003229] | 34.8% |
| UniGen | 0.821896 | 0.816470 | -0.005426 [-0.013868,+0.002845] | 37.4% |
| CoreBench | 0.507022 | 0.504116 | -0.002906 [-0.006895,+0.000974] | 32.5% |
| WISE | 0.772644 | 0.761837 | **-0.010806 [-0.018204,-0.003307]** | 38.0% |

Target Static LP在五个目标、全部seed上仍选择`0% 4B + 86.82% F10 + 0% F50 + 13.18% TTS`；
公共Router却把32.5%–47.6%的prompt分配给4B。GM和WISE上的廉价动作误选造成显著质量损失。

### 动作是否有潜在价值

有。使用目标test标签的Flux4 oracle macro为`0.726467`，相对Target LP高`+0.091025`，说明4B
可以释放预算、为真正需要的prompt换取更多高成本动作。问题是当前prompt Ridge无法可靠判断哪些
prompt适合4B：即使见过目标300/50标签的Flux4 specialist，macro也只有`0.638666`，GM仍显著
低于Static LP `-0.015283`；CoreBench则显著高`+0.020268`。这是action admission/calibration问题，
不是4B动作本身没有oracle价值。

### 结论

1. **Flux4动作空间已完成五benchmark数据量实验，但naive公共Router失败。** n=300 macro相对
   target-seen Static LP为`-0.007748`，GM/WISE显著负迁移。
2. **动作空间越丰富，零样本routing条件越严格。** 极低成本但质量分布不同的4B会改变lambda策略，
   source域上学到的“哪些prompt可降级”不能直接迁移到held-out benchmark。
3. **下一步不应继续单纯扩source样本。** 应加入source-only action admission：只有在nested LOBO
   中证明4B非负的域才开放4B；其余目标回退Flux3，或者限制4B最大流量并校准其预测值。

### 产物

- `run_exp55_flux4_pooled_router_data_curve.py`
- [results/exp55_flux4_pooled_router_data_curve.json](results/exp55_flux4_pooled_router_data_curve.json)
- [results/exp55_flux4_pooled_router_data_curve.png](results/exp55_flux4_pooled_router_data_curve.png)
- 50份n=100/200预测缓存；n=300复用§39独立action heads。
<!-- report:exp55:end -->

<a id="exp56"></a>

<!-- report:exp56:start -->
## §56 Flux4 公共 Router：每个来源 1,000 条训练数据

> 2026-09-11 整理时，根据已有结果 JSON 和运行脚本补齐记录；本次没有重新训练或评测。[返回实验目录](README.md)。

### 研究问题

§55 在每源 100/200/300 条时发现 Flux4 公共 Router 的跨域负迁移。本实验把每个 source 的训练量增加到 1,000，检查是否出现可靠的实例级路由收益。

### 数据与协议

- 动作库：`flux4b_4steps / flux_10steps / flux_50steps / flux_tts`，成本 `[0.455, 4.7, 23.1, 44.9]`，平均 budget=10。
- 五个目标：GM、AM、UniGen、CoreBench、WISE；每次完整留出一个目标，只混合其他四个 source 训练。
- 每源 1,000 train + 50 alpha validation；每个 pooled Router 合计 4,000 train + 200 validation。
- construction seeds 为 42–46；逐 prompt 配对差值在 5 个 split 上取均值，再做 5,000 次 bootstrap。
- 主基线为使用目标域 1,000 条训练标签的 Target Static LP；Public LP 仅用 source 标签。目标标签不参与 pooled Router 拟合。

| 数据域 | construction pool | 固定 test |
|---|---:|---:|
| GM | 2400 | 800 |
| AM | 2400 | 800 |
| UniGen | 2400 | 600 |
| CoreBench | 1080 | 400 |
| WISE | 1200 | 200 |

CoreBench/WISE 复用 §47/49 的固定测试划分：保留一部分历史 raw 为 test，其余并入 construction，并检查 prompt 无交集。这里的扩量不是后续 construction-v2 新生成文本。GM/AM 是同一 prompt 域的两种评分聚合，不能把两者之间的迁移视为完全独立 prompt 域泛化。

### 结果

| Held-out target | Router | Target LP | Router−LP | 配对 95% CI |
|---|---:|---:|---:|---|
| GM | 0.303570 | 0.323623 | -0.020053 | [-0.031383, -0.008448] |
| AM | 0.754173 | 0.752023 | +0.002150 | [-0.002045, +0.006399] |
| UniGen | 0.816572 | 0.821896 | -0.005324 | [-0.012916, +0.002338] |
| CoreBench | 0.521177 | 0.524621 | -0.003444 | [-0.011879, +0.004849] |
| WISE | 0.777492 | 0.779122 | -0.001630 | [-0.018633, +0.016523] |
| 五域等权 macro | 0.634597 | 0.640257 | -0.005660 | — |

Public LP 与 Target LP 在本次结果中逐目标质量相同。正点估计 1/5，显著正 0/5，显著负 1/5；显著负项为 GM。

### 结论与比较限制

1. 每源增加到 1,000 条后，当前 Flux4 pooled Router 仍没有可靠的跨目标正收益：macro 低于 Target LP，GM 显著为负。
2. 这支持继续检查动作选择和跨域校准，而不能声称只增加 source 样本就已解决 Flux4 误路由。
3. **不能把 §55 和 §56 的 macro 差值解释为严格的数据量提升。** CoreBench/WISE 的 construction/test 支持改变，Target LP 训练量也从 300 改为 1,000；若要报告同口径学习曲线，需在相同划分和基线标签预算下重评估较小训练量。
4. 本页只整理现有结果。JSON 中的数值审计为：聚合分有限、最大策略成本 10、预算满足、JSON 中 `construction_test_overlap=0` 是硬编码，运行时只检查了重划后的 CoreBench/WISE；本次发现旧 UniGen pool/test 实际重合 400 条。该字段不能证明五域全部无交集。

### 产物

- [主结果 JSON](results/exp56_flux4_pooled_1k.json)
- [运行脚本](run_exp56_flux4_pooled_1k.py)
- 预测缓存：`results/exp56_flux4_pooled_1k_target-{target}_seed{seed}.npz`
- 前序：[§55 Flux4 100/200/300 数据量实验](exp55_56_flux4_pooled_router.md#exp55)
- 数据来源说明：[基础数据盘点](base_model_data_inventory_5bench.md)、[§47](exp47_49_flux_only_extension.md#exp47)、[§49](exp47_49_flux_only_extension.md#exp49)
<!-- report:exp56:end -->

</details>
