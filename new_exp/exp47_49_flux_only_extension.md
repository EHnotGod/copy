# §47–49：Flux-only Router 数据扩展

[返回实验目录](README.md)。三份记录合并，按测试集版本保留各自数值。

| 实验 | 数据域 | construction 上限 | 固定 test | 关键区别 |
|---|---|---:|---|---|
| §47 | WISE / CoreBench | 1000 / 1080 | raw400 / raw400 | 将 raw 剩余部分纳入 construction |
| §48 | UniGen | 3500 | raw600 | 采用与 test 不重合的新 pool；本次再次核对交集为 0 |
| §49 | WISE | 1200 | raw200 | 重新固定测试支持并重算整条曲线，不与旧 raw400 直接拼接 |

这三项均使用 Flux-9B 的 10/50/TTS 三动作与 budget=10，展示绝对平均质量，不是 AUDC 表。WISE/CoreBench 的 raw 重划同时改变 construction 分布，不能将所有收益都归因于样本数量。

<a id="exp47"></a>

<!-- report:exp47:start -->
## §47 WISE/CoreBench Flux-only Router 数据扩展

### 口径

- 动作空间仅为 Flux-9B `{10 steps, 50 steps, TTS/edit}`，预算10。
- 只展示 Router 绝对平均质量，不减任何 baseline。
- 原 generated pool 400条；从原 raw 集合预先固定留出400条test，剩余已评分prompt加入construction。
- WISE construction上限1000；CoreBench上限1080；test与construction严格不重叠。

![Flux-only small benchmark curves](results/exp47_flux_only_smallbench_extension.png)

### wise

| k | Router平均质量 | 95% CI |
|---:|---:|---:|
| 200 | 0.783433 | [0.761601, 0.804649] |
| 300 | 0.784321 | [0.761495, 0.805627] |
| 400 | 0.781041 | [0.757349, 0.803067] |
| 600 | 0.794446 | [0.772833, 0.814994] |
| 800 | 0.796862 | [0.776138, 0.816416] |
| 1000 | 0.801767 | [0.779152, 0.822802] |

400→1000变化 `+0.0207`，paired 95% CI `[+0.0061,+0.0364]`。

### corebench

| k | Router平均质量 | 95% CI |
|---:|---:|---:|
| 200 | 0.540069 | [0.517252, 0.562704] |
| 300 | 0.542290 | [0.519477, 0.564758] |
| 400 | 0.542482 | [0.519733, 0.565914] |
| 600 | 0.541923 | [0.519145, 0.563746] |
| 800 | 0.542376 | [0.519297, 0.564346] |
| 1000 | 0.538723 | [0.516093, 0.561909] |
| 1080 | 0.540371 | [0.516390, 0.564750] |

400→1080变化 `-0.0021`，paired 95% CI `[-0.0091,+0.0045]`。

### 产物

- 结果：[results/exp47_flux_only_smallbench_extension.json](results/exp47_flux_only_smallbench_extension.json)
- 图片：[results/exp47_flux_only_smallbench_extension.png](results/exp47_flux_only_smallbench_extension.png)
- 脚本：`run_exp47_flux_only_smallbench_extension.py`
<!-- report:exp47:end -->

<a id="exp48"></a>

<!-- report:exp48:start -->
## §48：UniGen 独立 3500 条 construction

本实验只采用已完整评测的 3500 条新增 prompt。旧 gen2400 与 raw600 有 400 条 instruction 重合，不能拼入本实验的独立 construction。历史补跑没有补出剩余 500 条，最终使用已有 3500 条的决定与故障记录已保存在合并前归档。



不再补齐缺失500条。为保证严格无泄漏，construction只使用现有3500条完整评测；它们与固定
raw600 test 的 instruction 交集为0。旧gen2400与raw600存在400条instruction交集，因此没有
把旧2400拼入本实验construction。

动作空间为Flux-9B `{10 steps, 50 steps, TTS/edit}`，预算10；5个nested construction seeds，
只报告固定raw600上的Router绝对平均质量。

![UniGen Flux-only curve](results/exp48_flux_only_unigen_extension.png)

| construction k | Router平均质量 | 95% CI |
|---:|---:|---:|
| 200 | 0.822037 | [0.803817, 0.840311] |
| 300 | 0.825499 | [0.808042, 0.842893] |
| 400 | 0.823971 | [0.805159, 0.841311] |
| 600 | 0.824652 | [0.806820, 0.843132] |
| 800 | 0.825085 | [0.806078, 0.842780] |
| 1000 | 0.824856 | [0.806519, 0.842460] |
| 1200 | 0.824112 | [0.805625, 0.842172] |
| 1800 | 0.826940 | [0.808704, 0.845129] |
| 2400 | 0.827134 | [0.809454, 0.845614] |
| 3000 | 0.828730 | [0.810768, 0.846897] |
| 3500 | 0.830260 | [0.811654, 0.847462] |

配对变化：

- 200→3500：`+0.008223`，95% CI `[+0.002561,+0.014645]`，总体提升成立；
- 2400→3500：`+0.003126`，95% CI `[-0.000352,+0.006913]`；
- 3000→3500：`+0.001530`，95% CI `[-0.000831,+0.004029]`。

结论：UniGen从200扩到3500有小幅但明确的总体提升；2400以后仍呈正向点估计，但增幅已经很小，
置信区间跨0，应判断为接近平台，而不是“数据越多仍稳定显著提升”。

产物：

- [results/exp48_flux_only_unigen_extension.json](results/exp48_flux_only_unigen_extension.json)
- [results/exp48_flux_only_unigen_extension.png](results/exp48_flux_only_unigen_extension.png)
- [results/exp48_flux_only_benchmark_extension.png](results/exp48_flux_only_benchmark_extension.png)（WISE/CoreBench/UniGen统一图）
- `run_exp48_flux_only_unigen_extension.py`
- Qwen3 embedding cache：`codes/.cache_embed/emb_c23138ded8f5491bb68a9daaba3f193b.npy`
<!-- report:exp48:end -->

<a id="exp49"></a>

<!-- report:exp49:start -->
## §49 WISE Flux-only Router 扩展到1200

### 口径

- 只使用Flux-9B `{10 steps, 50 steps, TTS/edit}`，预算10。
- 只报告Router绝对平均质量。
- 原始数据为400条generated与1000条raw；固定raw200作为最终test，将其余raw800加入construction，
  因而construction上限从1000扩到1200。
- test与construction的instruction交集为0；5个nested construction seeds。
- 这是新的raw200 test口径，整条曲线均已重跑，不能把数值与旧raw400曲线逐点直接比较。

![WISE Flux-only extension](results/exp49_flux_only_wise_extension.png)

| construction k | Router平均质量 | 95% CI |
|---:|---:|---:|
| 200 | 0.776378 | [0.741489, 0.809179] |
| 300 | 0.772305 | [0.735209, 0.806579] |
| 400 | 0.765846 | [0.727590, 0.801013] |
| 600 | 0.789324 | [0.757215, 0.819621] |
| 800 | 0.797162 | [0.764853, 0.827114] |
| 1000 | 0.798069 | [0.766438, 0.828675] |
| 1200 | 0.800203 | [0.768002, 0.831095] |

配对变化：

- 200→1200：`+0.023825`，95% CI `[+0.004879,+0.044893]`；
- 400→1200：`+0.034358`，95% CI `[+0.012845,+0.059136]`；
- 1000→1200：`+0.002134`，95% CI `[-0.001999,+0.007085]`。

结论：从小数据扩到1200的总体提升成立；新增的最后200条只有很小的正向点估计且CI跨0，
WISE在k约800—1000后已经接近平台。

### 三项benchmark更新图

WISE使用本实验raw200口径；CoreBench与UniGen沿用§47/§48的严格不重叠口径。

![Flux-only benchmark extension](results/exp49_flux_only_benchmark_extension.png)

### 产物

- [results/exp49_flux_only_wise_extension.json](results/exp49_flux_only_wise_extension.json)
- [results/exp49_flux_only_wise_extension.png](results/exp49_flux_only_wise_extension.png)
- [results/exp49_flux_only_benchmark_extension.png](results/exp49_flux_only_benchmark_extension.png)
- `run_exp49_flux_only_wise_extension.py`
<!-- report:exp49:end -->
