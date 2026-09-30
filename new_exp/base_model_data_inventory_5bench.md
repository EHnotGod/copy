# 五个主 Benchmark 的基础模型数据盘点

> 路由重跑的可用支持：UniGen物理源文件仍为gen2400；排除与raw600完全相同的400条prompt后，新实验使用pool2000/test600。原图片与标签不删除，精确保留行及排除列表见 [冻结划分](results/prompt_disjoint_v1/unigen_clean_split.json)。


物理图片盘点日期：2026-09-03；标签可用性与数据划分于 2026-09-11 复核。图片历史总数未在本轮重算，后续 v2 图片不计入该存量表。

## 统计口径

- `construction/test` 表示可用于训练或模型选择的 construction pool，以及原始 benchmark 测试集。
- 数据量按唯一的 `prompt × action × generation seed` 去重。相同 prompt/action 的历史重复生成不重复计入。
- 默认每个完整 action 有 5 个生成 seed；“图片数”是所有所列 action、两个 split 的合计。
- “已评分 prompt”按 `prompt × action` 的聚合标签计数；per-seed 可恢复性另行说明。
- GenEval2-GM 和 GenEval2-AM 使用同一批生成图片，只是评分指标不同。因此两行可以分别统计标签，但物理图片总量只能计算一次。
- `TTS/edit` 是 text-to-text 改写后再编辑的最终图片；`more` 是另一套 staged action，单独统计。

## 一眼看全

| 基础模型 | 覆盖的 benchmark | 去重后的有效图片数 | 主要缺口 |
|---|---|---:|---|
| Qwen-Image-2512 | GM、AM、UniGen、CoreBench、WISE | 182,600 | 新增 1,600 条已有可恢复 AM（§46）；UniGen `more` 仅 construction 400 |
| FLUX.2-Klein-9B | GM、AM、UniGen、CoreBench、WISE | 235,100 | 新增 1,600 条已有可恢复 AM（§46）；UniGen `more` 仅 construction 400 |
| Boogu-Image | GM、AM、UniGen | 93,000 | CoreBench、WISE 没有数据 |
| FLUX.2-Klein-4B | GM、AM、UniGen、CoreBench、WISE | 73,800 | 新增 1,600 条已有可恢复 AM（§46）；UniGen 仅 2,400 construction；WISE raw 的逐 seed 标签只保留 123/1,000 |
| SD3.5 | GM、CoreBench | 24,354 | 无 AM/UniGen/WISE；CoreBench construction 仅部分出图且未评分 |
| SD3.5M | GM、CoreBench | 19,200 | 无 AM/UniGen/WISE；CoreBench 无 construction，raw 只覆盖 8/12 类 |
| **合计** | 五个主 benchmark | **628,054** | GM/AM 共享图片，只计一次 |

上表的 628,054 是实验可用数据量，不是磁盘上 PNG 的机械总数。UniGen 早期 400 条和后来的 2,400 条存在重复 prompt，Flux4B 的早期 400 条 4-step 也在 2,400 条全集中；这些重复生成在此处只算一次。

## GenEval2-GM

| 模型 | actions | 出图 prompt（construction/test） | 已评分 prompt（construction/test） | 有效图片数 | 状态 |
|---|---|---:|---:|---:|---|
| Qwen | 10-step、50-step、TTS/edit | 4,000 / 800 | 4,000 / 800 | 72,000 | 5 seeds 完整 |
| Qwen | more | 400 / 800 | 400 / 800 | 6,000 | 5 seeds 完整 |
| Flux-9B | 10-step、50-step、TTS/edit | 4,000 / 800 | 4,000 / 800 | 72,000 | 5 seeds 完整 |
| Flux-9B | more | 400 / 800 | 400 / 800 | 6,000 | 5 seeds 完整 |
| Boogu | 10-step、50-step、TTS/edit | 2,400 / 800 | 2,400 / 800 | 48,000 | 5 seeds 完整 |
| Flux-4B | 4-step | 4,000 / 800 | 4,000 / 800 | 24,000 | 5 seeds 完整 |
| Flux-4B | 20-step | 400 / 800 | 400 / 800 | 6,000 | 5 seeds 完整 |
| SD3.5 | 10-step、28-step | 400 / 800 | 400 / 800 | 12,000 | raw 10-step 少 1 个 seed 分：3,999/4,000 |
| SD3.5M | 10-step、40-step | 400 / 800 | 400 / 800 | 12,000 | 5 seeds 完整 |

GM 合计有 51,600 个 `prompt × action` 聚合标签；对应逐 seed 评分应为 258,000 个，其中实际可恢复 257,999 个，唯一缺口是 SD3.5 raw 10-step 的一个 seed。

## GenEval2-AM

AM 复用上节全部 258,000 张 GenEval2 图片，因此下表“出图”不再纳入总图片数；重点是哪些图片已经有 AM 标签。

| 模型 | actions | 已有图片 prompt（construction/test） | AM 已评分 prompt（construction/test） | 尚待 AM 评分 |
|---|---|---:|---:|---|
| Qwen | 10-step、50-step、TTS/edit | 4,000 / 800 | 4,000 / 800（含重聚合） | 无需重评；后扩使用已有 score_list |
| Qwen | more | 400 / 800 | 0 / 0 | 全部 |
| Flux-9B | 10-step、50-step、TTS/edit | 4,000 / 800 | 4,000 / 800（含重聚合） | 无需重评；后扩使用已有 score_list |
| Flux-9B | more | 400 / 800 | 0 / 0 | 全部 |
| Boogu | 10-step、50-step、TTS/edit | 2,400 / 800 | 2,400 / 800 | 无 |
| Flux-4B | 4-step | 4,000 / 800 | 4,000 / 800（含重聚合） | 无需重评；后扩使用已有 score_list |
| Flux-4B | 20-step | 400 / 800 | 0 / 0 | 全部 |
| SD3.5 | 10-step、28-step | 400 / 800 | 0 / 0 | 全部 |
| SD3.5M | 10-step、40-step | 400 / 800 | 0 / 0 | 全部 |

按本表动作范围，AM 原显式文件有 32,000 个 prompt×action 标签；加上后扩 1,600×7=11,200 个可恢复标签，共可取得 43,200 个 prompt×action 标签及 216,000 个逐 seed AM 分数。Flux4 公共 construction 可用上限为 4,000（含重聚合），不代表已另行物化所有 AM JSON。

## UniGen

| 模型 | actions | 出图且评分 prompt（construction/test） | 有效图片数 | 状态 |
|---|---|---:|---:|---|
| Qwen | 10-step、50-step、TTS/edit | 2,400 / 600 | 45,000 | 完整；最终 v2 JSON 存均值，原始 agg 保留 5-seed 图级分 |
| Qwen | more | 400 / 0 | 2,000 | 仅 construction |
| Flux-9B | 10-step、50-step、TTS/edit | **5,900 / 600** | 97,500 | 2,400 与新增 3,500 的 instruction 交集为 0 |
| Flux-9B | more | 400 / 0 | 2,000 | 仅 construction |
| Boogu | 10-step、50-step、TTS/edit | 2,400 / 600 | 45,000 | 完整；原始 agg 有 5-seed 图级分 |
| Flux-4B | 4-step | 2,400 / 600 | 15,000 | 完整；原始 agg 有 5-seed 图级分 |
| SD3.5 / SD3.5M | — | 0 / 0 | 0 | 未跑 |

UniGen 合计有 41,300 个 `prompt × action` 聚合标签、206,500 个图级评分。

需要特别注意：Flux 目录名为 `gen4000`，但其中最终完整且已评分的是 3,500 条；与原 2,400 条完全不重合，所以 Flux-9B 的 construction 总量是 **5,900**。目录中另有 3,507 个 direct prompt 目录和 3,506 个 edit prompt 目录，未进入最终评分 JSON 的零散目录不计为有效数据。

## CoreBench

| 模型 | actions | 出图 prompt（construction/test） | 已评分 prompt（construction/test） | 有效图片数 | 状态 |
|---|---|---:|---:|---:|---|
| Qwen | 10-step、50-step、more、edit | 400 / 1,080 | 400 / 1,080 | 29,600 | 5 seeds 完整 |
| Flux-9B | 10-step、50-step、more、edit | 400 / 1,080 | 400 / 1,080 | 29,600 | 5 seeds 完整 |
| Flux-4B | 4-step、20-step | 400 / 1,080 | 400 / 1,080 | 14,800 | 图级评分可恢复 |
| SD3.5 | 10-step、28-step | 部分 160 / 1,080 | 0 / 1,080 | 12,354 | construction 只有 1,554 张零散图且未评分；raw 完整 |
| SD3.5M | 10-step、40-step | 0 / 720 | 0 / 720 | 7,200 | raw 仅覆盖 C-MA/C-MI/C-MR/C-TR/R-BR/R-HR/R-LR/R-PR 八类 |
| Boogu | — | 0 / 0 | 0 / 0 | 0 | 未跑 |

CoreBench 合计有 18,400 个 `prompt × action` 聚合标签、92,000 个图级评分。另有 SD3.5 construction 的 1,554 张未评分图片，不应当直接当作 Router 标签使用。

## WISE

| 模型 | actions | 出图且评分 prompt（construction/test） | 有效图片数 | per-seed 标签状态 |
|---|---|---:|---:|---|
| Qwen | 10-step、50-step、more、edit | 400 / 1,000 | 28,000 | 5 seeds 完整 |
| Flux-9B | 10-step、50-step、more、edit | 400 / 1,000 | 28,000 | 5 seeds 完整 |
| Flux-4B | 4-step、20-step | 400 / 1,000 | 14,000 | 聚合分完整；construction 400 条有逐 seed，raw 仅 123/1,000 条保留逐 seed |
| Boogu / SD3.5 / SD3.5M | — | 0 / 0 | 0 | 未跑 |

WISE 合计有 14,000 个 `prompt × action` 聚合标签。对应 70,000 张图片全部有聚合后的 action 分；当前可直接恢复的逐 seed 标签为 61,230 个，其中 Flux-4B raw 缺少 877 条 prompt 的逐 seed 明细，但不缺 prompt-level 平均分。

## 对当前 Flux4 动作库最有用的交集

固定动作库 `{Flux4B-4, Flux9B-10, Flux9B-50, Flux9B-TTS/edit}` 时，四个动作同时有分的最大 prompt 数为：

| benchmark | construction | 原始 test/raw | 当前瓶颈 |
|---|---:|---:|---|
| GenEval2-GM | 4,000 | 800 | 已齐 |
| GenEval2-AM | 4,000（含重聚合） | 800 | §46 已核对后扩 score_list；物化文件与可恢复标签分开统计 |
| UniGen | 2,400 | 600 | Flux-4B 只有 2,400；Flux-9B 单独已有 5,900 |
| CoreBench | 400 | 1,080 | 原 generated construction 只有 400；可按实验设计从 raw 划分补充 |
| WISE | 400 | 1,000 | 原 generated construction 只有 400；可按实验设计从 raw 划分补充 |

因此，之前“每个 benchmark 都有 1k Flux4 construction”不是原始生成池的事实描述：CoreBench/WISE 的 1k 是通过重新划分现有 raw 标签得到的；GM/AM/UniGen 则有独立 construction pool。

## 核对来源

- GenEval2 Qwen/Flux/Flux4B/SD：`data/final_data/geneval2/` 与外部 `Results/geneval2_*` 图像目录。
- UniGen：外部 `Results/unigen_qwen`、`unigen_flux`、`unigen_flux4b`、`unigenbench_boogu` 的 `eval/*.json` 和 `unigen_agg_raw.json`。
- CoreBench：`data/final_data/corebench/`；SD 系列按 12 个子类原始 Gemini 评分文件逐项计数。
- WISE：`data/final_data/wise/`。
- 所有主 JSON 均检查 instruction 唯一性和 action key 覆盖；GM/AM 的后扩 1,600 与原 2,400 交集为 0；UniGen Flux 的新增 3,500 与原 2,400 交集为 0。


## 数据划分复核

2026-09-11 逐条核对：旧 UniGen gen2400 与 raw600 的 instruction 交集为 400。因此“可用图片数”不等于独立训练/测试数据量。§48 的新 3500 pool 与 raw600 交集为 0，独立学习曲线见 [§47–49 合并记录](exp47_49_flux_only_extension.md#exp48)。GM/AM 2400/800 的精确 prompt 交集为 0；后扩 A7 三份1600标签共有 56,000 份非空有限 score_list，基础2400条 AM 重聚合误差为 2.22e-16。
