# CoreBench / WISE v2：数据构建与首轮图像评测

[返回项目入口](../项目导航.md)。2026-09-11 根据正式数据、manifest 和评分 JSON 核对，合并原 construction 过程记录与 WISE 计划。下表是本地已落盘状态；不代表查询过实时集群任务。

## 已完成的正式文本

| 数据集 | 原池 | 保留 | 新增入选 | 正式总量 | 类别 |
|---|---:|---:|---:|---:|---|
| CoreBench v2 | 400 | 383 | 617 | 1000 | 12 类，每类 83/84 |
| WISE v2 | 400 | 369 | 631 | 1000 | 6 大类、26 子类 |

- CoreBench：读取 12 个正式 JSON，合计 1000 个唯一 ID；617 条新样本来自 KEEP，作者为 Gemini 309 / Qwen 308。权威来源：[manifest](results/construction_v2/final/corebench_v2_manifest.json)、[正式数据](../codes/corebench/generate_json_v2/_附件目录.md)。
- WISE：正式 JSON 的 1000 个 ID 和 casefold prompt 均唯一；新增作者为 GPT-5.5 481 / Gemini 150。类别计数为 Cultural knowledge 400，Biology / Physical Knowledge / Chemistry / time / Space 各 120。权威来源：[manifest](results/construction_v2/final/wise_v2_manifest.json)、[正式数据](../codes/wise/wise_generate_v2.json)。
- WISE 正式 JSON SHA-256：`3d1f435bd0b3d06dc8e263c9c3cb1f484f26c1458a7ab0d3eb53596ab93cdb09`。
- WISE 第一轮 1262 是过采样候选数，不能当作正式训练量；旧日志的“520 条原池、12 类”与正式产物不符。

<a id="flux-full-inference-authorization-2026-09-06"></a>
## 正式图像与评分

| 数据集 | 模型 / 配置 | 图片与评测量 | 平均图像分 | 评分口径 | 结果 |
|---|---|---:|---:|---|---|
| CoreBench v2 | FLUX 4B / 4 steps / seed0 | 1000 | 0.4799053968 | historical_corebench_gemini3flash | [JSON](results/construction_v2/image_stage/corebench_full_s0/flux4b/eval/gemini/summary.json) |
| CoreBench v2 | FLUX base 9B / 20 steps / seed0 | 1000 | 0.5307153862 | historical_corebench_gemini3flash | [JSON](results/construction_v2/image_stage/corebench_full_s0_flux9b/flux9b/eval/gemini/summary.json) |
| WISE v2 正式集 | FLUX 4B / 4 steps / seed0 | 1000 | 0.448 | Qwen Prompt+Explanation binary | [JSON](results/construction_v2/image_stage/wise_formal_full_s0/flux4b/eval/summary.json) |

这只是已完成的模型/步数/seed 组合，不能称为完整 Flux3/Flux4 或 A7 多 seed 标签矩阵。历史 WISE Prompt-only 三次 0–10 评分，与 v2 Prompt+Explanation 单次 binary 评分不同，不能直接横向拼接。

<a id="wise-formal-freeze-and-final-flux-delta-2026-09-06"></a>
## WISE 正式冻结与 provisional 的区别

原 provisional 使用未完成全部事实核验的配额候选快照，已有分数 0.46。正式候选经过后续事实核验与替换，最终冻结为 369 保留 + 631 新样本。正式 selector 只接受内容哈希一致的 `fact_verified` 候选。

正式集与 provisional 有 510 条不同；经逐行内容相等检查复用 490 张图及匹配 judge cache，对其余 510 条重新生成和评分。最终正式分数为 0.448，不能把 provisional 分数作为正式集分数。正式产物：[bundle](results/construction_v2/image_stage/wise_formal_full_s0/bundle.json)、[delta manifest](results/construction_v2/image_stage/wise_formal_full_s0/delta_manifest.json)、[job](results/construction_v2/image_stage/wise_formal_full_s0/job.json)。

最后的替换运行在原记录中为 59928018 / `psxhg0mxrr7ko5yn`，2026-09-07 00:47 CST 完成。原失败、限流、各 wave 中间数量都保留在合并前归档，已不作为当前待办。

## Router 标签缺口（本轮核对）

已检查三套正式 bundle 的内容哈希、1000个唯一来源ID、1000张非空图片及评分数量，共3000张。详见 [动作与 seed 清单](results/prompt_disjoint_v1/v2_action_seed_inventory.json)。

如果下一步采用旧主线 Flux4 四动作 × 5个生成seeds：

| 正式集 | 目标动作/seed标签数 | 可直接计入 | 尚缺 |
|---|---:|---:|---:|
| CoreBench v2 | 20000 | 4B/4step/seed0 的1000条 | 19000 |
| WISE v2 | 20000 | 4B/4step/seed0 的1000条 | 19000 |

CoreBench现有9B/20step/seed0不属于Flux3的10/50/TTS动作，不能顶替；38000是动作/seed评分条目，TTS可能包含多次内部生成，不能当成图片或GPU时长估计。CoreBench与WISE的评分协议仍分别固定，旧WISE分数不能与新binary标签混用。

2026-09-12 用户决定暂缓标签补齐与 Hope；上述缺口保留为以后恢复工作的规模依据。Qwen图片任务继续暂停。图像preview跨域复验和旧CLIP/TFMLP比较同样暂缓，当前只完善已有结果和本地CPU流程。

## 固定构建原则

1. 正式 v2 独立输出，不在旧 generated 池上直接追加；按固定类别/子类配额选择。
2. 作者仅看设计指南与新 ID，不读取原始 benchmark、拒绝清单或碰撞审计反馈。
3. 由不同模型独立审核；作者输出冻结后再比较历史/raw/候选间碰撞。自动无匹配不能被解释为穷尽语义去重。
4. WISE 需额外保存外部来源证据、冻结内容哈希和事实结论；模型审核 PASS 不等于 fact_verified，也不等于正式入选。
5. 执行失败保留错误和已有缓存；断点续跑按冻结输入复用。不能因 quota 尚缺就降低审核门槛。

## 复现与续做入口

| 内容 | 路径 |
|---|---|
| 固定配额 | [CoreBench](results/construction_v2/corebench_quota.json)、[WISE](results/construction_v2/wise_quota.json) |
| 原池清洗 | [审计记录](results/construction_audit/README.md)、[人工复核](../codes/construction_audit_overrides.json) |
| CoreBench author/review/audit 缓存 | [full](results/construction_v2/full/_附件目录.md) |
| WISE 各波候选与审核 | [construction-v2 产物目录](results/construction_v2/_附件目录.md)，`wise_full` 与 `wise_wave2`–`wise_wave14` |
| WISE 事实来源与待定候选 | [事实目录](results/construction_v2/wise_facts/_附件目录.md)、[汇总](results/construction_v2/wise_facts/summary.json)；未入选的 pending 不妨碍已冻结正式集完成 |
| WISE 正式导出器 | [select_wise_v2_final.py](../codes/select_wise_v2_final.py) |
| 图像阶段说明与编排 | [README](../codes/construction_v2_images/README.md)、[图像产物目录](results/construction_v2/image_stage/_附件目录.md) |
| 预览与浏览 | [construction 页面](../附件范围.md#attachment-4)、[20 条预览](results/construction_v2/wise_preview20/wise_preview20.md)；预览不自动计入正式配额 |

原记录的模型决策：当前比较矩阵为 FLUX 4B、FLUX base 9B、Qwen-Image；Qwen 图片任务保持暂停，SD3.5 系列被排除。这里记录既有决定，不自动启动后续实验。若下一步构造 Router 标签矩阵，先按正式数据版本列清 action × seed × judge 的缺口，再独立安排。

<a id="wise-design"></a>

<!-- report:wise-design:start -->
## WISE 设计与审核合同

### 目标与配额

正式数据保留 369 条历史高置信样本，新增 631 条，最终 1,000 条。
类别、子类、保留数量和新增 ID 范围全部沿用
[results/construction_v2/wise_quota.json](results/construction_v2/wise_quota.json)，不按生成成功率改变目标分布。

这轮先生成 60 条全新候选，挑出 20 条供人工看题型与难度。
预览希望覆盖 Biology 3、Chemistry 3、Cultural knowledge 6、Physical Knowledge 3、Space 3、time 2；
这是覆盖性配额，并非最终 benchmark 比例。若某类合格数不足，补充候选，不拿未通过样本填数。

### 改进生成方式

之前 60 条试点仅 31 条通过，主要问题是知识目标碰撞，以及事实不准确、答案不唯一或难以从图判断。
本轮先设计稳定知识关系、必要条件、可视证据与容易混淆的替代答案，再写自然的图像生成描述。

- 风格采用独立画面的自然描述，优先名词短语；避免 Show、Change、Replace 等操作口吻和逐步编辑指令。比较题直接描述同一画面中的对象，保留必要条件，不泄露结果。
- Prompt 保持 7–21 个英文词，使用自然图像请求；不能是问答题，不能直接泄露目标答案。
- Explanation 保持 12–48 个英文词，给出目标及判分所需的可视证据。
- 生物与化学题写明使现象成立的物种、环境、材料及条件；不能只凭颜色或普通外观猜唯一物质。
- 文化题提供足够独特的线索，避免靠地区、宗教或节日的宽泛联想得到多个正确答案。
- 空间与时间题要求可见的完整状态；观察方向、参照系、日期/时区等必要条件不能缺失。
- 保留 fact_ledger：target_answer、knowledge_relation、required_visual_evidence、ambiguity_notes。

### 作者、审核与隔离

本轮预览使用 Gemini 3 Flash 与 GPT-5.5 各生成 30 条，并由另一模型审核。
这是为尽快检查 WISE 新方案而设置的 API 预览组合，不冒充既有 Gemini/Qwen 试点。
正式规模仍保留至少两个作者、不同模型交叉审核的要求；预览是否计入正式池待审阅后决定。

作者脚本只能读取设计指南与新 ID，不打开 raw、历史生成池或碰撞审计结果。
所有作者输出冻结后，独立审核先仅看 Prompt 推断目标，再读取 Explanation 检查事实、唯一性及可视性。
失败输出记录错误并重试；真实质量拒绝保留拒绝原因，不伪造合格结果。

离线审计全局比较原始 WISE、保留的 369 条、旧试点以及本轮不同作者候选，不能仅在同子类比较。
词面相似度和答案词重叠只作召回；还要人工比较知识目标、条件和关系，避免同一题换说法或换类别。
原始题目、原始目标、拒绝清单不回灌作者或独立审核模型。

候选选定前用外部权威资料核对事实，并保存可点击来源。没有足够证据的题替换或标为待核验，不计入合格 20 条。
<!-- report:wise-design:end -->

## 历史过程的保存位置

原 construction 记录和 `construction_v2_record.md#wise-design` 的全部内容及文件哈希保存在 [合并前归档](../附件范围.md#attachment-2) 与 [合并映射](../docs/record_merge_manifest.json)。包含各轮失败、提交信息、缓存路径和曾被替换的中间判断；本页保留正式状态与可复现合同。
