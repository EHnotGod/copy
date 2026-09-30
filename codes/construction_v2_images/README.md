# CoreBench / WISE 下一阶段：图片推理与判分

当前状态：CoreBench-only 30 条小批任务已成功完成，并以历史 Gemini 协议得到 48.72%。FLUX.2-klein-4B 的正式 1,000 图和历史 Gemini 评分均已完成，均分 47.99%；FLUX.2-klein-base-9B 也已完成 1,000 图和评分，均分 53.07%。WISE 的 1,000 条 provisional 推理快照已冻结，FLUX 4B / seed0 / 4 steps 任务为 run 59890348 / app psxh4yaxrr7ko5yn。SD3.5 已按用户决定排除，Qwen-Image 仍暂停。

## 小批验证

输入已冻结到 `new_exp/results/construction_v2/image_stage/smoke/bundle.json`：CoreBench 从 54 条接受试点中按子类轮流抽 30 条；WISE 使用两轮预览共 30 条。固定 seed=20260905，保留 source_id、原始字段及来源文件哈希。试点字符串 Checklist 仅包装成 question 字段，文本不改写。

默认每题 seed=0、4 steps、1024×1024，共 60 张；该批用于检查图片可视性和判分可用性，不代表两个正式 1,000 条数据集完成。

在项目根目录执行只检查配置（不加载模型）：

```bash
IMAGE_PLAN_ONLY=1 bash codes/construction_v2_images/run_hl.sh
```

HL GPU 入口：`codes/construction_v2_jobs/submit_images_smoke_hl/images_smoke_hl.hope`。使用 1 worker × 8 GPU，独立 subprocess，CUDA_VISIBLE_DEVICES 在 torch 导入前设置，先预热 safetensors、错开加载，最多 3 轮续跑。提交目录只含 Hope 文件；本轮未提交。

## 全量运行

`submit_images_full_hl/images_full_hl.hope` 与 smoke 分开。默认每个 bench 1,000 条，5 seeds × 2 steps（4/20），共 20,000 张。全量模式要求显式最终数据输入并验证各 bench 总数和每子类配额；WISE 最终输入尚未齐备，因此联合全量不会误跑候选池。CoreBench 先运行独立的 seed0 / 4 steps 首轮，提交文件为 `submit_corebench_images_full_s0_hl/corebench_images_full_s0_hl.hope`，输出与未来多种子运行隔离。

默认最终路径：

- CoreBench：`codes/corebench/generate_json_v2/`
- WISE：`codes/wise/wise_generate_v2.json`

最终筛选器应只将审核、去重、事实核验完成的行写到最终路径。prepare.py 做结构/数量/配额检查，不替代语义审核。

可配置环境：`IMAGE_STAGE_MODE`、`CORE_FINAL_SOURCE`、`WISE_FINAL_SOURCE`、`IMAGE_STAGE_OUTPUT`、`IMAGE_GEN_PYTHON`、`IMAGE_JUDGE_PYTHON`、`IMAGE_MODEL_PATH`、`IMAGE_JUDGE_MODEL_PATH`、`IMAGE_PIPELINE`（flux/auto）、`IMAGE_SEEDS`、`IMAGE_STEPS`、`IMAGE_GUIDANCE`、`IMAGE_TRUE_CFG`。切换模型或采样参数使用新的输出目录。

## 输出与恢复

- `config.json` 绑定 bundle 哈希、模型路径、采样参数；变更配置不能复用旧目录。
- `images/<benchmark>/<case_index>/seed_<seed>/steps<N>.png`：两类统一路径，bundle 中 case_index 对应原始 source_id。临时 PNG 完整保存后再替换，复跑验证图片可解码。
- `eval/cache/<benchmark>/<case_index>/<seed>_<steps>.json`：保存逐项判分与原文，绑定图片 SHA256。0 分是有效结果，不因分数为零而重跑。
- `logs/`：各轮各 GPU 日志与错误项。任何缺失或损坏图片都会阻止判分；任何缺失分数都会阻止生成汇总。
- `eval/summary.json`：每个 benchmark / seed / steps 单独汇总，禁止混成一个跨 benchmark 分数。

恢复命令与初次命令相同。可以运行 `run.py --stage check --output <run目录>`；存在缺失时非零退出。

## 判分协议（按用户反馈修正）

CoreBench 必须使用历史 Gemini 协议：`judge_core_gemini.py` 为已有图片建立相对软链接和原始 ID 数据视图，调用未改动的 `eval_corebench_direct_gemini.py`。沿用 gemini-3-flash-preview、原模板、JPEG q80/max_side1024、temperature0、max_tokens1024、原 yes/no 解析和 image_score 聚合。逐项完整性检查通过才写 Gemini 汇总；缓存绑定图像与输入哈希。

结果单独写到 `eval/gemini/`；`eval/gemini_cache/` 给可视化使用。首次 smoke 的 `eval/cache/` 和 `eval/summary.json` 是本地 Qwen 诊断结果，不能作为历史可比指标。`run_hl.sh` 现已自动接 Gemini CoreBench，WISE 仍使用既有 Prompt+Explanation 本地二元协议，各自分开。

仅重评现有 CoreBench 图片，无需 GPU 重新出图：

```bash
python3 codes/construction_v2_images/judge_core_gemini.py --output <run目录>
python3 codes/construction_v2_images/visualize.py --output <run目录>
```

默认可视化读 Gemini；`--judge qwen` 另存诊断页 `visualization_qwen.html`。比较历史结果时仍须区分样本数量、类别分布和种子配置。

## 已完成验证

- 确定性抽样、ID 唯一、60 条 smoke bundle。
- 全量不足 1,000 条时拒绝；缺失图片计数正确。
- 配置改变拒绝复用；有效 0 分可恢复，图片变化使评分缓存失效，损坏 PNG 被识别。
- 4 项 unittest 通过，shell 语法和 dry-run 通过。
- FLUX 环境 diffusers 0.38.0 / Flux2KleinPipeline 导入通过；judge 环境 AutoModelForImageTextToText / AutoProcessor 导入通过。
- CoreBench-only 30 条 GPU smoke 已完成模型加载、生成和 Gemini 判分端到端验证；正式 1,000 图生成也已完成，Gemini 评分仍在运行。

## 已提交 CoreBench 独立小批

Hope：`submit_corebench_images_smoke_hl/corebench_images_smoke_hl.hope`，run 59776889，app psxhzl3xrr4mwj8c。`IMAGE_BENCHMARK=corebench`，独立输出 `image_stage/corebench_smoke/flux4b`。`prepare.py --benchmark corebench|wise|both` 支持分别准备输入。新增 CoreBench-only 抽样验证后共 5 项测试通过。
