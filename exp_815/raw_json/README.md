# exp_815 原始数据说明（raw_json/）

本文件夹汇总 exp_815 全部实验的**原始结果 JSON**，统一信封格式、统一命名。
由 `exp_815/organize_raw_json.py` 从 `exp_815/results/` 生成，原始数据逐字段保留，
只把 "list-of-3（带 dataset 键）" 重排为 `data: {"gm": ..., "am": ..., "unigen": ...}`。

## 任务说明（主线实验设置）

主题：CATImage 式 **compute routing**——给定 prompt，predictor 预测 7 个生成配置（cfg）
的质量分，路由器在 cost ≤ budget 约束下选最优 cfg（λ 扫描口径，与原论文一致）。

- **标签**：score 模式（非 delta），每条 prompt 的 7 维真实质量分
- **7 个 cfg 与 cost**（qwen50 = 10.0 为 anchor/baseline 基准）：

| cfg | 来源 | cost（归一化 TFLOPs） |
|---|---|---|
| qwen_10steps | qwen | 2.0 |
| qwen_50steps | qwen | 10.0（anchor） |
| flux_10steps | flux-9b | 4.7 |
| flux_50steps | flux-9b | 23.1 |
| boogu_10steps | boogu | 4.3 |
| boogu_50steps | boogu | 21.5 |
| flux4b_4steps | flux4b | 0.455（= 实测 127.26 TFLOPs / 279.7） |

- **数据集**：gm（GenEval2-edit）、am（GenEval2-edit_am）、unigen（UniGenBench）。
  pool/test 均为 qwen/flux/boogu/flux4b 四来源 instruction 交集
- **划分**：train 2200 / val 200 / test = 800（gm、am）或 600（unigen）
- **随机种子**：seeds = [0, 1, 2]，表格报均值±std
- **predictor**：ridge / knn / tfmlp（输入为 prompt 的 Qwen3-Embedding-8B 特征）
- **对照**：always_qwen50（budget=10 的 anchor 基线）、LP、OC、Oracle（真值上界）

## 文件清单

| 文件 | 实验 | 内容 | 生成脚本 |
|---|---|---|---|
| `exp1_direct_predict_r2.json` | 实验一 | 直接预测的 R²：每个 cfg 的全体/中段 R² 与 Spearman（证明 R² 虚高、中段塌陷） | `run_exp1_predict.py` |
| `exp1_direct_predict_preds.json` | 实验一 | test 集每条的 instruction、7 维真实分 `true`、三个 predictor 的预测 `ridge/knn/tf` | `run_exp1_predict.py` |
| `exp2_routing_budgets.json` | 实验二 | budgets=5/10/20 的路由质量：`budgets["10.0"]` 下含 oracle/always_qwen50/lp/oc/ridge/knn/tf 及各自 `*_seeds` | `run_exp2_routing.py` |
| `exp3_trim_extremes.json` | 实验三 | 掐头去尾：`trims` 下 0.2/0.4/0.6/0.8/1.0 五档裁剪后的路由质量（budget=10） | `run_exp3_trim.py` |
| `exp4_train_seeds.json` | 实验四 | seed 消融：`seed_counts` 下 k=1..5 的路由质量（budget=10） | `run_exp4_seeds.py` |
| `exp5_counterfactual_gap.json` | 实验五 | 反事实分解：base / CF-extreme / CF-mid / Oracle 质量与极端样本统计 | `run_exp5_counterfactual.py` |
| `exp6_twostage_predictor.json` | 实验六 | 两阶段 predictor：检测器准确率/混淆矩阵 + base vs twostage 质量 | `run_exp6_twostage.py` |
| `appendix_unigen_perseed_labels.json` | 附录 | unigen 每 instruction×来源×steps 的 5 个生成 seed 原始分（实验四底层数据） | `run_exp4_seeds.py` |

信封字段：`experiment / title / description / generated_by / source_file /
common_setup / exp_config / data`。`source_file` 指向 `results/` 里的原始文件。

## 底层标签数据

per-prompt 真实分（含各 seed）已统一归档到 **`raw_labels/`**（24 个文件，
见该文件夹 README）；加载与划分逻辑见 `exp_815/common.py`。

## 备注

- 历史备份（名单修正前、单 budget 版、已删除的旧实验五等）在 `results/backup_*/`，不在本文件夹内。
- 复现核对关系：exp5/6 的 base 配置与 exp2 budget=10 逐位一致（脚本内置 assert）；exp4 k=5 与 exp2 budget=10 一致（unigen 因交集差异 ≤0.003）。
