# exp_815 — 7-cfg 跨模型路由汇总实验

> **历史协议说明（2026-09-12）**：本页 UniGen 使用旧 gen2400/raw600，已有400条完全相同 prompt 的重合证据，不能据此主张独立 prompt 泛化。本轮未对本目录全部 predictor 重跑；现行修复范围见 [修复清单](../修改日志.md)。GM/AM 与 UniGen 分别判断，本文七动作不含 TTS，也不是后期公共 A7。

主线设置：**train:2200 / val:200**，统一 **score 模式**（直接预测绝对分，不用 delta）。
Budget = 10.0（= qwen50 实测 2797 TFLOPs 归一化，1 unit ≈ 279.7 TFLOPs）。

## 路由池（7 个 cfg）

| idx | cfg | cost | 数据来源（GM/AM） | 数据来源（UniGen） |
|-----|-----|------|------|------|
| 0 | qwen_10steps  | 2.0  | `data/final_data/geneval2/edit*/geneval2_qwen_edit_{gen,gen2000}_perseed.json` | `Results/unigen_qwen/eval/gen2400_edit_v2.json` |
| 1 | qwen_50steps  | 10.0（anchor / baseline） | 同上 | 同上 |
| 2 | flux_10steps  | 4.7  | `.../geneval2_flux_edit_{gen,gen2000}_perseed.json` | `Results/unigen_flux/eval/gen2400_edit_v2.json` |
| 3 | flux_50steps  | 23.1 | 同上 | 同上 |
| 4 | boogu_10steps | 4.3  | `Results/geneval2_boogu/eval/gen2400[_am_v2]_perseed.json` | `Results/unigenbench_boogu/eval/gen2400_tts_edit_v2.json`（文件名含 tts，但取 "10steps"/"50steps" key） |
| 5 | boogu_50steps | 21.5 | 同上 | 同上 |
| 6 | flux4b_4steps | 0.455（实测 127.26 TFLOPs / 279.7，`logs/bench_flops_new.log`） | `data/final_data/geneval2/direct/geneval2_flux4b_direct_gen2400_{gm,am}_perseed.json` | `Results/unigen_flux4b/eval/gen2400_direct_v2.json`（已补齐至 2400） |

- qwen/flux/boogu 各取 10steps 和 50steps 两档 + flux4b 4steps，**不含任何 tts 档**。
- sd35 / sd35m 的 gen 池只有 400 条，不满足 train:2200，**不纳入**。
- test 集：GM/AM = raw800（flux4b: `..._raw_{gm,am}_perseed.json`），UniGen = raw600（均已补齐）。

## 数据划分

- pool = 各 cfg 来源的 instruction 交集（三个数据集均为 2400）。
- 固定 seed=42 shuffle，前 200 条作 external val，其余 2200 作 train pool（主线取全部 2200）。
- 随机性：predictor 训练 / LP / OC 采样用 seeds [0,1,2]，报均值。

## 实验一：直接预测（证明 R² 虚高）

对每个 dataset × cfg，用 train:2200 训练、val:200 选参/早停，在 test 上逐条预测 score。

- Predictor：Ridge（Qwen3-Embedding-8B 4096d + StandardScaler，α 用 val R² 选）、KNN（CLIP-ViT-L/14 mean-pool 768d，k=min(50, n//3)）、TFMLP（CATImage Appendix D：CLIP per-token 77×768 → 2 层 self-attn + per-token MLP，`codes/catimage_tf_mlp.py`）。
- 产出：
  - 散点图：pred vs true，每 dataset 一张（3 predictor × 7 cfg 网格，格内标 R²(all)/R²(mid)）。
  - 分布柱状图：pool（2400 条）true score 的直方图（每 cfg 一格），标 frac<0.05 / frac>0.95，证明 0–1 分布不均匀。
  - R² 表格：R²(all)，dataset × cfg × predictor。
  - R²(mid) 表格：仅 y∈(0.10, 0.90) 的子集 R²，预期普遍塌陷 → "R² 虚高"。

## 实验二：带路由（score 模式）

Lagrangian 路由：`argmax(pred − λ·cost)`，扫 λ 得到 quality-cost 曲线，在 budget ∈ {5.0, 10.0, 20.0} 处线性插值（predictor 预测只算一次，三档复用）。

- Predictor：**Ridge**（Qwen3-Embedding-8B + StandardScaler，α 按 val R² 逐 cfg 选）、**KNN、TFMLP**（CATImage predictor）。
- Baselines：**always_qwen50**（= test 上 qwen50 列均值，budget 无关锚点；budget=5 时已超支仅作参考）、**LP**（train 均值凸包在 budget 处两档随机混合）、**OC**（train 集 oracle 选择分布 p_oc，超预算则向最便宜档收缩后随机分配）。另报 **Oracle**（true 路由上界）作参考。
- 产出：得分表格，每 budget 一张 dataset × {Oracle, LP, OC, always_qwen50, Ridge, KNN, TFMLP}。

## 待办

- [x] flux4b UniGen 补齐：gen2400 / raw600 已补齐（旧 2100/450 结果备份在 [results/backup_exp1_unigen2100.json](results/backup_exp1_unigen2100.json)）。UniGen 实验一已用 2400/600 重跑。
- [x] flux4b_4steps cost 已实测填入 `common.py`：127.26 TFLOPs / 279.7 = 0.455（`logs/bench_flops_new.log`）。实验二三数据集已用该值正式跑完。
- [x] 8/18 修正 7-cfg 名单：去掉 qwen_tts / boogu_tts，改为 qwen/flux/boogu 各 10steps+50steps + flux4b_4steps；实验一/二已按新名单重跑，旧名单结果备份在 `results/backup_roster_with_tts/` 和 `figs/backup_roster_with_tts/`。
