# exp_815 底层原始标签（raw_labels/）

exp_815 全部实验使用的 **per-prompt 真实质量分**原始数据，24 个文件 =
3 数据集 × 2 划分 × 4 模型来源。由 `exp_815/organize_raw_labels.py` 生成，
已与 `common.py` 的加载口径逐值对账（每文件抽查 50 条 × 全部 cfg，0 不符）。

## 命名与格式

文件命名：`{dataset}_{split}_{model}.json`
- dataset ∈ `gm`（GenEval2-edit）/ `am`（GenEval2-edit_am）/ `unigen`（UniGenBench）
- split ∈ `train`（pool 2400 条）/ `test`（gm、am 800 条；unigen 600 条）
- model ∈ `qwen` / `flux` / `boogu` / `flux4b`

统一信封：

```json
{
  "dataset": "gm", "split": "train", "model_source": "qwen",
  "cfgs": ["qwen_10steps", "qwen_50steps"],
  "n_instructions": 2400,
  "source_files": ["...原始文件绝对路径..."],
  "dropped_keys": ["tts"],
  "format": "data[instr][steps_key] = {avg, per_seed, per_seed_subscores?}",
  "data": {
    "<instruction>": {
      "10steps": {"avg": 0.0047..., "per_seed": [s0, s1, s2, s3, s4]},
      "50steps": {"avg": ..., "per_seed": [...]}
    }
  }
}
```

- `avg`：5 个生成 seed 的均值（exp_815 score 模式的标签就是它；已校验 avg == mean(per_seed)）
- `per_seed`：seed_0..seed_4 逐 seed 分；**缺失为 null**（仅 unigen 部分 test 指令，
  见下）
- `per_seed_subscores`：仅 flux4b 有——每 seed 的子项分列表（源文件的 score_list）
- steps_key：qwen/flux/boogu 为 `10steps`、`50steps`；flux4b 为 `4steps`

## 归一化与取舍

- gm/am 的 qwen/flux train 各由两段拼合：`geneval2_{model}_edit_gen_perseed.json` + `gen2000_perseed.json`（edit_am 目录下同文件名）
- 源文件的 `tts` 档不属于 7-cfg roster，一律丢弃（含 tts 的旧版实验备份在 `results/backup_roster_with_tts/`）
- unigen 的 v2 源文件只有 avg；`per_seed` 从 `results/unigen_perseed_cache.json` 补齐：
  train 基本全覆盖（flux4b 2397/2400，即实验四的 per-seed 交集）；test 覆盖 400/600
  （flux4b 399/600），其余为 null
- flux4b 的 per-seed 原始值是 `{agg, score_list}`：agg 进 `per_seed`，score_list 进 `per_seed_subscores`

## 原始文件路径对照

| 文件 | source_files |
|---|---|
| `gm_train_{qwen,flux}` | `26AAAIworks/data/final_data/geneval2/edit/geneval2_{model}_edit_gen_perseed.json` + `gen2000_perseed.json` |
| `gm_test_{qwen,flux}` | `.../edit/geneval2_{model}_edit_raw_perseed.json` |
| `am_*_{qwen,flux}` | 同上，`edit_am/` 目录 |
| `gm_train_boogu` / `gm_test_boogu` | `Results/geneval2_boogu/eval/gen2400_perseed.json` / `raw800_perseed.json` |
| `am_train_boogu` / `am_test_boogu` | `Results/geneval2_boogu/eval/gen2400_am_v2_perseed.json` / `raw800_am_perseed.json` |
| `{gm,am}_*_flux4b` | `26AAAIworks/data/final_data/geneval2/direct/geneval2_flux4b_direct_{gen2400,raw}_{gm,am}_perseed.json` |
| `unigen_*_{qwen,flux}` | `Results/unigen_{model}/eval/{gen2400,raw600}_edit_v2.json` |
| `unigen_*_boogu` | `Results/unigenbench_boogu/eval/{gen2400,raw600}_tts_edit_v2.json` |
| `unigen_*_flux4b` | `Results/unigen_flux4b/eval/{gen2400,raw600}_direct_v2.json` |

## 与实验结果的关系

`raw_json/` 是**实验结果**归档（R²、路由质量等）；本文件夹是**标签底座**。
train/val/test 的 2200/200/800(600) 划分、四来源交集、score 矩阵 (N,7) 的构建逻辑见
`exp_815/common.py`（`_load_7cfg` + `load_dataset`）。
