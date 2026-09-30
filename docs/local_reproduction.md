> **原项目操作记录**：下文命令用于原计算环境。交流包的使用与本地文件校验见 [阅读说明](../README.md)。

# 本地检查与 CPU 复现

2026-09-12。适用于 `new_exp/results/prompt_disjoint_v1/` 已修复的 CPU 分支。标签补齐、Hope、CLIP 重提取和图像 preview 跨域实验暂缓。这里不把历史 KNN/TFMLP、§32 或全部旧实验算作本轮重跑完成。

## 只核对已有产物

在项目根目录运行；仅需 Python 标准库，不加载模型或启动实验：

```bash
python3 new_exp/verify_local_results.py
```

检查已登记结果的 SHA-256、JSON 可解析且数值有限、各 CPU 分支应有的主结果、UniGen 冻结划分与数据清单的 prompt 哈希、本地 Markdown 路径和 `new_exp` 顶层 Python 语法。失败返回非零退出码。

该检查不重新提取或比对源 embedding、不访问远程链接、不检查 Markdown 锚点，也不替代训练与统计方法审查。原始标签/embedding 的数据身份由实际复现时的加载器再核对。

保存本次检查报告可加 `--output docs/local_verification_20260912.json`；不传该参数时不写文件。结果 JSON 和原始数据不会被该命令修改。

## CPU 环境与回归检查

当前已验证的项目环境为 hl02 上的 venv，直接调用解释器，不使用 conda。实测版本：Python 3.11.8、NumPy 2.4.4、SciPy 1.17.1、scikit-learn 1.9.0、PyTorch 2.8.0+cu128、Matplotlib 3.11.0、Transformers 4.57.1。版本记录用于复现定位，不表示其他版本已经验证：

```bash
ROUTER_PY=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/flux2_env/bin/python
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "$ROUTER_PY" -m unittest discover -s new_exp -p 'test_*.py' -v
```

原有11项检查覆盖 prompt 对齐、训练/测试分离、数据变更拒绝缓存、CPU Ridge 等价性、预算分配不依赖测试真值、训练点大小和 CLIP 缓存顺序。新增4项覆盖结果文件同长度篡改、缺失结果、全新结果目录和非有限 JSON；这些检查都只用 CPU 与小型构造数据。

## 查看或续跑 CPU 工作

只打印命令，无需科学计算环境，也不创建结果目录：

```bash
python3 new_exp/run_cpu_repairs.py --suite all --plan
```

实际续跑：

```bash
"$ROUTER_PY" new_exp/run_cpu_repairs.py --suite all --threads 4
```

`--suite` 可选 `legacy`（§18–22 Ridge）、`core`（动作增长、泛化与样本效率）、`selection`（§31、§35–37），默认 `all`。所有已登记且哈希一致的完整结果组会跳过，包括14份旧 Ridge 重跑结果；缺失组才进入复现。未登记文件不被视为已核验完成。

脚本会先核对源数据身份，再复用缓存；报告生成也会先验证已登记结果，不能把改坏的 JSON 重新计算哈希后当成合法结果。源数据改变或希望完整重做时，使用新目录：

```bash
ROUTER_RESULTS_DIR="$PWD/new_exp/results/<新的数据版本名>" "$ROUTER_PY" new_exp/run_cpu_repairs.py --suite all --threads 4
```

新目录仍可复用只读原始特征。不要把 `ROUTER_RESULTS_DIR` 指向历史 `new_exp/results/`，也不要通过删除旧清单来绕过身份校验。完整 CPU 重做有实际运行成本，不是检查命令。

## 更新实验说明

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "$ROUTER_PY" new_exp/update_repair_reports.py
```

该命令读取结果，更新实验页首的修复章节、状态清单和 Flux4 汇总图。旧正文折叠保留；人工新增说明放在自动章节之外。实验解读的更新不代表重新训练，当前数值来源仍是2026-09-11的修复结果。

## 本轮保存位置

- [检查报告](local_verification_20260912.json)：可重复运行上述核对命令生成。
- [修改前备份](../附件范围.md#attachment-1)：57份相关文档、脚本及状态清单，按项目相对路径保存；结果 JSON 本轮没有重算或改写。
- [实验修复完成清单](../new_exp/results/prompt_disjoint_v1/repair_status.json)：各结果哈希和暂缓事项。
- [写作入口](../writing/README.md)：当前主张与实验来源对应。
