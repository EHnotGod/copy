# §34 GenEval2 result-aware RAG preview

## 方法

复用 §26/§27 已有的 Flux4B probe 图和 Qwen3-VL-8B embedding，不新增 GPU 推理。
Base predictor 是 matched `prompt+probe-image` QEval embedding 上的 Ridge；RAG 仅检验
“在强 result-aware 表征上再检索相似历史案例”是否有额外收益。

检索键：

- `prompt`：原 prompt embedding；
- `joint`：Qwen3-VL matched prompt+probe embedding；
- `visual_delta`：matched embedding − text-only embedding，作为图片特有信息的代理；
- `prompt+visual`：prompt cosine 和 visual-delta cosine 各 0.5；
- 同时测试 joint/visual-delta 先减训练均值再做 cosine，修正 embedding 各向异性。

检索值仍为 QEval-Ridge 的逐动作训练残差。拆分为 2000 train / 200 alpha-val /
200 RAG-val / 800 test，seeds=42/43/44。Flux probe 成本已计入 cascade cost；
Qwen3-VL evaluator 成本仍未计入，因此这里只是信息价值实验。

## Validation 冻结后的主结果

| 动作库 | QEval Ridge test | 选中 RAG test | ΔRAG | 95% CI |
|---|---:|---:|---:|---:|
| 完整 A10 | 0.4002 | 0.3988 | -0.0014 | [-0.0055,+0.0025] |
| QTTS+F10+B10 | 0.3939 | 0.3939 | +0.0001 | [-0.0043,+0.0045] |
| QTTS+F10+B10+F4B4 | 0.4021 | 0.4029 | +0.0007 | [-0.0025,+0.0042] |

四动作库中，预先单列的 `prompt+visual` 模式达到 0.4032，相对同库 QEval Ridge
+0.0010，但 CI `[-0.0024,+0.0045]` 跨零；而在全部检索模式中按 validation 选择的是
centered visual-delta，冻结 test 只增加 +0.0007。不能据此声称 RAG 有收益。

## 为什么直接图文检索没奏效

- raw joint embedding 的 test top-1 cosine 均值高达 0.995、std 仅 0.0016，公共 template/
  表征方向占主导，邻居区分度很弱。
- 中心化后 top-1 cosine 降到约 0.886、方差变大，但 Router 仍未改善，说明问题不只是
  embedding 各向异性。
- QEval Ridge 本身已经吸收了 probe 图信息；相似样本残差没有稳定的局部相关结构。
- “图文相似”仍不等价于“失败方式相同、同一个后续动作能救回来”。

## 下一版应改成显式 deficit-aware RAG

1. 对 probe 图回答 GenEval2 原子问题，输出结构化状态：pass/fail、observed answer、错误类型
   （count/attribute/spatial/text 等）和置信度。
2. 第一阶段按 prompt 召回约 100 条；第二阶段按失败原子/错误类型重排，避免全库昂贵图文检索。
3. 检索库存的 value 改成 `score(action) − score(probe)` 或相对当前动作的 rescue gain，
   不再用通用 Ridge residual。
4. 必须加入 deficit-shuffled 对照，并完整计入 Flux probe 与 Qwen3.5-2B evaluator 成本。

这个结构才真正回答“当前图哪里不足，以及历史上哪个动作修好了同类不足”。

代码：`run_exp34_result_aware_rag_preview.py`；结果：
[results/exp34_result_aware_rag_preview.json](results/exp34_result_aware_rag_preview.json)。
