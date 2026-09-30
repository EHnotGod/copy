# §33 GenEval2 text-only RAG Router preview

## 方法

- GenEval2 GM，split seeds=42/43/44。
- 2000 条 Ridge/train+检索库，200 条 alpha-val，另 200 条 RAG 参数选择，800 test。
- Query 和检索库均使用现有 Qwen3-Embedding-8B prompt embedding，cosine top-k。
- 检索值不是邻居原始分数，而是各动作的 Ridge 训练残差。
- `prediction_RAG = prediction_Ridge + beta × top-k mean residual`。
- validation 搜索 `k∈{4,8,16,32,64}`、`beta∈{0.25,0.5,1}`；test 不参与选择。
- 原始动作成本、budget=10；不需要图像生成、VLM 或 GPU。

## 结果

| 动作库 | Validation ΔRAG | Ridge test | RAG test | Test ΔRAG | 95% CI |
|---|---:|---:|---:|---:|---:|
| 完整 A10 | +0.0051 | 0.3816 | 0.3792 | -0.0023 | [-0.0060,+0.0012] |
| QTTS+F10+B10 | +0.0015 | 0.3916 | 0.3906 | -0.0011 | [-0.0029,+0.0007] |
| QTTS+F10+B10+F4B4 | +0.0052 | 0.3870 | 0.3855 | -0.0015 | [-0.0044,+0.0015] |

三个动作库均出现 validation 正增益、test 负增益。检索超参数在小 validation 上追到了
局部噪声，未形成可泛化收益。虽然 test top-1 邻居 cosine 约 0.906，但 prompt 语义相似
不代表“哪个动作相对更好”的残差相似。

## 结论

纯文本语义 RAG 不值得继续扩 k、融合权重或 predictor 变体。若继续 RAG，应改变检索键，
使其描述 probe 图的失败状态，而不是只描述 prompt 主题。

代码：`run_exp33_rag_preview.py`；结果：[results/exp33_rag_preview.json](results/exp33_rag_preview.json)。
