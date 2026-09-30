# 实验二c：路由质量（RQ）× Ridge-only（固定变量）

**预测器**：Ridge + Qwen3-Embedding-8B (4096 dim)，α ∈ {1,10,100,1000}，external val R² 选最优。
**路由**：Lagrangian sweep → (avg_cost, avg_quality) → budget 处线性插值。
**n_train**：2200（pool 2400 去掉 200 fixed ext_val），3 seeds 均值。

| dataset | model | RQ | oracle | 50steps | gain |
|---|---|---|---|---|---|
| GenEval2 GM | qwen | 0.3269 | 0.4058 | 0.2482 | +0.0787 |
| GenEval2 GM | flux | 0.3449 | 0.4335 | 0.3321 | +0.0128 |
| GenEval2 GM | boogu | 0.4022 | 0.4921 | 0.3947 | +0.0075 |
| GenEval2 AM | flux | 0.7666 | 0.8077 | 0.7680 | -0.0014 |
| GenEval2 AM | qwen | 0.7596 | 0.7953 | 0.7300 | +0.0295 |
| GenEval2 AM | boogu | 0.8085 | 0.8416 | 0.8130 | -0.0045 |
| UniGenBench | boogu | 0.9106 | 0.9430 | 0.8887 | +0.0219 |
| UniGenBench | flux | 0.8680 | 0.9012 | 0.8275 | +0.0405 |
| UniGenBench | qwen | 0.8399 | 0.8782 | 0.7518 | +0.0881 |
