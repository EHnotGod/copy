# CATImage router 复现（扩展到 corebench / coco）

完整复现 CATImage（arXiv 2506.14753）的路由框架 + 两个 predictor：
- **路由**：λ-sweep，每 prompt 独立 `argmax(γ̂(m) − λ·cost(m))`
- **CATImage-KNN**：CLIP text embedding 空间 k 近邻均值（非参，k=50）
- **CATImage-TFMLP**：frozen CLIP-ViT-L/14 per-token → Transformer-MLP → M 档质量概率（忠实 Appendix D）
- **指标**：AUDC（area under deferral curve，横轴 cost 归一到 [10,100]），越高越好
- **设置**：gen 库 300 train / 100 val，raw 全量测试，seed 0/1/2 均值±std
- coco：2000 raw 按 seed 20260723 切 400 train / 1600 test

## geneval2（gen训 300:100 → raw 800 测）

| bench | model | stage | oracle | CATImage-KNN | CATImage-TFMLP |
|---|---|---|---|---|---|
| ge | qwen | edit | 0.4269 | 0.3579±0.0012 (83.8%) | 0.3585±0.0020 (84.0%) |
| ge | flux | edit | 0.4130 | 0.3453±0.0004 (83.6%) | 0.3472±0.0021 (84.1%) |

## corebench（gen训 300:100 → raw 1080 测）

| bench | model | stage | oracle | CATImage-KNN | CATImage-TFMLP |
|---|---|---|---|---|---|
| co | qwen | edit | 0.5806 | 0.5393±0.0008 (92.9%) | 0.5421±0.0025 (93.4%) |
| co | flux | edit | 0.5971 | 0.5576±0.0007 (93.4%) | 0.5593±0.0008 (93.7%) |

## wise（gen训 300:100 → raw 1000 测）

| bench | model | stage | oracle | CATImage-KNN | CATImage-TFMLP |
|---|---|---|---|---|---|
| wi | qwen | edit | 0.8757 | 0.8223±0.0010 (93.9%) | 0.8279±0.0046 (94.5%) |
| wi | flux | edit | 0.8800 | 0.8285±0.0007 (94.1%) | 0.8339±0.0009 (94.8%) |

## coco（2000 raw 切 400/1600）

| bench | model | stage | oracle | CATImage-KNN | CATImage-TFMLP |
|---|---|---|---|---|---|
| co | qwen | edit | 0.9579 | 0.9329±0.0003 (97.4%) | 0.9333±0.0019 (97.4%) |
| co | flux | edit | 0.9696 | 0.9498±0.0004 (98.0%) | 0.9498±0.0005 (98.0%) |

## 汇总对比（%oracle，edit stage）

| bench | model | oracle | CATImage-KNN | CATImage-TFMLP |
|---|---|---|---|---|
| geneval2 | qwen | 0.4269 | 0.3579 (83.8%) | 0.3585 (84.0%) |
| geneval2 | flux | 0.4130 | 0.3453 (83.6%) | 0.3472 (84.1%) |
| corebench | qwen | 0.5806 | 0.5393 (92.9%) | 0.5421 (93.4%) |
| corebench | flux | 0.5971 | 0.5576 (93.4%) | 0.5593 (93.7%) |
| wise | qwen | 0.8757 | 0.8223 (93.9%) | 0.8279 (94.5%) |
| wise | flux | 0.8800 | 0.8285 (94.1%) | 0.8339 (94.8%) |
| coco | qwen | 0.9579 | 0.9329 (97.4%) | 0.9333 (97.4%) |
| coco | flux | 0.9696 | 0.9498 (98.0%) | 0.9498 (98.0%) |

## 结论

- 8 组结果中：KNN 领先 0 组，TFMLP 领先 4 组，持平（差距在 std 内）4 组。
- corebench / wise 上两者接近，geneval2 上 TFMLP 略优 KNN，coco 上两者几乎相同。
- 整体与论文原始观察一致：小数据（300 条）下非参 KNN 与参数化 TFMLP 差距甚微。