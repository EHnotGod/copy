# backup_lp_equality：LP baseline 旧实现（E[c] = B 等成本混合）的结果备份

下线原因（2026-08-21）：`run_exp2_routing.py` 的 `lp_quality()` 旧实现照搬 exp_815，
把 LP 的预算约束实现成 E[c]=B（强制花满预算）：hull 段内用 alpha=(B−c_lo)/(c_hi−c_lo)
强制等成本混合，budget 超过 hull 最高成本时停在最贵顶点。正确约束是 E[c]≤B
（允许预算闲置，V_mix(B) 单调不降，高预算应停在质量峰值顶点纯分配）。
旧实现导致 sharp/imagereward/aesthetic 的 LP 随 budget 下降（infinity 最便宜且
质量最好，却被强制混向贵而差的成员）。

本目录为修正前最后一版结果：exp2_routing.json / exp2_routing.md /
exp2_deferral.json / exp2_deferral_full.png（四指标 × 6 budget，其中
predictor/Oracle/OC/fixed 数字与修正后逐值一致，仅 LP 列作废）。
