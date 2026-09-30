#!/usr/bin/env python3
"""Generate the §26 report and a dependency-free SVG summary."""
from __future__ import annotations

import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figs")
RESULT = os.path.join(RES, "exp26_probe_image_router_gm.json")
REPORT = os.path.join(HERE, "exp26_probe_image_router.md")
SVG = os.path.join(FIG, "exp26_probe_image_router.svg")


def load_result():
    with open(RESULT) as f:
        return json.load(f)


def f4(value, signed=False):
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def make_svg(result):
    os.makedirs(FIG, exist_ok=True)
    width, height = 980, 560
    left, top, plot_h = 85, 105, 315
    panel_w = 380
    qualities = [
        ("LP\n(cascade)", result["baselines"]["lp_probe"]["quality"], "#9aa0a6"),
        ("Prompt\n(original)", result["routes"]["prompt_original_cost"]["quality"], "#6f92c8"),
        ("Prompt\n(cascade)", result["routes"]["prompt_probe_cost"]["quality"], "#9bb7de"),
        ("QEval-only\n(cascade)", result["routes"]["qeval_only_probe_cost"]["quality"], "#2f70b7"),
        ("Prompt+QEval\n(cascade)", result["routes"]["prompt_qeval_probe_cost"]["quality"], "#6a9fd4"),
    ]
    y_min, y_max = 0.36, 0.415
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:22px;font-weight:600}.panel{font-size:17px;font-weight:600}.tick{font-size:12px;fill:#555}.label{font-size:12px}.value{font-size:13px;font-weight:600}.note{font-size:13px;fill:#444}</style>',
        '<text x="490" y="36" text-anchor="middle" class="title">Flux4B probe + Qwen evaluation embedding</text>',
        '<text x="275" y="76" text-anchor="middle" class="panel">A. Router quality (budget = 10)</text>',
    ]
    for tick in [0.36, 0.37, 0.38, 0.39, 0.40, 0.41]:
        y = top + plot_h * (y_max - tick) / (y_max - y_min)
        svg += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" stroke="#e6e6e6"/>',
            f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>',
        ]
    bar_w, gap = 56, 18
    for i, (label, value, color) in enumerate(qualities):
        x = left + 9 + i * (bar_w + gap)
        y = top + plot_h * (y_max - value) / (y_max - y_min)
        base = top + plot_h
        svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{base-y:.1f}" rx="3" fill="{color}"/>')
        svg.append(f'<text x="{x + bar_w/2}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.4f}</text>')
        for j, line in enumerate(label.split("\n")):
            svg.append(f'<text x="{x + bar_w/2}" y="{base + 22 + j*15}" text-anchor="middle" class="label">{line}</text>')

    right_left, right_w = 575, 330
    svg.append('<text x="740" y="76" text-anchor="middle" class="panel">B. Information gain by split seed</text>')
    gain_min, gain_max = 0.0, 0.045
    for tick in [0.0, 0.01, 0.02, 0.03, 0.04]:
        y = top + plot_h * (gain_max - tick) / (gain_max - gain_min)
        svg += [
            f'<line x1="{right_left}" y1="{y:.1f}" x2="{right_left + right_w}" y2="{y:.1f}" stroke="#e6e6e6"/>',
            f'<text x="{right_left - 9}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:+.2f}</text>',
        ]
    rows = result["split_stability"]["per_seed"]
    xs = [right_left + 55 + i * 90 for i in range(len(rows))]
    for x, row in zip(xs, rows):
        value = row["information_gain"]
        y = top + plot_h * (gain_max - value) / (gain_max - gain_min)
        base = top + plot_h
        svg.append(f'<rect x="{x-23}" y="{y:.1f}" width="46" height="{base-y:.1f}" rx="3" fill="#2f70b7"/>')
        svg.append(f'<text x="{x}" y="{y-8:.1f}" text-anchor="middle" class="value">{value:+.4f}</text>')
        svg.append(f'<text x="{x}" y="{base+23}" text-anchor="middle" class="label">seed {row["seed"]}</text>')
    agg = result["split_stability"]["aggregate"]["information_gain"]
    lo, hi = agg["mean_over_splits_paired_bootstrap_ci95"]
    svg += [
        f'<text x="{right_left + right_w/2}" y="472" text-anchor="middle" class="note">mean {agg["mean"]:+.4f}; paired 95% CI [{lo:+.4f}, {hi:+.4f}]</text>',
        '<text x="490" y="520" text-anchor="middle" class="note">Flux generation cost is included. Qwen3-VL evaluator compute is excluded: this is an information-value screen, not final net utility.</text>',
        '</svg>',
    ]
    with open(SVG, "w") as f:
        f.write("\n".join(svg))
    return SVG


def make_report(result):
    routes = result["routes"]
    cond = result["conditions"]
    cmp = result["comparisons"]
    stability = result["split_stability"]
    info = cmp["information_gain"]
    e2e = cmp["optimistic_e2e_gain_excluding_qeval_compute"]
    learned = cmp["learned_gain_over_probe_lp"]
    lines = [
        "# §26 结果图评价 embedding 辅助 Router",
        "",
        "## 设置",
        "",
        "- GenEval2 GM、完整 A10、预算 `E[c]≤10`，Ridge-only。",
        "- 每条 prompt 先用 `flux4b_4steps / seed_0` 生成 probe 图；Flux 成本 0.455 已计入，选择 Flux 时复用该图。",
        "- Qwen3-VL-8B-Instruct 同时读取原 prompt 和 probe 图。不生成评价文字或 vocabulary logits，只取固定评价 query 最后一个非 padding token 的 4096 维 hidden state。",
        "- 目标标签仍是每动作 5 个 image seeds 的平均质量。训练划分为 2200/200，测试 800；split seeds 为 " + ", ".join(str(x) for x in result["split"]["stability_seeds"]) + "。",
        "- Qwen evaluator 自身计算成本尚未换算为 TFLOPs，因此以下端到端收益是乐观上界。",
        "",
        "![probe image router](figs/exp26_probe_image_router.svg)",
        "",
        "## seed=42 主结果",
        "",
        "| Router 输入 | Test macro R² | Top-1 action acc | Router quality | 实际成本 |",
        "|---|---:|---:|---:|---:|",
        f"| Prompt（原成本） | {f4(cond['prompt']['test']['macro'])} | {cond['prompt']['test']['top1_action_accuracy']:.4f} | {f4(routes['prompt_original_cost']['quality'])} | {routes['prompt_original_cost']['cost']:.4f} |",
        f"| Prompt（计 probe 成本） | {f4(cond['prompt']['test']['macro'])} | {cond['prompt']['test']['top1_action_accuracy']:.4f} | {f4(routes['prompt_probe_cost']['quality'])} | {routes['prompt_probe_cost']['cost']:.4f} |",
        f"| **QEval embedding only** | **{f4(cond['qeval_only']['test']['macro'])}** | **{cond['qeval_only']['test']['top1_action_accuracy']:.4f}** | **{f4(routes['qeval_only_probe_cost']['quality'])}** | {routes['qeval_only_probe_cost']['cost']:.4f} |",
        f"| Prompt + QEval | {f4(cond['prompt_qeval']['test']['macro'])} | {cond['prompt_qeval']['test']['top1_action_accuracy']:.4f} | {f4(routes['prompt_qeval_probe_cost']['quality'])} | {routes['prompt_qeval_probe_cost']['cost']:.4f} |",
        f"| Prompt + true Flux score（不可部署） | {f4(cond['prompt_true_probe_score']['test']['macro'])} | {cond['prompt_true_probe_score']['test']['top1_action_accuracy']:.4f} | {f4(routes['prompt_true_probe_score_probe_cost']['quality'])} | {routes['prompt_true_probe_score_probe_cost']['cost']:.4f} |",
        "",
        "QEval-only 是正式主条件；该 embedding 已由 prompt+image 联合编码。直接再拼 4096 维 prompt embedding 将维度翻倍，seed=42 的质量反而从 0.3997 降到 0.3970，因此只保留为拼接消融。",
        "",
        "## 三种收益口径",
        "",
        "| 比较 | seed=42 | 95% CI | 3-split mean | 3-split paired 95% CI |",
        "|---|---:|---|---:|---|",
    ]
    agg = stability["aggregate"]
    for label, key, item in [
        ("信息增益：QEval(cascade) − Prompt(cascade)", "information_gain", info),
        ("乐观端到端：QEval(cascade) − Prompt(original)", "optimistic_e2e_gain_excluding_qeval_compute", e2e),
        ("Learned gain：QEval(cascade) − LP(cascade)", "learned_gain_over_probe_lp", learned),
    ]:
        ci = item["ci95"]
        aci = agg[key]["mean_over_splits_paired_bootstrap_ci95"]
        lines.append(
            f"| {label} | {f4(item['value'], True)} | [{f4(ci[0], True)}, {f4(ci[1], True)}] | "
            f"{f4(agg[key]['mean'], True)} | [{f4(aci[0], True)}, {f4(aci[1], True)}] |")

    diag = result["diagnostics"]
    p_rank = diag["pairwise_ranking_accuracy"]["prompt"]
    q_rank = diag["pairwise_ranking_accuracy"]["qeval_only"]
    static = diag["matched_allocation_static_control"]["qeval_only_probe_cost"]
    lines += [
        "",
        "## 诊断",
        "",
        f"- 相对 Flux4B 的 pairwise 动作排序准确率从 {p_rank['vs_flux4b_4steps']:.4f} 提高到 {q_rank['vs_flux4b_4steps']:.4f}；相对 Qwen50 从 {p_rank['vs_qwen_50steps']:.4f} 提高到 {q_rank['vs_qwen_50steps']:.4f}。",
        f"- 不是只靠改变全局动作比例：固定为 QEval router 的同一动作 allocation、但随机分配给 prompt 时，质量只有 {static['same_allocation_static_quality']:.4f}；实例级路由达到 {static['router_quality']:.4f}，差 {static['instance_routing_gain']:+.4f}。",
        "- 按 probe 真值质量分桶：",
        "",
        "| Flux probe quartile | n | Prompt(cascade) | QEval(cascade) | 信息增益 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in diag["probe_true_quality_quartiles"]:
        lines.append(
            f"| Q{row['quartile']} | {row['count']} | {f4(row['prompt_probe_quality'])} | "
            f"{f4(row['qeval_only_probe_quality'])} | {f4(row['information_gain'], True)} |")
    lines += [
        "",
        "## 结论",
        "",
        "这个预实验支持原假设：prompt-only 的信息确实不够，快速生成结果能暴露 prompt 本身无法体现的实例难度和生成失败模式。QEval embedding 对全部 10 个动作的预测均有帮助，其中 Flux4B 动作 R² 提升最大（约 0.357→0.685）。",
        "",
        "但它目前只证明了信息价值，尚未证明最终部署净收益。Flux probe 成本已经完整计入，而 Qwen3-VL-8B evaluator 的 TFLOPs 尚未计入；8B VLM 若每次都运行，很可能过重。下一步应先准确核算 evaluator FLOPs，再决定采用轻量 VLM、低层/低分辨率特征，或把 QEval embedding 蒸馏到小型 image-prompt encoder。",
        "",
        "## 数值审计",
        "",
        "- seed=42/43/44 的所有冻结路由策略实际平均成本均为 10（浮点误差小于 `2e-15`）。",
        "- λ 与混合比例仅由 prediction 和 action cost 选择；test true score 只在策略冻结后进入评价。",
        "- 3200 个 probe 图均使用 seed_0；同一 prompt 没有跨 train/test 泄漏。",
        "",
    ]
    from report_sections import write_report_section
    write_report_section(REPORT, "exp26-result", "\n".join(lines))
    return REPORT


def main():
    result = load_result()
    print(f"[saved] {make_svg(result)}")
    print(f"[saved] {make_report(result)}")


if __name__ == "__main__":
    main()
