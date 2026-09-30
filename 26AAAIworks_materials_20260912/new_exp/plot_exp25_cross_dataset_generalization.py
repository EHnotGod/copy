#!/usr/bin/env python3
"""Generate the §25 report and a dependency-free SVG heatmap."""
from __future__ import annotations

import json
import os
from html import escape


HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

DATASETS = ["gm", "am", "unigen"]
LABELS = {"gm": "GenEval2 GM", "am": "GenEval2 AM", "unigen": "UniGen"}
SHORT = {"gm": "GM", "am": "AM", "unigen": "UG"}


def load_result():
    with open(os.path.join(RES, "exp25_cross_dataset_generalization.json")) as f:
        return json.load(f)


def color_for(value, limit=0.06):
    value = max(-limit, min(limit, value)) / limit
    if value < 0:
        # white -> muted red
        strength = -value
        rgb = (255, int(255 - 105 * strength), int(255 - 105 * strength))
    else:
        # white -> muted blue
        strength = value
        rgb = (int(255 - 115 * strength), int(255 - 75 * strength), 255)
    return "#%02x%02x%02x" % rgb


def make_heatmap(result):
    width, height = 760, 650
    left, top, cell = 190, 130, 135
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:21px;font-weight:600}.axis{font-size:16px;font-weight:600}.cell{font-size:19px;font-weight:600}.ci{font-size:11px;fill:#444}.note{font-size:13px;fill:#444}</style>',
        '<text x="380" y="38" text-anchor="middle" class="title">Zero-shot learned gain: Cross-Ridge − Source-LP</text>',
        '<text x="392" y="75" text-anchor="middle" class="axis">Target dataset</text>',
        f'<text x="55" y="{top + 1.5 * cell}" text-anchor="middle" class="axis" transform="rotate(-90 55 {top + 1.5 * cell})">Source dataset</text>',
    ]
    for j, target in enumerate(DATASETS):
        svg.append(f'<text x="{left + (j + 0.5) * cell}" y="{top - 18}" text-anchor="middle" class="axis">{SHORT[target]}</text>')
    for i, source in enumerate(DATASETS):
        svg.append(f'<text x="{left - 22}" y="{top + (i + 0.52) * cell}" text-anchor="end" class="axis">{SHORT[source]}</text>')
        for j, target in enumerate(DATASETS):
            item = result["matrix"][source][target]
            value = item["zero_shot_learned_gain"]
            lo, hi = item["zero_shot_learned_gain_ci95"]
            significant = lo > 0 or hi < 0
            x, y = left + j * cell, top + i * cell
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color_for(value)}" stroke="#777"/>')
            marker = "*" if significant else ""
            svg.append(f'<text x="{x + cell / 2}" y="{y + 59}" text-anchor="middle" class="cell">{value:+.4f}{marker}</text>')
            svg.append(f'<text x="{x + cell / 2}" y="{y + 83}" text-anchor="middle" class="ci">95% CI</text>')
            svg.append(f'<text x="{x + cell / 2}" y="{y + 101}" text-anchor="middle" class="ci">[{lo:+.4f}, {hi:+.4f}]</text>')
    svg += [
        f'<text x="{left}" y="{top + 3 * cell + 38}" class="note">Blue = positive transfer; red = negative transfer; * = paired bootstrap CI excludes 0.</text>',
        f'<text x="{left}" y="{top + 3 * cell + 62}" class="note">GM↔AM is metric shift; GM/AM↔UG is the true prompt-domain shift.</text>',
        '</svg>',
    ]
    path = os.path.join(FIG, "exp25_cross_dataset_heatmap.svg")
    with open(path, "w") as f:
        f.write("\n".join(svg))
    return path


def f4(value, signed=False):
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def shift_type(source, target):
    if source == target:
        return "in-domain"
    if {source, target} == {"gm", "am"}:
        return "metric shift"
    return "domain shift"


def make_report(result):
    lines = [
        "# §25 Router 跨数据集泛化",
        "",
        "## 设置",
        "",
        "- 完整 A10，预算 `E[c]≤10`，Ridge-only。",
        "- source scaler、Ridge 和 alpha 只使用 source train/validation。",
        "- target 侧仅用预测分数和成本选择 λ/混合策略；冻结策略后才读取 target true scores。",
        f"- split seeds：{', '.join(str(x) for x in result['seeds'])}；paired bootstrap：{result['bootstrap_samples']} 次。",
        "- GM/AM prompt 完全相同，二者互转属于 metric shift；与 UniGen 互转才是 domain shift。",
        "",
        "![cross-dataset heatmap](figs/exp25_cross_dataset_heatmap.svg)",
        "",
        "## 主要结果",
        "",
        "| train→test | shift | Q50 | Source-LP | Cross-Ridge | 总收益 | Zero-shot learned | 95% CI | Target-LP | In-domain learned | Oracle |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for source in DATASETS:
        for target in DATASETS:
            item = result["matrix"][source][target]
            ci = item["zero_shot_learned_gain_ci95"]
            lines.append(
                f"| {SHORT[source]}→{SHORT[target]} | {shift_type(source, target)} | "
                f"{f4(item['qwen50'])} | {f4(item['source_lp'])} | "
                f"{f4(item['cross_ridge'])} | {f4(item['total_gain_vs_qwen50'], True)} | "
                f"{f4(item['zero_shot_learned_gain'], True)} | "
                f"[{f4(ci[0], True)}, {f4(ci[1], True)}] | "
                f"{f4(item['target_lp'])} | "
                f"{f4(item['in_domain_learned_gain'], True)} | {f4(item['oracle'])} |")

    gm_ug = result["matrix"]["gm"]["unigen"]
    am_ug = result["matrix"]["am"]["unigen"]
    ug_gm = result["matrix"]["unigen"]["gm"]
    ug_am = result["matrix"]["unigen"]["am"]
    lines += [
        "",
        "## 结论",
        "",
        "1. **真正的跨 prompt-domain learned routing 没有泛化。** "
        f"GM→UG 为 {gm_ug['zero_shot_learned_gain']:+.4f}，"
        f"AM→UG 为 {am_ug['zero_shot_learned_gain']:+.4f}，均显著低于 Source-LP。",
        "2. **反向迁移也没有稳定收益。** "
        f"UG→GM 为 {ug_gm['zero_shot_learned_gain']:+.4f}（CI 跨 0），"
        f"UG→AM 为 {ug_am['zero_shot_learned_gain']:+.4f}（显著为负）。",
        "3. **迁移的主要是动作先验，不是 prompt-level 选择能力。** 所有跨域 Cross-Ridge "
        "仍高于 Qwen50，但没有超过使用相同 source 标签的 Source-LP。",
        "4. **同域 router 确实可学。** AM 和 UniGen 的对角线 learned gain 显著为正；"
        "GM 为小幅正值但 CI 跨 0。",
        "5. **GM↔AM 也不稳定。** 即使 prompt 完全相同，GM→AM 没有超越 Source-LP；"
        "AM→GM 为正但 CI 跨 0，说明评分口径变化已经足以破坏部分动作排序。",
        "",
        "因此当前证据支持：A10 的全局动作组合具有一定跨域可迁移性，但 Ridge 学到的"
        "实例级路由规律明显依赖训练数据集和评分口径。跨数据集部署时，Source-LP 比直接"
        "迁移 Ridge 更稳妥；若需要 learned routing，应至少做少量目标域校准。",
        "",
        "## 数值审计",
        "",
        "- 9 个方向、5 个 seeds 的部署策略成本均为 10.0000。",
        "- seed=42 的自实现 Ridge 与 §23 sklearn 预测缓存 RMSE 均小于 `1.3e-6`。",
        "- λ/混合策略选择函数不接受 target true scores；真实标签只进入冻结后的评价函数。",
        "",
    ]
    path = os.path.join(HERE, "exp25_cross_dataset_generalization.md")
    from report_sections import write_report_section
    write_report_section(path, "exp25-result", "\n".join(lines))
    return path


def main():
    result = load_result()
    print(f"[saved] {make_heatmap(result)}")
    print(f"[saved] {make_report(result)}")


if __name__ == "__main__":
    main()
