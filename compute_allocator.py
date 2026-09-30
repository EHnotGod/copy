"""
compute_allocator.py

输入: 各档位的 (cost, mean_score) 和平均预算 B_avg
输出: 最优两档混合方案

原理: 目标是线性的，可行域是线段，最优解永远是两档混合。
      遍历所有满足 ci <= B_avg <= cj 的档位对，取期望分最高的。
"""

def best_allocation(tiers: list[tuple[float, float]], budget: float) -> dict:
    """
    Parameters
    ----------
    tiers  : list of (cost, mean_score), e.g. [(10, 24.21), (50, 24.82), (100, 44.82)]
    budget : average compute budget per case, e.g. 50

    Returns
    -------
    {
        "expected_score": float,
        "allocation": [(cost, mean_score, fraction), ...]   # only the two selected tiers
    }
    """
    best_score = -float("inf")
    best_plan = None

    for i in range(len(tiers)):
        for j in range(i, len(tiers)):
            ci, mi = tiers[i]
            cj, mj = tiers[j]

            if ci > budget or cj < budget:
                continue

            if ci == cj:
                if abs(ci - budget) < 1e-9:
                    score = mi
                    plan = [(ci, mi, 1.0)]
                else:
                    continue
            else:
                pj = (budget - ci) / (cj - ci)
                pi = 1.0 - pj
                score = pi * mi + pj * mj
                plan = [(ci, mi, pi), (cj, mj, pj)]

            if score > best_score:
                best_score = score
                best_plan = plan

    return {"expected_score": best_score, "allocation": best_plan}


if __name__ == "__main__":
    # 当前数据示例
    examples = {
        "geneval2/qwen":  [(10, 24.21), (50, 24.82), (100, 44.82)],
        "geneval2/flux":  [(10, 31.63), (50, 33.21), (100, 37.21)],
        "corebench/qwen": [(10, 45.63), (50, 46.70), (100, 58.60)],
        "corebench/flux": [(10, 49.43), (50, 51.40), (100, 59.05)],
        "wise/qwen":      [(10, 74.57), (50, 73.49), (100, 88.22)],
        "wise/flux":      [(10, 75.66), (50, 75.95), (100, 87.82)],
    }

    budget = 50.0

    print(f"{'dataset':<18} {'expected':>9}  allocation")
    print("-" * 60)
    for name, tiers in examples.items():
        result = best_allocation(tiers, budget)
        alloc_str = " + ".join(
            f"{f*100:.1f}%×cost{int(c)}" for c, m, f in result["allocation"]
        )
        print(f"{name:<18} {result['expected_score']:>9.2f}  {alloc_str}")
