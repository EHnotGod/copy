#!/usr/bin/env python3
"""
§14 Routing Tricks: gm/qwen, CLIP mean-pool + Ridge
只改训练目标 / 路由后处理，不改特征。

variants:
  score          baseline: Ridge 预测 3 个绝对分
  delta          Ridge 预测 [q50-q10, qtts-q50]，路由分 [-d1, 0, d2]
  delta_costnorm Ridge 预测 [(q50-q10)/(c50-c10), (qtts-q50)/(ctts-c50)]
  rank_norm      Ridge 预测 q_j - mean_query(q) （exp13 best）
  rank_zscore    Ridge 预测 (q_j - mean_q) / (std_q + ε)
  budget_tuned   score 预测 + val 上 grid-search 最优 B'
  offset_calib   score 预测 + val 上搜索 per-config 加法偏置 a_j
  bootstrap      10 Ridge 模型（80% 子采样）平均预测
"""
import os, sys, json, random, argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import itertools

HERE  = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.join(HERE, "..", "codes")
sys.path.insert(0, CODES)
import catimage_tf_mlp as TF

DATA     = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/data/final_data"
CLIP_DIR = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/zhouwei/clip-vit-large-patch14"
CFGS     = ["10steps", "50steps", "tts"]
COSTS    = np.array([2.0, 10.0, 30.5])
BUDGET   = 10.0
N_TRAIN  = 2200
N_VAL    = 200

LAMBDAS  = np.unique(np.concatenate([
    np.linspace(0, 0.05, 2000), np.linspace(0.05, 3.0, 700), [10., 1e6]]))


# ── 数据 ──────────────────────────────────────────────────────
def load_scores(path):
    d = json.load(open(path))
    instrs, scores = [], {c: [] for c in CFGS}
    for x in d:
        sc = x["scores"]; vals = {}; ok = True
        for c in CFGS:
            v = sc.get(c)
            if v is None: ok = False; break
            if isinstance(v, dict): v = v.get("avg")
            if v is None: ok = False; break
            vals[c] = float(v)
        if not ok: continue
        instrs.append(x["instruction"])
        for c in CFGS: scores[c].append(vals[c])
    return instrs, {c: np.array(scores[c], np.float32) for c in CFGS}


# ── Lagrangian routing（指定 budget）────────────────────────────
def route_quality(pred_mat, true_mat, budget=BUDGET):
    pts = []
    for lam in LAMBDAS:
        adj    = pred_mat - lam * COSTS[None, :]
        chosen = np.argmax(adj, axis=1)
        pts.append((float(np.mean(COSTS[chosen])),
                    float(np.mean(true_mat[np.arange(len(true_mat)), chosen]))))
    bucket = defaultdict(float)
    for c, q in pts:
        bucket[round(c, 6)] = max(bucket[round(c, 6)], q)
    xs = np.array(sorted(bucket.keys()))
    qs = np.array([bucket[x] for x in xs])
    if budget <= xs[0]:  return float(qs[0])
    if budget >= xs[-1]: return float(qs[-1])
    idx = int(np.searchsorted(xs, budget))
    c0, c1 = xs[idx-1], xs[idx]; q0, q1 = qs[idx-1], qs[idx]
    return float(q0 + (budget - c0) / (c1 - c0) * (q1 - q0))


# ── Ridge fit helper ──────────────────────────────────────────
def fit_ridge_col(Xtr, ytr, Xva, yva, Xte):
    best_mse, best_m = 1e18, None
    for a in [0.1, 1., 10., 100., 1000., 1e4]:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        mse = float(np.mean((m.predict(Xva) - yva) ** 2))
        if mse < best_mse: best_mse = mse; best_m = m
    return best_m.predict(Xte)

def fit_ridge_mat(Xtr, Ytr, Xva, Yva, Xte):
    """fit 3 独立 Ridge，返回 (n_te, 3)"""
    return np.stack([
        fit_ridge_col(Xtr, Ytr[:, j], Xva, Yva[:, j], Xte)
        for j in range(Ytr.shape[1])], axis=1)


# ── 主实验 ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    print("=== §14 Routing Tricks: gm/qwen ===")
    p1 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen_perseed.json"
    p2 = f"{DATA}/geneval2/edit/geneval2_qwen_edit_gen2000_perseed.json"
    i1, s1 = load_scores(p1); i2, s2 = load_scores(p2)
    pool_instrs = i1 + i2
    pool_scores = {c: np.concatenate([s1[c], s2[c]]) for c in CFGS}

    te_instrs, yte_cfg = load_scores(
        f"{DATA}/geneval2/edit/geneval2_qwen_edit_raw_perseed.json")
    n_te = len(te_instrs)
    true_mat = np.stack([yte_cfg[c] for c in CFGS], axis=1)
    oracle_q = route_quality(true_mat, true_mat)
    always50 = float(yte_cfg["50steps"].mean())
    print(f"  oracle={oracle_q:.4f}  50steps={always50:.4f}  n_te={n_te}")

    # CLIP mean-pool
    Fpool, Mpool = TF.clip_token_features(pool_instrs, "exp4_gm_qwen_pool", CLIP_DIR, dev)
    Fte,   Mte   = TF.clip_token_features(te_instrs,   "exp4_gm_qwen_test", CLIP_DIR, dev)
    def mean_pool(F, M):
        m = M[:, :, None]
        return (F * m).sum(1) / np.clip(m.sum(1), 1e-6, None)
    Xpool = mean_pool(Fpool, Mpool)
    Xte   = mean_pool(Fte, Mte)

    rng_fix = random.Random(42)
    all_idx  = list(range(len(pool_instrs))); rng_fix.shuffle(all_idx)
    ext_val  = np.array(all_idx[:N_VAL]); lc_pool = all_idx[N_VAL:]
    Yva_mat  = np.stack([pool_scores[c][ext_val] for c in CFGS], 1)
    Xva_raw  = Xpool[ext_val]

    VARIANTS = ["score", "delta", "delta_costnorm",
                "rank_norm", "rank_zscore",
                "budget_tuned", "offset_calib", "bootstrap"]
    results = {v: [] for v in VARIANTS}

    # 只有一个有效 seed（N_TRAIN=2200=pool 大小），跑 3 次取均值意义不大
    # 改为: seed=0 取全部 2200；seed=1/2 各取 1800 的子集来观察方差
    SEED_SIZES = [(0, 2200), (1, 1800), (2, 1800)]

    for seed, n_tr in SEED_SIZES:
        rng = random.Random(seed)
        sub = np.array(rng.sample(lc_pool, min(n_tr, len(lc_pool))))
        Ytr_mat = np.stack([pool_scores[c][sub] for c in CFGS], 1)   # (n_tr, 3)

        sc = StandardScaler().fit(Xpool[sub])
        Xtr = sc.transform(Xpool[sub])
        Xva = sc.transform(Xva_raw)
        Xte_s = sc.transform(Xte)

        row = []

        # ── 1. score（baseline）──────────────────────────────
        pred_s = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, Xte_s)
        rq = route_quality(pred_s, true_mat)
        results["score"].append(rq); row.append(f"score={rq:.4f}")

        # ── 2. delta ──────────────────────────────────────────
        d1_tr = Ytr_mat[:, 1] - Ytr_mat[:, 0]   # q50 - q10
        d2_tr = Ytr_mat[:, 2] - Ytr_mat[:, 1]   # qtts - q50
        d1_va = Yva_mat[:, 1] - Yva_mat[:, 0]
        d2_va = Yva_mat[:, 2] - Yva_mat[:, 1]
        p_d1 = fit_ridge_col(Xtr, d1_tr, Xva, d1_va, Xte_s)
        p_d2 = fit_ridge_col(Xtr, d2_tr, Xva, d2_va, Xte_s)
        pred_delta = np.stack([-p_d1, np.zeros(n_te), p_d2], 1)
        rq = route_quality(pred_delta, true_mat)
        results["delta"].append(rq); row.append(f"delta={rq:.4f}")

        # ── 3. delta_costnorm ─────────────────────────────────
        # marginal quality gain per unit cost
        dc_10_50 = COSTS[1] - COSTS[0]   # 8.0
        dc_50_tts = COSTS[2] - COSTS[1]  # 20.5
        mn1_tr = d1_tr / dc_10_50;  mn1_va = d1_va / dc_10_50
        mn2_tr = d2_tr / dc_50_tts; mn2_va = d2_va / dc_50_tts
        p_mn1 = fit_ridge_col(Xtr, mn1_tr, Xva, mn1_va, Xte_s)
        p_mn2 = fit_ridge_col(Xtr, mn2_tr, Xva, mn2_va, Xte_s)
        # 路由分：将 marginal value × cost_diff 换回 delta，再构造 [-d1, 0, d2]
        pred_dcn = np.stack([-p_mn1 * dc_10_50, np.zeros(n_te), p_mn2 * dc_50_tts], 1)
        rq = route_quality(pred_dcn, true_mat)
        results["delta_costnorm"].append(rq); row.append(f"delta_cn={rq:.4f}")

        # ── 4. rank_norm ──────────────────────────────────────
        mu_tr = Ytr_mat.mean(1, keepdims=True)
        mu_va = Yva_mat.mean(1, keepdims=True)
        Ytr_rn = Ytr_mat - mu_tr; Yva_rn = Yva_mat - mu_va
        pred_rn = fit_ridge_mat(Xtr, Ytr_rn, Xva, Yva_rn, Xte_s)
        rq = route_quality(pred_rn, true_mat)
        results["rank_norm"].append(rq); row.append(f"rank_norm={rq:.4f}")

        # ── 5. rank_zscore ────────────────────────────────────
        std_tr = Ytr_mat.std(1, keepdims=True) + 1e-6
        std_va = Yva_mat.std(1, keepdims=True) + 1e-6
        Ytr_rz = (Ytr_mat - mu_tr) / std_tr; Yva_rz = (Yva_mat - mu_va) / std_va
        pred_rz = fit_ridge_mat(Xtr, Ytr_rz, Xva, Yva_rz, Xte_s)
        rq = route_quality(pred_rz, true_mat)
        results["rank_zscore"].append(rq); row.append(f"rank_z={rq:.4f}")

        # ── 6. budget_tuned ───────────────────────────────────
        # 在 val 上搜索最优 budget B'，然后用 B' 对 test 做路由
        best_b, best_vq = BUDGET, -1e9
        for b_try in np.arange(6.0, 16.0, 0.5):
            vq = route_quality(
                fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat,
                              sc.transform(Xva_raw)),   # val 上的 pred
                Yva_mat, budget=b_try)
            if vq > best_vq: best_vq = vq; best_b = b_try
        rq = route_quality(pred_s, true_mat, budget=best_b)
        results["budget_tuned"].append(rq)
        row.append(f"budget_tuned={rq:.4f}(B'={best_b:.1f})")

        # ── 7. offset_calib ───────────────────────────────────
        # 在 val 上搜索 per-config 加法偏置 [a0, a1, a2]，a1=0 固定
        # 搜索空间：a0 ∈ [-0.05,0.05], a2 ∈ [-0.05,0.05]，步长 0.005
        pred_va_s = fit_ridge_mat(Xtr, Ytr_mat, Xva, Yva_mat, sc.transform(Xva_raw))
        best_a, best_vq = (0.0, 0.0), -1e9
        for a0 in np.arange(-0.05, 0.055, 0.005):
            for a2 in np.arange(-0.05, 0.055, 0.005):
                offset = np.array([a0, 0.0, a2])
                vq = route_quality(pred_va_s + offset[None, :], Yva_mat)
                if vq > best_vq: best_vq = vq; best_a = (a0, a2)
        a0, a2 = best_a
        offset = np.array([a0, 0.0, a2])
        rq = route_quality(pred_s + offset[None, :], true_mat)
        results["offset_calib"].append(rq)
        row.append(f"offset_calib={rq:.4f}(a=[{a0:.3f},0,{a2:.3f}])")

        # ── 8. bootstrap ensemble ─────────────────────────────
        N_BOOT = 10; preds_boot = []
        rng_b = np.random.RandomState(seed)
        n_sub = int(len(sub) * 0.8)
        for _ in range(N_BOOT):
            idx_b = rng_b.choice(len(sub), n_sub, replace=False)
            sc_b = StandardScaler().fit(Xpool[sub[idx_b]])
            p_b = fit_ridge_mat(
                sc_b.transform(Xpool[sub[idx_b]]), Ytr_mat[idx_b],
                sc_b.transform(Xva_raw), Yva_mat,
                sc_b.transform(Xte))
            preds_boot.append(p_b)
        pred_boot = np.mean(preds_boot, axis=0)
        rq = route_quality(pred_boot, true_mat)
        results["bootstrap"].append(rq); row.append(f"bootstrap={rq:.4f}")

        print(f"  seed={seed}(n={n_tr})  " + "  ".join(row), flush=True)

    print("\n=== 汇总 ===")
    print(f"  oracle={oracle_q:.4f}  50steps={always50:.4f}")
    score_mean = np.mean(results["score"])
    for v in VARIANTS:
        vals = results[v]
        print(f"  {v:<16}: mean={np.mean(vals):.4f}  "
              f"vs_score={np.mean(vals)-score_mean:+.4f}  "
              f"vals={[f'{x:.4f}' for x in vals]}")

    out = os.path.join(HERE, "results", "exp14_routing_tricks.json")
    json.dump({
        "oracle": oracle_q, "always_50steps": always50,
        **{f"{v}_mean": float(np.mean(results[v])) for v in VARIANTS},
        "per_run": {v: [float(x) for x in results[v]] for v in VARIANTS},
    }, open(out, "w"), indent=2)
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
